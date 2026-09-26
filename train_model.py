"""
train_model.py
Trains the LightGBM pairwise entity matching classifier with:
1. Stratified leak-free 60k/20k train/val split across US, India, France, and unknown.
2. 22 Pairwise features including rapidfuzz string similarity and postal/PIN matching.
3. Hard negative mining (4 negatives per positive, 3 per singleton).
4. Global and per-country threshold calibration for Macro F_0.5.
5. Margin-based singleton abstention analysis.
"""

import os
import sys
import time
import random
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

sys.path.append("src")
sys.path.append("amc2026/src")

from features import extract_pair_features, FEATURE_NAMES
from model import train_lgbm_model, find_optimal_threshold_for_f05, save_matcher_model
from evaluate import compute_entity_f05, evaluate_predictions


def load_pool_subset(s2_path: str, s3_path: str, needed_ids: Set[str]) -> Dict[str, dict]:
    """Loads only the candidate records needed from Source 2 and Source 3."""
    pool_dict = {}
    needed = set(needed_ids)
    for path in [s2_path, s3_path]:
        with open(path, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\n").split("\t")
            id_idx = header.index("entity_id")
            country_idx = header.index("country") if "country" in header else header.index("country_extracted")
            name_idx = header.index("business_name_clean")
            addr_idx = header.index("business_address_clean")
            postal_idx = header.index("postal_code") if "postal_code" in header else -1
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[id_idx]
                if eid in needed:
                    pool_dict[eid] = {
                        "entity_id": eid,
                        "country": parts[country_idx] if len(parts) > country_idx else "",
                        "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                        "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                        "postal_code": parts[postal_idx] if postal_idx >= 0 and len(parts) > postal_idx else "",
                    }
                    needed.discard(eid)
                    if not needed:
                        break
    return pool_dict


def main():
    print("=" * 60)
    print("STEP 1: LOADING GROUND TRUTH & STRATIFIED CANDIDATES")
    print("=" * 60)

    gt_file = "data/raw/train_ground_truth.tsv"
    cand_file = "outputs/train_candidates_adaptive.tsv"
    s1_file = "data/processed/source1_clean.tsv"
    s2_file = "data/processed/source2_clean.tsv"
    s3_file = "data/processed/source3_clean.tsv"

    print(f"Loading candidates from {cand_file}...")
    s1_candidates: Dict[str, List[str]] = {}
    needed_cand_ids = set()
    sample_s1_list = []

    with open(cand_file, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1 = parts[0]
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            sample_s1_list.append(s1)
            s1_candidates[s1] = cands
            for item in cands:
                cid = item.split(":", 1)[0] if ":" in item else item
                needed_cand_ids.add(cid)

    sample_s1_set = set(sample_s1_list)
    print(f"Loaded {len(sample_s1_set):,} Source 1 entities from candidate set.")

    # Load ground truth for sampled entities
    print("Loading ground truth for sampled entities...")
    gt_mapping: Dict[str, Set[str]] = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            s1 = parts[0]
            if s1 in sample_s1_set:
                if len(parts) > 1 and parts[1] and parts[1] != "nan":
                    trues = {x.strip() for x in parts[1].split(",") if x.strip()}
                    gt_mapping[s1] = trues
                    needed_cand_ids.update(trues)
                else:
                    gt_mapping[s1] = set()

    for s1 in sample_s1_set:
        if s1 not in gt_mapping:
            gt_mapping[s1] = set()

    # Load Source 1 records for sampled entities
    print("Loading Source 1 records for sampled entities...")
    s1_dict: Dict[str, dict] = {}
    with open(s1_file, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        country_idx = header.index("country") if "country" in header else header.index("country_extracted")
        name_idx = header.index("business_name_clean")
        addr_idx = header.index("business_address_clean")
        postal_idx = header.index("postal_code") if "postal_code" in header else -1
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1 = parts[id_idx]
            if s1 in sample_s1_set:
                s1_dict[s1] = {
                    "entity_id": s1,
                    "country": parts[country_idx] if len(parts) > country_idx and parts[country_idx] else "unknown",
                    "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                    "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                    "postal_code": parts[postal_idx] if postal_idx >= 0 and len(parts) > postal_idx else "",
                }
                if len(s1_dict) == len(sample_s1_set):
                    break

    print(f"Total distinct candidate pool records needed: {len(needed_cand_ids):,}")
    print("Loading candidate pool records from S2 and S3...")
    t0 = time.time()
    pool_dict = load_pool_subset(s2_file, s3_file, needed_cand_ids)
    print(f"Loaded {len(pool_dict):,} candidate pool records in {time.time() - t0:.2f}s.")

    # Stratified Train/Val Split (75% / 25%) by Country
    print("\nPerforming Stratified 75/25 Train/Val Split by Country...")
    country_to_s1 = defaultdict(list)
    for s1_id in sample_s1_list:
        c = s1_dict[s1_id]["country"]
        country_to_s1[c].append(s1_id)

    train_s1_ids = []
    val_s1_ids = []
    random.seed(42)

    for country, s1_ids in country_to_s1.items():
        random.shuffle(s1_ids)
        n_train = int(len(s1_ids) * 0.75)
        train_s1_ids.extend(s1_ids[:n_train])
        val_s1_ids.extend(s1_ids[n_train:])
        print(f"  {country:10s}: {n_train:,} Train, {len(s1_ids) - n_train:,} Val (Total {len(s1_ids):,})")

    random.shuffle(train_s1_ids)
    random.shuffle(val_s1_ids)
    print(f"\nFinal Split: {len(train_s1_ids):,} Train entities, {len(val_s1_ids):,} Validation entities.")

    # ----------------------------------------------------------------------
    # Step 2: Build Training Feature Matrix (Pairs with Hard Negatives)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: EXTRACTING PAIRWISE FEATURES FOR TRAINING (22 FEATURES)")
    print("=" * 60)

    t0 = time.time()
    X_train_list = []
    y_train_list = []

    MAX_NEG_PER_POS = 4
    MAX_NEG_SINGLETON = 3

    for s1_id in train_s1_ids:
        s1_row = s1_dict[s1_id]
        true_ids = gt_mapping.get(s1_id, set())
        cands = s1_candidates.get(s1_id, [])

        pos_count = 0
        neg_count = 0
        # Asymmetric negative mining: India has dense franchise lookalikes requiring more hard negatives
        is_india = s1_row.get("country") == "India"
        max_neg_per_pos = 7 if is_india else 4
        max_neg_singleton = 5 if is_india else 3
        max_neg = max_neg_per_pos * len(true_ids) if true_ids else max_neg_singleton

        for rank, item in enumerate(cands):
            if ":" in item:
                cid, count_str = item.split(":", 1)
                count = float(count_str)
            else:
                cid = item
                count = 1.0

            cand_row = pool_dict.get(cid)
            if not cand_row:
                continue

            is_match = 1 if cid in true_ids else 0
            if is_match:
                feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=count)
                X_train_list.append([feats[f] for f in FEATURE_NAMES])
                y_train_list.append(1)
                pos_count += 1
            elif neg_count < max_neg:
                feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=count)
                X_train_list.append([feats[f] for f in FEATURE_NAMES])
                y_train_list.append(0)
                neg_count += 1

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"Extracted {len(X_train):,} training pairs ({sum(y_train):,} positive, {len(y_train) - sum(y_train):,} negative) in {time.time() - t0:.2f}s.")

    # Extract Validation pairs
    val_s1_candidates: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for s1_id in val_s1_ids:
        s1_row = s1_dict[s1_id]
        cands = s1_candidates.get(s1_id, [])
        pair_list = []
        for rank, item in enumerate(cands):
            if ":" in item:
                cid, count_str = item.split(":", 1)
                count = float(count_str)
            else:
                cid = item
                count = 1.0

            cand_row = pool_dict.get(cid)
            if cand_row:
                feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=count)
                pair_list.append((cid, np.array([feats[f] for f in FEATURE_NAMES], dtype=np.float32)))
        val_s1_candidates[s1_id] = pair_list

    # ----------------------------------------------------------------------
    # Step 3: Train LightGBM Model
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING LIGHTGBM CLASSIFIER")
    print("=" * 60)

    t0 = time.time()
    clf = train_lgbm_model(X_train, y_train, feature_names=FEATURE_NAMES)
    print(f"Trained LightGBM model in {time.time() - t0:.2f}s.")

    # Feature importances
    print("\nFeature Importances (Gain):")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"  {FEATURE_NAMES[i]:25s}: {importances[i]:.1f}")

    # Precompute probabilities for validation set
    print("\nPrecomputing validation prediction probabilities...")
    s1_val_probs: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id in val_s1_ids:
        cands = val_s1_candidates.get(s1_id, [])
        if not cands:
            s1_val_probs[s1_id] = []
            continue
        X_val_entity = np.array([c[1] for c in cands])
        probs = clf.predict_proba(X_val_entity)[:, 1]
        s1_val_probs[s1_id] = list(zip([c[0] for c in cands], probs))

    # ----------------------------------------------------------------------
    # Step 4A: Global Threshold Optimization for Macro F_0.5
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4A: GLOBAL DECISION THRESHOLD OPTIMIZATION (MACRO F_0.5)")
    print("=" * 60)

    def eval_global_threshold(thresh: float, ids: List[str]) -> float:
        total_f05 = 0.0
        for s1 in ids:
            gt_set = gt_mapping.get(s1, set())
            cands = s1_val_probs.get(s1, [])
            preds = {cid for cid, p in cands if p >= thresh}
            total_f05 += compute_entity_f05(gt_set, preds)
        return total_f05 / len(ids) if ids else 0.0

    # Grid search for global threshold
    best_global_th = 0.5
    best_global_f05 = -1.0
    for th in np.linspace(0.10, 0.95, 86):
        score = eval_global_threshold(th, val_s1_ids)
        if score > best_global_f05:
            best_global_f05 = score
            best_global_th = round(float(th), 4)

    # Refine around peak
    for th in np.linspace(max(0.10, best_global_th - 0.04), min(0.95, best_global_th + 0.04), 41):
        score = eval_global_threshold(th, val_s1_ids)
        if score > best_global_f05:
            best_global_f05 = score
            best_global_th = round(float(th), 4)

    print(f">>> Best Global Decision Threshold: {best_global_th:.4f} (Macro F_0.5 = {best_global_f05:.4f}) <<<")

    # ----------------------------------------------------------------------
    # Step 4B: Country-Specific Threshold Calibration
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4B: COUNTRY-SPECIFIC THRESHOLD CALIBRATION")
    print("=" * 60)

    country_thresholds = {}
    country_val_scores = {}
    unique_countries = sorted(list(country_to_s1.keys()))

    for c in unique_countries:
        c_val_ids = [s1 for s1 in val_s1_ids if s1_dict[s1]["country"] == c]
        if not c_val_ids:
            country_thresholds[c] = best_global_th
            continue

        best_c_th = best_global_th
        best_c_f05 = -1.0

        for th in np.linspace(0.10, 0.95, 86):
            score = eval_global_threshold(th, c_val_ids)
            if score > best_c_f05:
                best_c_f05 = score
                best_c_th = round(float(th), 4)

        for th in np.linspace(max(0.10, best_c_th - 0.04), min(0.95, best_c_th + 0.04), 41):
            score = eval_global_threshold(th, c_val_ids)
            if score > best_c_f05:
                best_c_f05 = score
                best_c_th = round(float(th), 4)

        country_thresholds[c] = best_c_th
        country_val_scores[c] = best_c_f05
        global_sub_score = eval_global_threshold(best_global_th, c_val_ids)
        print(f"  {c:10s} (N={len(c_val_ids):,}): Optimal tau = {best_c_th:.4f} (F_0.5 = {best_c_f05:.4f}) | At global tau: {global_sub_score:.4f}")

    # Evaluate overall validation score with country-specific thresholds
    country_pred_mapping = {}
    for s1 in val_s1_ids:
        c = s1_dict[s1]["country"]
        th = country_thresholds.get(c, best_global_th)
        cands = s1_val_probs.get(s1, [])
        country_pred_mapping[s1] = {cid for cid, p in cands if p >= th}

    detailed_country = evaluate_predictions(gt_mapping, country_pred_mapping, required_s1_ids=val_s1_ids)
    print(f"\n>>> Combined Country-Specific Macro F_0.5: {detailed_country['macro_f05']:.4f} <<<")
    print(f"  Delta over Global Threshold: {detailed_country['macro_f05'] - best_global_f05:+.4f}")

    # ----------------------------------------------------------------------
    # Step 4C: Margin-Based Singleton Abstention Analysis
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4C: MARGIN-BASED SINGLETON ABSTENTION ANALYSIS")
    print("=" * 60)

    best_margin_f05 = detailed_country["macro_f05"]
    best_abstain_th = 0.0

    # Grid search for top-probability abstention threshold (tau_abstain >= tau_country)
    for abstain_offset in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        margin_pred_mapping = {}
        for s1 in val_s1_ids:
            c = s1_dict[s1]["country"]
            th = country_thresholds.get(c, best_global_th)
            cands = s1_val_probs.get(s1, [])
            if not cands:
                margin_pred_mapping[s1] = set()
                continue

            max_p = max(p for _, p in cands)
            if max_p < th + abstain_offset:
                # Top candidate is ambiguous / insufficient confidence -> abstain as singleton
                margin_pred_mapping[s1] = set()
            else:
                margin_pred_mapping[s1] = {cid for cid, p in cands if p >= th}

        m_res = evaluate_predictions(gt_mapping, margin_pred_mapping, required_s1_ids=val_s1_ids)
        print(f"  Abstain Offset +{abstain_offset:.2f}: Macro F_0.5 = {m_res['macro_f05']:.4f} | Prec: {m_res['mean_precision']:.4f} | Rec: {m_res['mean_recall']:.4f} | Singleton Acc: {m_res['singleton_accuracy']:.4f}")
        if m_res["macro_f05"] > best_margin_f05:
            best_margin_f05 = m_res["macro_f05"]
            best_abstain_th = abstain_offset

    print(f"\n>>> Best Margin Abstention Offset: +{best_abstain_th:.2f} (Macro F_0.5 = {best_margin_f05:.4f}) <<<")

    # ----------------------------------------------------------------------
    # Final Validation Breakdown by Country
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL DETAILED VALIDATION BREAKDOWN (PER-COUNTRY)")
    print("=" * 60)

    # Compute final predictions using calibrated country thresholds + best margin abstention
    final_preds = {}
    for s1 in val_s1_ids:
        c = s1_dict[s1]["country"]
        th = country_thresholds.get(c, best_global_th)
        cands = s1_val_probs.get(s1, [])
        if not cands:
            final_preds[s1] = set()
            continue
        max_p = max(p for _, p in cands)
        if max_p < th + best_abstain_th:
            final_preds[s1] = set()
        else:
            final_preds[s1] = {cid for cid, p in cands if p >= th}

    # Print overall
    final_overall = evaluate_predictions(gt_mapping, final_preds, required_s1_ids=val_s1_ids)
    print(f"OVERALL (N={len(val_s1_ids):,}):")
    print(f"  Macro F_0.5:        {final_overall['macro_f05']:.4f}")
    print(f"  Mean Precision:     {final_overall['mean_precision']:.4f}")
    print(f"  Mean Recall:        {final_overall['mean_recall']:.4f}")
    print(f"  Singleton Accuracy: {final_overall['singleton_accuracy']:.4f}")
    print(f"  Singletons Count:   {final_overall['n_singletons']:,}")
    print(f"  Matched Count:      {final_overall['n_with_matches']:,}")

    # Print per country
    for c in unique_countries:
        c_val_ids = [s1 for s1 in val_s1_ids if s1_dict[s1]["country"] == c]
        c_res = evaluate_predictions(gt_mapping, final_preds, required_s1_ids=c_val_ids)
        print(f"\nCOUNTRY: {c} (N={len(c_val_ids):,}, tau={country_thresholds[c]:.4f}):")
        print(f"  Macro F_0.5:        {c_res['macro_f05']:.4f}")
        print(f"  Mean Precision:     {c_res['mean_precision']:.4f}")
        print(f"  Mean Recall:        {c_res['mean_recall']:.4f}")
        print(f"  Singleton Accuracy: {c_res['singleton_accuracy']:.4f}")

    # Save model and thresholds
    if "France" not in country_thresholds:
        country_thresholds["France"] = best_global_th

    save_path = "models/matcher_lgbm.pkl"
    save_matcher_model(
        clf=clf,
        threshold=best_global_th,
        filepath=save_path,
        country_thresholds=country_thresholds,
        margin_params={"abstain_offset": best_abstain_th}
    )
    save_matcher_model(
        clf=clf,
        threshold=best_global_th,
        filepath="amc2026/models/matcher_lgbm.pkl",
        country_thresholds=country_thresholds,
        margin_params={"abstain_offset": best_abstain_th}
    )
    print(f"\nModel and calibrated thresholds saved to {save_path} and amc2026/{save_path}!")


if __name__ == "__main__":
    main()
