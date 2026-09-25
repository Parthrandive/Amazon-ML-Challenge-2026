"""
train_model.py
Phase 5: Model Training & Threshold Optimization for Amazon ML Challenge 2026.
Trains LightGBM classifier on hard-negative candidate pairs and optimizes decision
threshold directly for Macro F_0.5 (weighting precision 2x over recall).
"""

import os
import sys
import time
import pickle
import random
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("amc2026/src"))
from features import extract_pair_features, FEATURE_NAMES
from model import train_lgbm_model, find_optimal_threshold_for_f05, find_optimal_threshold_two_phase, save_matcher_model
from evaluate import compute_entity_f05, evaluate_predictions
from blocking import get_tight_blocking_keys

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("amc2026/models", exist_ok=True)


def load_pool_subset(s2_path: str, s3_path: str, needed_ids: Set[str]) -> Dict[str, dict]:
    """Loads only the candidate records needed for training using fast line streaming."""
    pool_dict = {}
    needed = set(needed_ids)
    for path in [s2_path, s3_path]:
        with open(path, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\n").split("\t")
            id_idx = header.index("entity_id")
            country_idx = header.index("country")
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
                        "postal_code": parts[postal_idx] if (postal_idx != -1 and len(parts) > postal_idx) else "",
                    }
                    needed.discard(eid)
                    if not needed:
                        break
    return pool_dict


def main():
    print("=" * 60)
    print("STEP 1: LOADING GROUND TRUTH & CANDIDATE PAIRS FOR TRAINING")
    print("=" * 60)

    gt_file = "data/raw/train_ground_truth.tsv"
    cand_file = "outputs/candidate_pairs.tsv"
    s1_file = "data/processed/source1_clean.tsv"
    s2_file = "data/processed/source2_clean.tsv"
    s3_file = "data/processed/source3_clean.tsv"

    SAMPLE_ENTITIES = 80000
    TRAIN_RATIO = 0.75

    print(f"Sampling {SAMPLE_ENTITIES:,} entities from candidate pairs...")
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
            needed_cand_ids.update(cands)
            if len(sample_s1_list) >= SAMPLE_ENTITIES:
                break

    sample_s1_set = set(sample_s1_list)

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
                    trues = set(parts[1].split(","))
                    gt_mapping[s1] = trues
                    needed_cand_ids.update(trues)
                else:
                    gt_mapping[s1] = set()

    # Ensure all sampled entities have an entry in gt_mapping
    for s1 in sample_s1_set:
        if s1 not in gt_mapping:
            gt_mapping[s1] = set()

    # Load Source 1 records for sampled entities
    print("Loading Source 1 records for sampled entities...")
    s1_dict: Dict[str, dict] = {}
    with open(s1_file, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        country_idx = header.index("country")
        name_idx = header.index("business_name_clean")
        addr_idx = header.index("business_address_clean")
        postal_idx = header.index("postal_code") if "postal_code" in header else -1
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1 = parts[id_idx]
            if s1 in sample_s1_set:
                s1_dict[s1] = {
                    "entity_id": s1,
                    "country": parts[country_idx] if len(parts) > country_idx else "",
                    "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                    "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                    "postal_code": parts[postal_idx] if (postal_idx != -1 and len(parts) > postal_idx) else "",
                }
                if len(s1_dict) == len(sample_s1_set):
                    break

    print(f"Total distinct candidate pool records needed: {len(needed_cand_ids):,}")
    print("Loading candidate pool records from S2 and S3...")
    t0 = time.time()
    pool_dict = load_pool_subset(s2_file, s3_file, needed_cand_ids)
    print(f"Loaded {len(pool_dict):,} candidate pool records in {time.time() - t0:.2f}s.")

    # Precompute blocking keys for fast, active shared_key_count calculation
    print("Precomputing multi-pass blocking keys for active shared_key_count...")
    t_keys = time.time()
    s1_keys_cache = {
        s1_id: set(get_tight_blocking_keys(row["country"], row["business_name_clean"], row["business_address_clean"], row.get("postal_code")))
        for s1_id, row in s1_dict.items()
    }
    cand_keys_cache = {
        cid: set(get_tight_blocking_keys(row["country"], row["business_name_clean"], row["business_address_clean"], row.get("postal_code")))
        for cid, row in pool_dict.items()
    }
    print(f"Cached blocking keys for {len(s1_keys_cache):,} S1 and {len(cand_keys_cache):,} candidates in {time.time() - t_keys:.2f}s.")

    # Split into Train and Validation entities
    all_s1_list = list(s1_dict.keys())
    random.shuffle(all_s1_list)
    n_train = int(len(all_s1_list) * TRAIN_RATIO)
    train_s1_ids = all_s1_list[:n_train]
    val_s1_ids = all_s1_list[n_train:]

    print(f"\nEntity Split: {len(train_s1_ids):,} Train entities, {len(val_s1_ids):,} Validation entities.")

    # ----------------------------------------------------------------------
    # Step 2: Build Training Feature Matrix (Pairs with Hard Negatives)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: EXTRACTING PAIRWISE FEATURES FOR TRAINING")
    print("=" * 60)

    t0 = time.time()
    X_train_list = []
    y_train_list = []

    MAX_NEG_PER_POS = 4
    MAX_NEG_SINGLETON = 3

    for s1_id in train_s1_ids:
        s1_row = s1_dict[s1_id]
        s1_keys = s1_keys_cache.get(s1_id, set())
        true_ids = gt_mapping.get(s1_id, set())
        cands = s1_candidates.get(s1_id, [])

        pos_count = 0
        neg_count = 0
        max_neg = MAX_NEG_PER_POS * len(true_ids) if true_ids else MAX_NEG_SINGLETON

        for rank, cid in enumerate(cands):
            cand_row = pool_dict.get(cid)
            if not cand_row:
                continue

            c_keys = cand_keys_cache.get(cid, set())
            shared_keys = len(s1_keys & c_keys) if (s1_keys and c_keys) else 1

            is_match = 1 if cid in true_ids else 0
            if is_match:
                feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=shared_keys)
                X_train_list.append([feats[f] for f in FEATURE_NAMES])
                y_train_list.append(1)
                pos_count += 1
            elif neg_count < max_neg:
                feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=shared_keys)
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
        s1_keys = s1_keys_cache.get(s1_id, set())
        cands = s1_candidates.get(s1_id, [])
        pair_list = []
        for rank, cid in enumerate(cands):
            cand_row = pool_dict.get(cid)
            if cand_row:
                c_keys = cand_keys_cache.get(cid, set())
                shared_keys = len(s1_keys & c_keys) if (s1_keys and c_keys) else 1
                feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=shared_keys)
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
    print("\nTop Feature Importances (Gain):")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:10]:
        print(f"  {FEATURE_NAMES[i]:25s}: {importances[i]:.1f}")

    # ----------------------------------------------------------------------
    # Step 4: Two-Phase Optimization for Macro F_0.5
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: TWO-PHASE THRESHOLD SEARCH FOR MACRO F_0.5")
    print("=" * 60)

    best_thresh, best_f05, history = find_optimal_threshold_two_phase(
        clf, val_s1_ids, val_s1_candidates, gt_mapping,
        coarse_range=np.arange(0.35, 0.90, 0.05),
        fine_radius=0.04,
        fine_step=0.005
    )

    print("\nTop Threshold Search Results around Peak:")
    sorted_th = sorted(history.keys(), key=lambda t: history[t], reverse=True)
    for th in sorted_th[:10]:
        marker = " <--- OPTIMAL" if abs(th - best_thresh) < 1e-4 else ""
        print(f"  Threshold {th:.3f}: Macro F_0.5 = {history[th]:.4f}{marker}")

    print(f"\n>>> Best Decision Threshold: {best_thresh:.3f} (Macro F_0.5 = {best_f05:.4f}) <<<")

    # Evaluate detailed metrics at best threshold
    val_pred_mapping = {}
    for s1_id in val_s1_ids:
        cands = val_s1_candidates.get(s1_id, [])
        if cands:
            X = np.array([c[1] for c in cands])
            probs = clf.predict_proba(X)[:, 1]
            matches = {cid for (cid, _), p in zip(cands, probs) if p >= best_thresh}
            val_pred_mapping[s1_id] = matches
        else:
            val_pred_mapping[s1_id] = set()

    detailed_metrics = evaluate_predictions(gt_mapping, val_pred_mapping, required_s1_ids=val_s1_ids)
    print("\nValidation Performance Summary:")
    print(f"  Macro F_0.5 Score:      {detailed_metrics['macro_f05']:.4f}")
    print(f"  Mean Precision:         {detailed_metrics['mean_precision']:.4f}")
    print(f"  Mean Recall:            {detailed_metrics['mean_recall']:.4f}")
    print(f"  Singleton Accuracy:     {detailed_metrics['singleton_accuracy']:.4f}")
    print(f"  Total Val Entities:     {detailed_metrics['n_total']:,}")
    print(f"  Singletons:             {detailed_metrics['n_singletons']:,}")
    print(f"  Entities with Matches:  {detailed_metrics['n_with_matches']:,}")

    # Save model
    save_path = "models/matcher_lgbm.pkl"
    save_matcher_model(clf, best_thresh, save_path)
    save_matcher_model(clf, best_thresh, "amc2026/models/matcher_lgbm.pkl")
    print(f"\nModel and threshold saved to {save_path} and amc2026/{save_path}")


if __name__ == "__main__":
    main()
