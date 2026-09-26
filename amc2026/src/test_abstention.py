"""
test_abstention.py
Diagnostic tool to inspect validation entity score distributions (singletons vs matches)
and evaluate candidate margin abstention strategies to restore singleton accuracy.
"""

import sys
import pickle
import random
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.append("amc2026/src")
from evaluate import compute_entity_f05, evaluate_predictions
from features import extract_pair_features, FEATURE_NAMES

def main():
    print("Loading model and validation data...")
    with open("models/matcher_lgbm.pkl", "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["model"]
    country_thresholds = bundle["country_thresholds"]
    global_th = bundle["threshold"]
    print("Country thresholds:", country_thresholds)

    cand_file = "outputs/train_candidates_adaptive.tsv"
    gt_file = "data/raw/train_ground_truth.tsv"
    s1_file = "data/processed/source1_clean.tsv"
    s2_file = "data/processed/source2_clean.tsv"
    s3_file = "data/processed/source3_clean.tsv"

    # 1. Load ground truth
    gt_mapping = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1] and parts[1] != "nan":
                gt_mapping[parts[0]] = {x.strip() for x in parts[1].split(",") if x.strip()}
            else:
                gt_mapping[parts[0]] = set()

    # 2. Load candidate pairs
    s1_candidates = {}
    sample_s1_list = []
    needed_cand_ids = set()
    with open(cand_file, "r", encoding="utf-8") as f:
        next(f)
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

    # 3. Load S1
    s1_dict = {}
    with open(s1_file, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        country_idx = header.index("country") if "country" in header else header.index("country_extracted")
        name_idx = header.index("business_name_clean")
        addr_idx = header.index("business_address_clean")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1 = parts[id_idx]
            if s1 in sample_s1_set:
                s1_dict[s1] = {
                    "entity_id": s1,
                    "country": parts[country_idx] if len(parts) > country_idx and parts[country_idx] else "unknown",
                    "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                    "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                }

    # 4. Stratified Split (identical seed 42)
    country_to_s1 = defaultdict(list)
    for s1_id in sample_s1_list:
        c = s1_dict[s1_id]["country"]
        country_to_s1[c].append(s1_id)

    val_s1_ids = []
    random.seed(42)
    for country, s1_ids in country_to_s1.items():
        random.shuffle(s1_ids)
        n_train = int(len(s1_ids) * 0.75)
        val_s1_ids.extend(s1_ids[n_train:])
    random.shuffle(val_s1_ids)
    print(f"Loaded {len(val_s1_ids):,} validation entities.")

    val_cand_ids = set()
    for s1 in val_s1_ids:
        for item in s1_candidates.get(s1, []):
            cid = item.split(":", 1)[0] if ":" in item else item
            val_cand_ids.add(cid)

    # Load needed pool records
    pool_dict = {}
    for path in [s2_file, s3_file]:
        with open(path, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\n").split("\t")
            id_idx = header.index("entity_id")
            country_idx = header.index("country") if "country" in header else header.index("country_extracted")
            name_idx = header.index("business_name_clean")
            addr_idx = header.index("business_address_clean")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[id_idx]
                if eid in val_cand_ids:
                    pool_dict[eid] = {
                        "entity_id": eid,
                        "country": parts[country_idx] if len(parts) > country_idx else "",
                        "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                        "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                    }
                    val_cand_ids.discard(eid)
                    if not val_cand_ids:
                        break

    print(f"Loaded {len(pool_dict):,} candidate pool records for validation.")

    # 5. Predict probabilities for validation
    print("Scoring validation candidates with LightGBM...")
    val_entity_probs = {}
    for s1_id in val_s1_ids:
        s1_row = s1_dict[s1_id]
        cands = s1_candidates.get(s1_id, [])
        if not cands:
            val_entity_probs[s1_id] = []
            continue
        feats_list = []
        cand_id_list = []
        for rank, item in enumerate(cands):
            cid = item.split(":", 1)[0] if ":" in item else item
            count = float(item.split(":", 1)[1]) if ":" in item else 1.0
            cand_row = pool_dict.get(cid)
            if cand_row:
                feat = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=count)
                feats_list.append([feat[f] for f in FEATURE_NAMES])
                cand_id_list.append(cid)
        if feats_list:
            probs = clf.predict_proba(np.array(feats_list, dtype=np.float32))[:, 1]
            # Sort descending by probability
            sorted_pairs = sorted(zip(cand_id_list, probs), key=lambda x: x[1], reverse=True)
            val_entity_probs[s1_id] = sorted_pairs
        else:
            val_entity_probs[s1_id] = []

    print("Scoring complete. Analyzing singletons vs matched entities...")

    # Diagnostic stats on singletons vs matched
    singleton_max_probs = []
    matched_max_probs = []

    for s1 in val_s1_ids:
        is_singleton = len(gt_mapping.get(s1, set())) == 0
        pairs = val_entity_probs.get(s1, [])
        p_max = pairs[0][1] if pairs else 0.0
        if is_singleton:
            singleton_max_probs.append(p_max)
        else:
            matched_max_probs.append(p_max)

    print(f"\n--- SCORE DISTRIBUTION ANALYSIS ---")
    print(f"Singletons (N={len(singleton_max_probs)}):")
    print(f"  p50: {np.percentile(singleton_max_probs, 50):.4f}, p75: {np.percentile(singleton_max_probs, 75):.4f}, p90: {np.percentile(singleton_max_probs, 90):.4f}, p95: {np.percentile(singleton_max_probs, 95):.4f}, p99: {np.percentile(singleton_max_probs, 99):.4f}")
    print(f"Matched Entities (N={len(matched_max_probs)}):")
    print(f"  p10: {np.percentile(matched_max_probs, 10):.4f}, p25: {np.percentile(matched_max_probs, 25):.4f}, p50: {np.percentile(matched_max_probs, 50):.4f}, p75: {np.percentile(matched_max_probs, 75):.4f}, p90: {np.percentile(matched_max_probs, 90):.4f}")

    # Baseline evaluation without margin abstention
    baseline_preds = {}
    for s1 in val_s1_ids:
        c = s1_dict[s1]["country"]
        th = country_thresholds.get(c, global_th)
        pairs = val_entity_probs.get(s1, [])
        baseline_preds[s1] = {cid for cid, p in pairs if p >= th}

    base_res = evaluate_predictions(gt_mapping, baseline_preds, required_s1_ids=val_s1_ids)
    print(f"\nBASELINE (No margin abstention):")
    print(f"  Macro F_0.5:        {base_res['macro_f05']:.4f}")
    print(f"  Precision:          {base_res['mean_precision']:.4f}")
    print(f"  Recall:             {base_res['mean_recall']:.4f}")
    print(f"  Singleton Accuracy: {base_res['singleton_accuracy']:.4f}")

    # Let's test various margin abstention strategies!
    print("\n" + "=" * 60)
    print("EVALUATING MARGIN ABSTENTION STRATEGIES")
    print("=" * 60)

    # Strategy 1: Top-1 score gap over threshold (Absolute Hurdle on top candidate)
    print("\n--- Strategy 1: Absolute Top Score Hurdle (top candidate must reach tau + offset) ---")
    for offset in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
        preds = {}
        for s1 in val_s1_ids:
            c = s1_dict[s1]["country"]
            th = country_thresholds.get(c, global_th)
            pairs = val_entity_probs.get(s1, [])
            if not pairs or pairs[0][1] < th + offset:
                preds[s1] = set()
            else:
                preds[s1] = {cid for cid, p in pairs if p >= th}
        r = evaluate_predictions(gt_mapping, preds, required_s1_ids=val_s1_ids)
        print(f"  Hurdle +{offset:.2f}: F0.5={r['macro_f05']:.4f} | Prec={r['mean_precision']:.4f} | Rec={r['mean_recall']:.4f} | SingletonAcc={r['singleton_accuracy']:.4f}")

    # Strategy 2: Relative Gap (p1 - p2 >= margin when p1 is in the marginal zone)
    print("\n--- Strategy 2: Relative Margin (if top candidate is marginal, require p1 - p2 >= gap) ---")
    for marginal_zone in [0.05, 0.10, 0.15]:
        for gap in [0.02, 0.05, 0.08, 0.10]:
            preds = {}
            for s1 in val_s1_ids:
                c = s1_dict[s1]["country"]
                th = country_thresholds.get(c, global_th)
                pairs = val_entity_probs.get(s1, [])
                if not pairs:
                    preds[s1] = set()
                    continue
                p1 = pairs[0][1]
                p2 = pairs[1][1] if len(pairs) > 1 else 0.0
                
                # If p1 is above threshold but inside marginal zone [th, th + marginal_zone]
                # and the runner-up is too close (p1 - p2 < gap), abstain!
                if p1 >= th and (p1 < th + marginal_zone) and (p1 - p2 < gap):
                    preds[s1] = set()
                else:
                    preds[s1] = {cid for cid, p in pairs if p >= th}
            r = evaluate_predictions(gt_mapping, preds, required_s1_ids=val_s1_ids)
            print(f"  Zone={marginal_zone:.2f}, Gap={gap:.2f}: F0.5={r['macro_f05']:.4f} | Prec={r['mean_precision']:.4f} | Rec={r['mean_recall']:.4f} | SingletonAcc={r['singleton_accuracy']:.4f}")

    # Strategy 3: Pure Relative Gap between Candidates (Competitor Abstention)
    print("\n--- Strategy 3: Competitor Gap (abstain if p1 >= tau but p1 - p2 < gap) ---")
    for gap in [0.01, 0.02, 0.03, 0.05]:
        preds = {}
        for s1 in val_s1_ids:
            c = s1_dict[s1]["country"]
            th = country_thresholds.get(c, global_th)
            pairs = val_entity_probs.get(s1, [])
            if not pairs:
                preds[s1] = set()
                continue
            p1 = pairs[0][1]
            p2 = pairs[1][1] if len(pairs) > 1 else 0.0
            
            if p1 >= th and len(pairs) > 1 and (p1 - p2 < gap):
                preds[s1] = set()
            else:
                preds[s1] = {cid for cid, p in pairs if p >= th}
        r = evaluate_predictions(gt_mapping, preds, required_s1_ids=val_s1_ids)
        print(f"  Competitor Gap={gap:.2f}: F0.5={r['macro_f05']:.4f} | Prec={r['mean_precision']:.4f} | Rec={r['mean_recall']:.4f} | SingletonAcc={r['singleton_accuracy']:.4f}")

    # Strategy 4: Per-Country Singleton Tuning
    print("\n--- Strategy 4: Per-Country Threshold & Singleton Calibration ---")
    us_val = [s for s in val_s1_ids if s1_dict[s]["country"] == "US"]
    in_val = [s for s in val_s1_ids if s1_dict[s]["country"] == "India"]

    for us_th in [0.82, 0.844, 0.86, 0.88, 0.90]:
        for us_gap in [0.0, 0.03, 0.05]:
            preds = {}
            for s1 in us_val:
                pairs = val_entity_probs.get(s1, [])
                if not pairs:
                    preds[s1] = set()
                    continue
                p1 = pairs[0][1]
                p2 = pairs[1][1] if len(pairs) > 1 else 0.0
                if p1 >= us_th and (us_gap == 0.0 or p1 - p2 >= us_gap):
                    preds[s1] = {cid for cid, p in pairs if p >= us_th}
                else:
                    preds[s1] = set()
            r = evaluate_predictions(gt_mapping, preds, required_s1_ids=us_val)
            print(f"  US th={us_th:.3f}, gap={us_gap:.2f}: F0.5={r['macro_f05']:.4f} | Prec={r['mean_precision']:.4f} | Rec={r['mean_recall']:.4f} | SingletonAcc={r['singleton_accuracy']:.4f}")

if __name__ == "__main__":
    main()
