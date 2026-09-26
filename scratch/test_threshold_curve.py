"""
test_threshold_curve.py
Diagnose the full probability threshold landscape from 0.10 to 0.99
using count-based endpoints to prevent floating point accumulation bugs.
"""

import os
import sys
import time
import pickle
import random
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append("src")
sys.path.append("amc2026/src")
from features import extract_pair_features, FEATURE_NAMES
from evaluate import compute_entity_f05, evaluate_predictions
from model import load_matcher_model

random.seed(42)
np.random.seed(42)

def main():
    print("=" * 60)
    print("DIAGNOSTIC: EVALUATING FULL THRESHOLD LANDSCAPE (0.10 - 0.99)")
    print("=" * 60)

    model_path = "models/matcher_lgbm.pkl"
    print(f"Loading trained model from {model_path}...")
    clf, orig_thresh = load_matcher_model(model_path)
    print(f"Loaded model. Previous threshold was: {orig_thresh:.3f}")

    # Load 20,000 validation entities
    cand_file = "outputs/candidate_pairs.tsv"
    gt_file = "data/raw/train_ground_truth.tsv"
    s1_file = "data/processed/source1_clean.tsv"
    s2_file = "data/processed/source2_clean.tsv"
    s3_file = "data/processed/source3_clean.tsv"

    SAMPLE_ENTITIES = 40000
    VAL_ENTITIES = 15000

    print(f"Sampling {VAL_ENTITIES:,} entities for pure validation sweep...")
    val_s1_list = []
    val_candidates = {}
    needed_cand_ids = set()

    # Skip first 60k entities (which were train set in train_model.py)
    # and take the next 15,000 entities as true held-out validation!
    with open(cand_file, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i < 60000:
                continue
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1 = parts[0]
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            val_s1_list.append(s1)
            val_candidates[s1] = cands
            needed_cand_ids.update(cands)
            if len(val_s1_list) >= VAL_ENTITIES:
                break

    val_s1_set = set(val_s1_list)
    print(f"Held-out validation entities: {len(val_s1_list):,}")

    # Load ground truth
    gt_mapping = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            s1 = parts[0]
            if s1 in val_s1_set:
                if len(parts) > 1 and parts[1] and parts[1] != "nan":
                    gt_mapping[s1] = set(parts[1].split(","))
                else:
                    gt_mapping[s1] = set()

    for s1 in val_s1_set:
        if s1 not in gt_mapping:
            gt_mapping[s1] = set()

    # Load S1 records
    s1_dict = {}
    with open(s1_file, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        country_idx = header.index("country")
        name_idx = header.index("business_name_clean")
        addr_idx = header.index("business_address_clean")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1 = parts[id_idx]
            if s1 in val_s1_set:
                s1_dict[s1] = {
                    "entity_id": s1,
                    "country": parts[country_idx] if len(parts) > country_idx else "",
                    "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                    "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                }
                if len(s1_dict) == len(val_s1_set):
                    break

    # Load pool records for validation
    pool_dict = {}
    needed = set(needed_cand_ids)
    for path in [s2_file, s3_file]:
        with open(path, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\n").split("\t")
            id_idx = header.index("entity_id")
            country_idx = header.index("country")
            name_idx = header.index("business_name_clean")
            addr_idx = header.index("business_address_clean")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[id_idx]
                if eid in needed:
                    pool_dict[eid] = {
                        "entity_id": eid,
                        "country": parts[country_idx] if len(parts) > country_idx else "",
                        "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                        "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                    }
                    needed.discard(eid)
                    if not needed:
                        break

    print(f"Loaded {len(pool_dict):,} candidate pool records.")

    # Extract features and compute probabilities
    print("Computing candidate probabilities with trained LightGBM model...")
    t0 = time.time()
    s1_cand_probs = {}
    all_probs = []

    for s1_id in val_s1_list:
        s1_row = s1_dict.get(s1_id)
        cands = val_candidates.get(s1_id, [])
        if not s1_row or not cands:
            s1_cand_probs[s1_id] = []
            continue

        valid_cands = []
        batch_feats = []
        for rank, cid in enumerate(cands):
            cand_row = pool_dict.get(cid)
            if cand_row:
                feat = extract_pair_features(s1_row, cand_row, cand_rank=rank)
                batch_feats.append([feat[f] for f in FEATURE_NAMES])
                valid_cands.append(cid)

        if batch_feats:
            probs = clf.predict_proba(np.array(batch_feats, dtype=np.float32))[:, 1]
            s1_cand_probs[s1_id] = list(zip(valid_cands, probs))
            all_probs.extend(probs)
        else:
            s1_cand_probs[s1_id] = []

    all_probs = np.array(all_probs)
    print(f"Computed probabilities for {len(all_probs):,} candidate pairs in {time.time() - t0:.2f}s.")
    print(f"Probability Distribution: min={all_probs.min():.4f}, 25%={np.percentile(all_probs, 25):.4f}, median={np.median(all_probs):.4f}, 75%={np.percentile(all_probs, 75):.4f}, 95%={np.percentile(all_probs, 95):.4f}, max={all_probs.max():.4f}")

    # Count-based broad search: 0.10 to 0.99 (90 points)
    print("\n--- BROAD SWEEP: 0.10 to 0.99 (count-based 90 points) ---")
    broad_grid = np.linspace(0.10, 0.99, 90)
    broad_results = {}
    best_broad_th = 0.5
    best_broad_f05 = -1.0

    n_val = len(val_s1_list)

    for th in broad_grid:
        total_f05 = 0.0
        for s1_id in val_s1_list:
            gt_set = gt_mapping.get(s1_id, set())
            cands = s1_cand_probs.get(s1_id, [])
            pred_set = {cid for cid, p in cands if p >= th}
            total_f05 += compute_entity_f05(gt_set, pred_set)
        score = total_f05 / n_val
        broad_results[round(float(th), 4)] = score
        if score > best_broad_f05:
            best_broad_f05 = score
            best_broad_th = float(th)

    for th in sorted(broad_results.keys()):
        if int(th * 100) % 5 == 0 or abs(th - best_broad_th) < 0.015:
            marker = " <=== BROAD OPTIMAL" if abs(th - best_broad_th) < 1e-4 else ""
            print(f"  Threshold {th:.4f}: Macro F_0.5 = {broad_results[th]:.4f}{marker}")

    # Fine pass: tight step around winner (+/- 0.06 with 61 points, step 0.002)
    print(f"\n--- FINE SWEEP: around {best_broad_th:.4f} (+/- 0.06, step ~0.002) ---")
    fine_grid = np.linspace(max(0.01, best_broad_th - 0.06), min(0.999, best_broad_th + 0.06), 61)
    fine_results = {}
    best_fine_th = best_broad_th
    best_fine_f05 = best_broad_f05

    for th in fine_grid:
        total_f05 = 0.0
        for s1_id in val_s1_list:
            gt_set = gt_mapping.get(s1_id, set())
            cands = s1_cand_probs.get(s1_id, [])
            pred_set = {cid for cid, p in cands if p >= th}
            total_f05 += compute_entity_f05(gt_set, pred_set)
        score = total_f05 / n_val
        fine_results[round(float(th), 4)] = score
        if score > best_fine_f05:
            best_fine_f05 = score
            best_fine_th = float(th)

    for th in sorted(fine_results.keys()):
        marker = " <=== FINE OPTIMAL" if abs(th - best_fine_th) < 1e-4 else ""
        print(f"  Threshold {th:.4f}: Macro F_0.5 = {fine_results[th]:.4f}{marker}")

    # Plateau analysis
    plateau_threshs = [th for th, s in fine_results.items() if s >= best_fine_f05 - 0.001]
    print(f"\nPlateau Analysis (within 0.001 of peak {best_fine_f05:.4f}):")
    print(f"  Span: [{min(plateau_threshs):.4f}, {max(plateau_threshs):.4f}] (width: {max(plateau_threshs) - min(plateau_threshs):.4f})")
    print(f"  Optimal Decision Threshold: {best_fine_th:.4f}")

if __name__ == "__main__":
    main()
