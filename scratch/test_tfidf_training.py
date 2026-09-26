"""
scratch/test_tfidf_training.py
Benchmark LightGBM performance with TF-IDF-weighted token features:
  - name_tfidf_jaccard: Jaccard weighted by token rarity
  - name_tfidf_cosine: Cosine similarity of token TF-IDF vectors (squares rarity)
Trained on 60,000 entities, evaluated on 20,000 validation entities.
"""

import os
import sys
import time
import math
import pickle
import random
from collections import defaultdict
import numpy as np
import lightgbm as lgb

sys.path.append("amc2026/src")
from features import extract_pair_features, FEATURE_NAMES
from evaluate import compute_entity_f05, evaluate_predictions
from model import train_lgbm_model, find_optimal_threshold_for_f05

# Load token IDF table
with open("models/token_idf.pkl", "rb") as f:
    idf_table = pickle.load(f)
default_idf = idf_table["__DEFAULT__"]


def compute_tfidf_name_features(name1: str, name2: str):
    toks1 = set(name1.split())
    toks2 = set(name2.split())
    if not toks1 or not toks2:
        return 0.0, 0.0
    
    inter = toks1 & toks2
    if not inter:
        return 0.0, 0.0
    
    union = toks1 | toks2
    # 1. Weighted Jaccard
    w_inter = sum(idf_table.get(t, default_idf) for t in inter)
    w_union = sum(idf_table.get(t, default_idf) for t in union)
    w_jacc = w_inter / w_union if w_union > 0 else 0.0
    
    # 2. Weighted Cosine (squared IDFs)
    dot = sum(idf_table.get(t, default_idf)**2 for t in inter)
    n1 = math.sqrt(sum(idf_table.get(t, default_idf)**2 for t in toks1))
    n2 = math.sqrt(sum(idf_table.get(t, default_idf)**2 for t in toks2))
    w_cos = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
    
    return w_jacc, w_cos


def main():
    print("=" * 60)
    print("TF-IDF WEIGHTED TOKEN FEATURE EXPERIMENT")
    print("=" * 60)
    
    cand_file = "outputs/train_candidates_adaptive.tsv"
    gt_file = "student_resource/dataset/train/train_ground_truth.tsv"
    s1_clean_path = "data/processed/source1_clean.tsv"
    s2_clean_path = "data/processed/source2_clean.tsv"
    s3_clean_path = "data/processed/source3_clean.tsv"
    
    # Load candidate pairs (80,000 entities)
    print("Loading candidate pairs...")
    s1_candidates = {}
    needed_cids = set()
    s1_order = []
    
    with open(cand_file, "r") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts: continue
            s1 = parts[0]
            s1_order.append(s1)
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            s1_candidates[s1] = cands
            for item in cands:
                cid = item.split(":", 1)[0]
                needed_cids.add(cid)
    
    print(f"Loaded {len(s1_order):,} entities. Needed pool IDs: {len(needed_cids):,}")
    
    # Load GT
    gt_mapping = {}
    with open(gt_file, "r") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1 = parts[0]
            if s1 in s1_candidates:
                gt_mapping[s1] = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
    for s1 in s1_order:
        if s1 not in gt_mapping: gt_mapping[s1] = set()
        
    # Load S1
    s1_dict = {}
    with open(s1_clean_path, "r") as f:
        hdr = next(f).rstrip("\n").split("\t")
        id_idx = hdr.index("entity_id")
        c_idx = hdr.index("country")
        n_idx = hdr.index("business_name_clean")
        a_idx = hdr.index("business_address_clean")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1 = parts[id_idx]
            if s1 in s1_candidates:
                s1_dict[s1] = {
                    "entity_id": s1,
                    "country": parts[c_idx] if len(parts) > c_idx else "",
                    "business_name_clean": parts[n_idx] if len(parts) > n_idx else "",
                    "business_address_clean": parts[a_idx] if len(parts) > a_idx else "",
                }
    
    # Load pool
    print("Loading candidate pool records...")
    t0 = time.time()
    pool_dict = {}
    needed = set(needed_cids)
    for p in [s2_clean_path, s3_clean_path]:
        with open(p, "r") as f:
            hdr = next(f).rstrip("\n").split("\t")
            id_idx = hdr.index("entity_id")
            c_idx = hdr.index("country")
            n_idx = hdr.index("business_name_clean")
            a_idx = hdr.index("business_address_clean")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[id_idx]
                if eid in needed:
                    pool_dict[eid] = {
                        "entity_id": eid,
                        "country": parts[c_idx] if len(parts) > c_idx else "",
                        "business_name_clean": parts[n_idx] if len(parts) > n_idx else "",
                        "business_address_clean": parts[a_idx] if len(parts) > a_idx else "",
                    }
                    needed.discard(eid)
                    if not needed: break
    print(f"Loaded {len(pool_dict):,} pool records in {time.time() - t0:.2f}s.")
    
    # Split 60k train / 20k validation
    train_s1 = s1_order[:60000]
    val_s1 = s1_order[60000:80000]
    print(f"Split: {len(train_s1):,} Train entities, {len(val_s1):,} Validation entities.")
    
    # Extract training features WITH TF-IDF
    print("Extracting training pairs with TF-IDF features (32 features)...")
    t0 = time.time()
    X_train_list = []
    y_train_list = []
    
    for s1_id in train_s1:
        s1_row = s1_dict.get(s1_id)
        if not s1_row: continue
        true_ids = gt_mapping.get(s1_id, set())
        cands = s1_candidates.get(s1_id, [])
        s1_name = s1_row.get("business_name_clean", "")
        
        pos_count = 0
        neg_count = 0
        max_neg = 4 * len(true_ids) if true_ids else 3
        
        for rank, item in enumerate(cands):
            cid, count = (item.split(":", 1)[0], float(item.split(":", 1)[1])) if ":" in item else (item, 1.0)
            cand_row = pool_dict.get(cid)
            if not cand_row: continue
            
            is_match = 1 if cid in true_ids else 0
            if is_match or neg_count < max_neg:
                base_feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=count)
                cand_name = cand_row.get("business_name_clean", "")
                wj, wcos = compute_tfidf_name_features(s1_name, cand_name)
                
                # Append base features + 2 TF-IDF features
                feat_vec = [base_feats[f] for f in FEATURE_NAMES] + [wj, wcos]
                X_train_list.append(feat_vec)
                y_train_list.append(is_match)
                if not is_match:
                    neg_count += 1
    
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"Extracted {len(X_train):,} training pairs ({sum(y_train):,} positive, {len(y_train) - sum(y_train):,} negative) in {time.time() - t0:.2f}s.")
    
    # Extract Validation pairs
    print("Extracting validation pairs...")
    t0 = time.time()
    val_s1_cands_feats = {}
    for s1_id in val_s1:
        s1_row = s1_dict.get(s1_id)
        cands = s1_candidates.get(s1_id, [])
        s1_name = s1_row.get("business_name_clean", "") if s1_row else ""
        pair_list = []
        if s1_row and cands:
            for rank, item in enumerate(cands):
                cid, count = (item.split(":", 1)[0], float(item.split(":", 1)[1])) if ":" in item else (item, 1.0)
                cand_row = pool_dict.get(cid)
                if cand_row:
                    base_feats = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=count)
                    cand_name = cand_row.get("business_name_clean", "")
                    wj, wcos = compute_tfidf_name_features(s1_name, cand_name)
                    feat_vec = np.array([base_feats[f] for f in FEATURE_NAMES] + [wj, wcos], dtype=np.float32)
                    pair_list.append((cid, feat_vec))
        val_s1_cands_feats[s1_id] = pair_list
    print(f"Validation pairs prepared in {time.time() - t0:.2f}s.")
    
    # Train LightGBM model
    print("Training LightGBM model with TF-IDF features...")
    t0 = time.time()
    clf = train_lgbm_model(X_train, y_train)
    print(f"Trained in {time.time() - t0:.2f}s.")
    
    # Feature importances
    feature_names_all = FEATURE_NAMES + ["name_tfidf_jaccard", "name_tfidf_cosine"]
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Feature Importances (Gain):")
    for i in sorted_idx[:10]:
        print(f"  {feature_names_all[i]:25s}: {importances[i]:,.1f}")
    
    # Optimal threshold search
    print("\nTuning decision threshold for Macro F0.5...")
    best_thresh, best_f05, history = find_optimal_threshold_for_f05(
        clf, val_s1, val_s1_cands_feats, gt_mapping
    )
    print(f"\n>>> Best Decision Threshold: {best_thresh:.4f} | Macro F_0.5 = {best_f05:.4f} <<<")
    
    # Detailed metrics
    val_pred = {}
    for s1 in val_s1:
        cands = val_s1_cands_feats.get(s1, [])
        matches = {cid for cid, fvec in cands if clf.predict_proba(fvec.reshape(1, -1))[0, 1] >= best_thresh} if cands else set()
        val_pred[s1] = matches
    
    detailed = evaluate_predictions(gt_mapping, val_pred, required_s1_ids=val_s1)
    print(f"  Macro F_0.5:      {detailed['macro_f05']:.4f}")
    print(f"  Mean Precision:   {detailed['mean_precision']:.4f}")
    print(f"  Mean Recall:      {detailed['mean_recall']:.4f}")
    print(f"  Singleton Acc:    {detailed['singleton_accuracy']:.4f}")


if __name__ == "__main__":
    main()
