"""
audit_k_recall.py
Audits candidate generation recall across K = [10, 20, 25, 30, 40, 50, 60, 80, 100, uncapped]
for clean stratified US and India partitions using C-backed rapidfuzz re-ranking.
"""

import os
import sys
import time
import array
import gc
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz

sys.path.append("src")
sys.path.append("amc2026/src")
from blocking import get_tight_blocking_keys, get_key_max_size

K_THRESHOLDS = [10, 20, 25, 30, 40, 50, 60, 80, 100]
TOP_M_RERANK = 2000


def audit_country_recall(
    country: str,
    s1_sub: pd.DataFrame,
    gt_mapping: Dict[str, Set[str]],
    s2_path: str,
    s3_path: str,
) -> Dict[str, float]:
    print(f"\n{'=' * 60}")
    print(f"AUDITING CANDIDATE RECALL FOR: {country} (N={len(s1_sub):,} S1 entities)")
    print(f"{'=' * 60}")

    # Total ground truth links for this S1 subset
    total_gt_pairs = sum(len(gt_mapping.get(r.entity_id, set())) for r in s1_sub.itertuples(index=False))
    print(f"Total Ground Truth Pairs in partition: {total_gt_pairs:,}")

    # 1. Index Candidate Pool
    t0 = time.time()
    pool_id_list: List[str] = []
    pool_id_to_idx: Dict[str, int] = {}
    pool_names: List[str] = []
    index = defaultdict(lambda: array.array("I"))

    for pool_name, path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        for chunk in pd.read_csv(
            path, sep="\t", chunksize=500000,
            usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]
        ):
            c_chunk = chunk[chunk["country"] == country]
            for r in c_chunk.itertuples(index=False):
                idx = len(pool_id_list)
                eid = str(r.entity_id)
                pool_id_list.append(eid)
                pool_id_to_idx[eid] = idx
                c_name = str(r.business_name_clean) if pd.notna(r.business_name_clean) else ""
                pool_names.append(c_name)

                keys = get_tight_blocking_keys(country, c_name, str(r.business_address_clean))
                for k in keys:
                    bucket = index[k]
                    if len(bucket) < get_key_max_size(k):
                        bucket.append(idx)

    print(f"Indexed {len(pool_id_list):,} {country} pool records ({len(index):,} keys) in {time.time() - t0:.1f}s.")

    # 2. Query S1 entities and evaluate ranking of ground truth matches
    t_q = time.time()
    k_hits = Counter()
    top_m_hits = 0
    uncapped_hits = 0

    s1_processed = 0
    for r in s1_sub.itertuples(index=False):
        s1_id = r.entity_id
        true_ids = gt_mapping.get(s1_id, set())
        if not true_ids:
            s1_processed += 1
            continue

        s1_name = str(r.business_name_clean) if pd.notna(r.business_name_clean) else ""
        s1_addr = str(r.business_address_clean) if pd.notna(r.business_address_clean) else ""

        # Map true IDs to pool indices
        true_indices = {pool_id_to_idx[tid] for tid in true_ids if tid in pool_id_to_idx}

        keys = get_tight_blocking_keys(country, s1_name, s1_addr)
        cand_counts = Counter()
        for k in keys:
            cand_counts.update(index.get(k, ()))

        if not cand_counts:
            s1_processed += 1
            continue

        # Uncapped check: was true index retrieved in ANY blocking key?
        for tidx in true_indices:
            if tidx in cand_counts:
                uncapped_hits += 1

        # Rapidfuzz Re-ranking
        if len(cand_counts) <= TOP_M_RERANK:
            candidate_pool = list(cand_counts.keys())
        else:
            candidate_pool = [c for c, _ in cand_counts.most_common(TOP_M_RERANK)]

        scored = []
        for int_id in candidate_pool:
            c_name = pool_names[int_id]
            if s1_name == c_name:
                sim = 1.0
            else:
                sim = 0.5 * (fuzz.token_set_ratio(s1_name, c_name) + fuzz.token_sort_ratio(s1_name, c_name)) / 100.0
            score = sim + 0.15 * min(cand_counts[int_id], 5)
            scored.append((int_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_pool = [int_id for int_id, _ in scored]
        ranked_set = set(ranked_pool)

        # Check hits at each K
        for tidx in true_indices:
            if tidx in ranked_set:
                top_m_hits += 1
                rank = ranked_pool.index(tidx) + 1  # 1-based rank
                for k in K_THRESHOLDS:
                    if rank <= k:
                        k_hits[k] += 1

        s1_processed += 1
        if s1_processed % 10000 == 0:
            print(f"  Processed {s1_processed:,}/{len(s1_sub):,} S1 entities ({time.time() - t_q:.1f}s)...")

    query_elapsed = time.time() - t_q
    print(f"Completed querying {s1_processed:,} {country} entities in {query_elapsed:.1f}s ({s1_processed/query_elapsed:,.0f} entities/s).")

    # 3. Print K-Recall Table
    print(f"\n--- {country.upper()} CANDIDATE RECALL VS K-CUTOFF ---")
    results = {}
    for k in K_THRESHOLDS:
        rec = (k_hits[k] / total_gt_pairs) * 100.0 if total_gt_pairs > 0 else 0.0
        results[f"K={k}"] = rec
        delta_str = f" (+{rec - results['K=20']:.2f}%)" if k > 20 else ""
        print(f"  K = {k:3d}: {k_hits[k]:8,d} / {total_gt_pairs:,} true pairs -> {rec:6.2f}% Recall{delta_str}")

    rec_top_m = (top_m_hits / total_gt_pairs) * 100.0 if total_gt_pairs > 0 else 0.0
    rec_uncapped = (uncapped_hits / total_gt_pairs) * 100.0 if total_gt_pairs > 0 else 0.0
    print(f"  Top-{TOP_M_RERANK}: {top_m_hits:8,d} / {total_gt_pairs:,} true pairs -> {rec_top_m:6.2f}% Recall")
    print(f"  Uncapped (K=inf): {uncapped_hits:8,d} / {total_gt_pairs:,} true pairs -> {rec_uncapped:6.2f}% Recall")

    # Clean up memory
    del pool_id_list, pool_id_to_idx, pool_names, index
    gc.collect()

    return results


def main():
    print("=" * 60)
    print("AUDITING K-RECALL CURVE (RAPIDFUZZ ACCELERATED)")
    print("=" * 60)

    gt_file = "data/raw/train_ground_truth.tsv"
    s1_clean_path = "data/processed/source1_clean.tsv"
    s2_clean_path = "data/processed/source2_clean.tsv"
    s3_clean_path = "data/processed/source3_clean.tsv"

    # 1. Load ground truth
    print(f"Loading ground truth from {gt_file}...")
    t0 = time.time()
    gt_mapping: Dict[str, Set[str]] = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1] and parts[1] != "nan":
                gt_mapping[parts[0]] = {x.strip() for x in parts[1].split(",") if x.strip()}
            else:
                gt_mapping[parts[0]] = set()
    print(f"Loaded ground truth for {len(gt_mapping):,} entities in {time.time() - t0:.2f}s.")

    # 2. Sample 80,000 clean S1 entities (48k US, 32k India, identical seed 42)
    print("Loading Source 1 and stratifying clean sample...")
    s1_df = pd.read_csv(
        s1_clean_path, sep="\t",
        usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]
    )
    s1_df["country"] = s1_df["country"].fillna("unknown")

    sample_dfs = []
    for c, target in [("India", 32000), ("US", 48000)]:
        c_df = s1_df[s1_df["country"] == c]
        sampled = c_df.sample(n=min(target, len(c_df)), random_state=42)
        sample_dfs.append(sampled)
    del s1_df
    gc.collect()

    s1_india = sample_dfs[0]
    s1_us = sample_dfs[1]

    # Audit India partition
    res_india = audit_country_recall("India", s1_india, gt_mapping, s2_clean_path, s3_clean_path)

    # Audit US partition
    res_us = audit_country_recall("US", s1_us, gt_mapping, s2_clean_path, s3_clean_path)

    print("\n" + "=" * 60)
    print("SUMMARY: RECALL RECOVERY ACROSS K-THRESHOLDS")
    print("=" * 60)
    print(f"{'K Threshold':<12s} | {'India Recall':<15s} | {'US Recall':<15s}")
    print("-" * 46)
    for k in K_THRESHOLDS:
        k_key = f"K={k}"
        print(f"{k_key:<12s} | {res_india.get(k_key, 0):.2f}%          | {res_us.get(k_key, 0):.2f}%")


if __name__ == "__main__":
    main()
