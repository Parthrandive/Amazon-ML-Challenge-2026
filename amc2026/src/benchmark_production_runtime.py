"""
benchmark_production_runtime.py
Times full-scale production throughput on real test data:
1. Indexing rate
2. Candidate generation throughput (entities/sec) at K=50
3. Feature extraction throughput (pairs/sec)
4. LightGBM matrix inference throughput (pairs/sec)
5. Full pipeline runtime projection for 1,732,544 test entities.
"""

import sys
import time
import pickle
import array
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.append("amc2026/src")
from blocking import get_tight_blocking_keys, get_key_max_size
from features import extract_pair_features, FEATURE_NAMES

def main():
    print("=" * 60)
    print("PRODUCTION RUNTIME DRY-RUN BENCHMARK (REAL TEST DATA)")
    print("=" * 60)

    test_s1_path = "data/processed/test_source1_clean.tsv"
    test_s2_path = "data/processed/test_source2_clean.tsv"
    test_s3_path = "data/processed/test_source3_clean.tsv"
    model_path = "models/matcher_lgbm.pkl"

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["model"]

    # 1. Benchmark indexing rate on 500k test pool records
    print("Benchmarking pool indexing on 500,000 test pool records...")
    t0 = time.time()
    pool_id_list = []
    pool_names = []
    index = defaultdict(lambda: array.array("I"))

    df_pool = pd.read_csv(
        test_s2_path, sep="\t", nrows=500000,
        usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]
    )
    for r in df_pool.itertuples(index=False):
        idx = len(pool_id_list)
        pool_id_list.append(r.entity_id)
        c_name = str(r.business_name_clean) if pd.notna(r.business_name_clean) else ""
        pool_names.append(c_name)
        keys = get_tight_blocking_keys(str(r.country), c_name, str(r.business_address_clean))
        for k in keys:
            bucket = index[k]
            if len(bucket) < get_key_max_size(k):
                bucket.append(idx)

    index_time = time.time() - t0
    index_rate = len(df_pool) / index_time
    print(f"Indexed 500,000 records in {index_time:.2f}s ({index_rate:,.0f} records/sec).")
    del df_pool

    # 2. Benchmark candidate generation on 10,000 test S1 entities
    N_TEST_ENTITIES = 10000
    print(f"\nBenchmarking candidate generation on {N_TEST_ENTITIES:,} test entities (K=50, Rapidfuzz)...")
    s1_test_df = pd.read_csv(
        test_s1_path, sep="\t", nrows=N_TEST_ENTITIES,
        usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]
    )

    K_CUTOFF = 50
    t_cg = time.time()
    candidates_by_entity = []
    total_cand_pairs = 0

    for r in s1_test_df.itertuples(index=False):
        s1_name = str(r.business_name_clean) if pd.notna(r.business_name_clean) else ""
        keys = get_tight_blocking_keys(str(r.country), s1_name, str(r.business_address_clean))
        cand_counts = Counter()
        for k in keys:
            cand_counts.update(index.get(k, ()))

        if not cand_counts:
            candidates_by_entity.append([])
            continue

        if len(cand_counts) <= K_CUTOFF:
            selected_ids = list(cand_counts.keys())
        else:
            pool = list(cand_counts.keys()) if len(cand_counts) <= 2000 else [c for c, _ in cand_counts.most_common(2000)]
            scored = []
            for int_id in pool:
                c_name = pool_names[int_id]
                sim = 0.5 * (fuzz.token_set_ratio(s1_name, c_name) + fuzz.token_sort_ratio(s1_name, c_name)) / 100.0 if s1_name != c_name else 1.0
                scored.append((int_id, sim + 0.15 * min(cand_counts[int_id], 5)))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected_ids = [int_id for int_id, _ in scored[:K_CUTOFF]]

        total_cand_pairs += len(selected_ids)
        candidates_by_entity.append(selected_ids)

    cg_time = time.time() - t_cg
    cg_rate = N_TEST_ENTITIES / cg_time
    print(f"Generated {total_cand_pairs:,} candidates for {N_TEST_ENTITIES:,} entities in {cg_time:.2f}s ({cg_rate:,.0f} entities/sec).")
    print(f"Average candidates per entity: {total_cand_pairs / N_TEST_ENTITIES:.1f}")

    # 3. Benchmark pairwise feature extraction on 50,000 candidate pairs
    N_PAIRS_SAMPLE = min(50000, total_cand_pairs)
    print(f"\nBenchmarking feature extraction on {N_PAIRS_SAMPLE:,} candidate pairs...")
    s1_rows = s1_test_df.to_dict(orient="records")

    pairs_to_extract = []
    for s1_idx, c_list in enumerate(candidates_by_entity):
        s1_row = s1_rows[s1_idx]
        for rank, c_idx in enumerate(c_list):
            cand_row = {
                "entity_id": pool_id_list[c_idx],
                "business_name_clean": pool_names[c_idx],
                "business_address_clean": "",
                "country": s1_row.get("country", "")
            }
            pairs_to_extract.append((s1_row, cand_row, rank, 1))
            if len(pairs_to_extract) >= N_PAIRS_SAMPLE:
                break
        if len(pairs_to_extract) >= N_PAIRS_SAMPLE:
            break

    t_fe = time.time()
    features_matrix = []
    for s1_r, c_r, rk, sk in pairs_to_extract:
        f = extract_pair_features(s1_r, c_r, cand_rank=rk, shared_key_count=sk)
        features_matrix.append([f[feat] for feat in FEATURE_NAMES])

    fe_time = time.time() - t_fe
    fe_rate = len(features_matrix) / fe_time
    print(f"Extracted {len(features_matrix):,} feature vectors in {fe_time:.2f}s ({fe_rate:,.0f} pairs/sec).")

    # 4. Benchmark LightGBM batched inference
    print(f"\nBenchmarking LightGBM matrix inference on {len(features_matrix):,} pairs...")
    X_bench = np.array(features_matrix, dtype=np.float32)
    t_inf = time.time()
    probs = clf.predict_proba(X_bench)[:, 1]
    inf_time = time.time() - t_inf
    inf_rate = len(X_bench) / inf_time
    print(f"Scored {len(X_bench):,} pairs in {inf_time:.3f}s ({inf_rate:,.0f} pairs/sec).")

    # 5. Full-Scale Production Runtime Projections (1,732,544 test entities)
    TOTAL_TEST_S1 = 1732544
    TOTAL_TEST_POOL = 4887273 + 5082316  # S2 + S3 = 9.97M

    proj_index_sec = TOTAL_TEST_POOL / index_rate
    proj_cg_sec = TOTAL_TEST_S1 / cg_rate

    # Projected candidate pairs at K=50, K=30, K=20
    for k_val, avg_pairs in [(50, 48.0), (30, 29.0), (20, 19.5)]:
        total_pairs_proj = TOTAL_TEST_S1 * avg_pairs
        proj_fe_sec = total_pairs_proj / fe_rate
        proj_inf_sec = total_pairs_proj / inf_rate
        total_sec = proj_index_sec + proj_cg_sec + proj_fe_sec + proj_inf_sec
        total_min = total_sec / 60.0
        total_hours = total_sec / 3600.0

        print(f"\n{'=' * 60}")
        print(f"PROJECTED RUNTIME AT K = {k_val} ({total_pairs_proj/1e6:.1f}M PAIRS)")
        print(f"{'=' * 60}")
        print(f"  1. Pool Indexing (9.97M records):     {proj_index_sec/60:.1f} minutes")
        print(f"  2. Candidate Generation ({TOTAL_TEST_S1:,} S1): {proj_cg_sec/60:.1f} minutes")
        print(f"  3. Feature Extraction ({total_pairs_proj/1e6:.1f}M pairs): {proj_fe_sec/60:.1f} minutes")
        print(f"  4. LightGBM Scoring ({total_pairs_proj/1e6:.1f}M pairs):   {proj_inf_sec/60:.1f} minutes")
        print(f"  --------------------------------------------------")
        print(f"  TOTAL END-TO-END PIPELINE:            {total_min:.1f} minutes ({total_hours:.2f} hours)")

if __name__ == "__main__":
    main()
