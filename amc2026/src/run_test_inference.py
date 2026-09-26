"""
run_test_inference.py
Generates the complete competition submission for all 1,732,544 test entities:
1. Stage 1: Candidate Generation (K=50, C-backed Rapidfuzz re-ranking)
2. Stage 2: Batched Feature Extraction & LightGBM Inference with Hardened Thresholds
3. Writes output/candidate_pairs.tsv and output/matching_results.tsv
4. Aligns row order to match test_source1.tsv exactly.
5. Runs validate_submission.py to verify formatting compliance.
"""

import os
import sys
import time
import pickle
import array
import gc
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.append("src")
sys.path.append("amc2026/src")
from blocking import get_tight_blocking_keys, get_key_max_size
from features import extract_pair_features, FEATURE_NAMES

K_CUTOFF = 50
TOP_M_RERANK = 2000
BATCH_SIZE = 5000


def main():
    t_pipeline_start = time.time()
    print("=" * 70)
    print("STARTING FULL TEST SET INFERENCE PIPELINE (1.73M ENTITIES)")
    print("=" * 70)

    test_s1_path = "data/processed/test_source1_clean.tsv"
    test_s2_path = "data/processed/test_source2_clean.tsv"
    test_s3_path = "data/processed/test_source3_clean.tsv"
    model_path = "models/matcher_lgbm.pkl"
    raw_test_s1_path = "student_resource/dataset/test/test_source1.tsv"

    os.makedirs("output", exist_ok=True)
    temp_cand_path = "output/temp_candidates.tsv"
    temp_match_path = "output/temp_matches.tsv"
    final_cand_path = "output/candidate_pairs.tsv"
    final_match_path = "output/matching_results.tsv"

    # 1. Load Model & Thresholds
    print(f"Loading trained LightGBM model from {model_path}...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["model"]
    country_thresholds = bundle["country_thresholds"]
    global_th = bundle["threshold"]

    # Hardened operating thresholds
    thresholds = {
        "India": country_thresholds.get("India", 0.7000),
        "US": country_thresholds.get("US", 0.8800),
        "France": country_thresholds.get("France", 0.7900),
    }
    print(f"Active Hardened Operating Thresholds: {thresholds}")

    # 2. Read full test S1 metadata
    print(f"Loading test Source 1 records from {test_s1_path}...")
    t0 = time.time()
    s1_df = pd.read_csv(
        test_s1_path, sep="\t",
        usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]
    )
    s1_df["country"] = s1_df["country"].fillna("unknown")
    total_test_s1 = len(s1_df)
    print(f"Loaded {total_test_s1:,} test S1 entities in {time.time() - t0:.2f}s.")
    print("Breakdown by country:")
    for c, cnt in s1_df["country"].value_counts().items():
        print(f"  {c:10s}: {cnt:,} entities")

    # Clear temp files
    with open(temp_cand_path, "w", encoding="utf-8") as f_c, open(temp_match_path, "w", encoding="utf-8") as f_m:
        pass

    total_processed_entities = 0
    total_matches_count = 0
    total_entities_with_matches = 0

    # 3. Process Country by Country (India, US, France)
    for country in ["India", "US", "France"]:
        print("\n" + "=" * 70)
        print(f"PROCESSING PARTITION: {country}")
        print("=" * 70)

        s1_country = s1_df[s1_df["country"] == country]
        n_country_s1 = len(s1_country)
        if n_country_s1 == 0:
            continue

        th = thresholds.get(country, global_th)
        print(f"Target entities: {n_country_s1:,} | Operating Threshold tau: {th:.4f}")

        # Index Candidate Pool for this country
        t_index = time.time()
        pool_id_list: List[str] = []
        pool_names: List[str] = []
        pool_addrs: List[str] = []
        index = defaultdict(lambda: array.array("I"))

        for pool_name, path in [("Source 2", test_s2_path), ("Source 3", test_s3_path)]:
            t_p = time.time()
            chunk_cnt = 0
            for chunk in pd.read_csv(
                path, sep="\t", chunksize=500000,
                usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]
            ):
                country_chunk = chunk[chunk["country"] == country]
                if country_chunk.empty:
                    continue
                for eid, c_name, c_addr in zip(
                    country_chunk["entity_id"],
                    country_chunk["business_name_clean"].fillna(""),
                    country_chunk["business_address_clean"].fillna("")
                ):
                    idx = len(pool_id_list)
                    pool_id_list.append(str(eid))
                    pool_names.append(str(c_name))
                    pool_addrs.append(str(c_addr))

                    keys = get_tight_blocking_keys(country, c_name, c_addr)
                    for k in keys:
                        bucket = index[k]
                        if len(bucket) < get_key_max_size(k):
                            bucket.append(idx)
                    chunk_cnt += 1

            print(f"  Indexed {pool_name} ({chunk_cnt:,} {country} records) in {time.time() - t_p:.1f}s.")

        print(f"Total {country} pool indexed: {len(pool_id_list):,} records ({len(index):,} keys) in {time.time() - t_index:.1f}s.")

        # Stream S1 and run inference in batches
        t_infer = time.time()
        c_country_processed = 0
        c_country_matched = 0
        c_country_links = 0

        # Convert S1 subset to list of tuples for fast iteration
        s1_tuples = list(zip(
            s1_country["entity_id"],
            s1_country["business_name_clean"].fillna(""),
            s1_country["business_address_clean"].fillna("")
        ))

        with open(temp_cand_path, "a", encoding="utf-8") as f_c_out, \
             open(temp_match_path, "a", encoding="utf-8") as f_m_out:

            for b_start in range(0, n_country_s1, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, n_country_s1)
                batch_s1 = s1_tuples[b_start:b_end]

                cand_lines = []
                match_lines = []

                batch_features = []
                batch_slices = []  # (s1_id, start_idx, end_idx, [cand_pool_idx])

                for s1_id, s1_name, s1_addr in batch_s1:
                    keys = get_tight_blocking_keys(country, s1_name, s1_addr)
                    cand_counts = Counter()
                    for k in keys:
                        cand_counts.update(index.get(k, ()))

                    if not cand_counts:
                        cand_lines.append(f"{s1_id}\t\n")
                        match_lines.append(f"{s1_id}\t\n")
                        c_country_processed += 1
                        continue

                    # Rapidfuzz candidate re-ranking
                    if len(cand_counts) <= K_CUTOFF:
                        selected_pool_indices = list(cand_counts.keys())
                    else:
                        pool = list(cand_counts.keys()) if len(cand_counts) <= TOP_M_RERANK else [c for c, _ in cand_counts.most_common(TOP_M_RERANK)]
                        scored = []
                        for int_id in pool:
                            c_name = pool_names[int_id]
                            if s1_name == c_name:
                                sim = 1.0
                            else:
                                sim = 0.5 * (fuzz.token_set_ratio(s1_name, c_name) + fuzz.token_sort_ratio(s1_name, c_name)) / 100.0
                            score = sim + 0.15 * min(cand_counts[int_id], 5)
                            scored.append((int_id, score))

                        scored.sort(key=lambda x: x[1], reverse=True)
                        selected_pool_indices = [int_id for int_id, _ in scored[:K_CUTOFF]]

                    cand_str = ",".join(pool_id_list[i] for i in selected_pool_indices)
                    cand_lines.append(f"{s1_id}\t{cand_str}\n")

                    # Queue pairs for feature extraction
                    f_start = len(batch_features)
                    s1_row = {
                        "entity_id": s1_id,
                        "country": country,
                        "business_name_clean": s1_name,
                        "business_address_clean": s1_addr,
                    }
                    for rank, p_idx in enumerate(selected_pool_indices):
                        cand_row = {
                            "entity_id": pool_id_list[p_idx],
                            "country": country,
                            "business_name_clean": pool_names[p_idx],
                            "business_address_clean": pool_addrs[p_idx],
                        }
                        feat = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=cand_counts[p_idx])
                        batch_features.append([feat[f] for f in FEATURE_NAMES])

                    f_end = len(batch_features)
                    batch_slices.append((s1_id, f_start, f_end, selected_pool_indices))
                    c_country_processed += 1

                # LightGBM inference for the batch
                if batch_features:
                    X_b = np.array(batch_features, dtype=np.float32)
                    probs = clf.predict_proba(X_b)[:, 1]

                    for s1_id, f_start, f_end, pool_indices in batch_slices:
                        cand_scores = [
                            (pool_id_list[pool_indices[offset]], probs[f_start + offset])
                            for offset in range(f_end - f_start)
                            if probs[f_start + offset] >= th
                        ]
                        if cand_scores:
                            cand_scores.sort(key=lambda x: x[1], reverse=True)
                            match_str = ",".join(cid for cid, _ in cand_scores)
                            match_lines.append(f"{s1_id}\t{match_str}\n")
                            c_country_matched += 1
                            c_country_links += len(cand_scores)
                        else:
                            match_lines.append(f"{s1_id}\t\n")
                else:
                    for s1_id, _, _, _ in batch_slices:
                        match_lines.append(f"{s1_id}\t\n")

                f_c_out.writelines(cand_lines)
                f_m_out.writelines(match_lines)

                if c_country_processed % 50000 == 0 or c_country_processed == n_country_s1:
                    rate = c_country_processed / (time.time() - t_infer)
                    print(f"  Processed {c_country_processed:,}/{n_country_s1:,} {country} entities ({rate:,.0f} entities/s) | {c_country_matched:,} with matches ({c_country_links:,} total links)")

        print(f"\nFinished {country}: {c_country_processed:,} entities in {time.time() - t_infer:.1f}s.")
        print(f"  Entities with matches: {c_country_matched:,} ({c_country_matched/c_country_processed*100:.2f}%)")
        print(f"  Singletons: {c_country_processed - c_country_matched:,} ({(c_country_processed - c_country_matched)/c_country_processed*100:.2f}%)")
        print(f"  Total matched links: {c_country_links:,}")

        total_processed_entities += c_country_processed
        total_entities_with_matches += c_country_matched
        total_matches_count += c_country_links

        # Free memory before next country
        del pool_id_list, pool_names, pool_addrs, index, s1_tuples
        gc.collect()

    print("\n" + "=" * 70)
    print("ALL PARTITIONS PROCESSED — ALIGNING OUTPUT TO EXACT TEST ORDER")
    print("=" * 70)
    t_sort = time.time()

    # Load required order of S1 IDs from raw test_source1.tsv
    print(f"Reading target entity ordering from {raw_test_s1_path}...")
    target_s1_ids = []
    with open(raw_test_s1_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0].strip():
                target_s1_ids.append(parts[0].strip())

    print(f"Target S1 entities: {len(target_s1_ids):,}.")

    # Load temp candidates into memory dict
    print("Reading generated candidates from temp file...")
    cand_dict = {}
    with open(temp_cand_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                cand_dict[parts[0]] = parts[1] if len(parts) > 1 else ""

    # Write aligned candidate_pairs.tsv
    print(f"Writing aligned {final_cand_path}...")
    with open(final_cand_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        lines = []
        for s1_id in target_s1_ids:
            cands = cand_dict.get(s1_id, "")
            lines.append(f"{s1_id}\t{cands}\n")
            if len(lines) >= 50000:
                f_out.writelines(lines)
                lines = []
        if lines:
            f_out.writelines(lines)

    del cand_dict
    gc.collect()
    print(f"Wrote {final_cand_path} in {time.time() - t_sort:.1f}s.")

    # Load temp matches into memory dict
    t_match_sort = time.time()
    print("Reading generated matches from temp file...")
    match_dict = {}
    with open(temp_match_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                match_dict[parts[0]] = parts[1] if len(parts) > 1 else ""

    # Write aligned matching_results.tsv
    print(f"Writing aligned {final_match_path}...")
    with open(final_match_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        lines = []
        for s1_id in target_s1_ids:
            matches = match_dict.get(s1_id, "")
            lines.append(f"{s1_id}\t{matches}\n")
            if len(lines) >= 50000:
                f_out.writelines(lines)
                lines = []
        if lines:
            f_out.writelines(lines)

    del match_dict
    gc.collect()
    print(f"Wrote {final_match_path} in {time.time() - t_match_sort:.1f}s.")

    # Clean up temp files
    if os.path.exists(temp_cand_path): os.remove(temp_cand_path)
    if os.path.exists(temp_match_path): os.remove(temp_match_path)

    total_pipeline_time = time.time() - t_pipeline_start
    print("\n" + "=" * 70)
    print("TEST INFERENCE PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Total entities processed: {total_processed_entities:,}")
    print(f"Entities with matches:    {total_entities_with_matches:,} ({total_entities_with_matches/total_processed_entities*100:.2f}%)")
    print(f"Singletons:               {total_processed_entities - total_entities_with_matches:,} ({(total_processed_entities - total_entities_with_matches)/total_processed_entities*100:.2f}%)")
    print(f"Total matched pairs:      {total_matches_count:,}")
    print(f"Total pipeline runtime:   {total_pipeline_time/60:.2f} minutes ({total_pipeline_time/3600:.2f} hours)")

    # 4. Run Validator
    print("\n" + "=" * 70)
    print("RUNNING SUBMISSION VALIDATOR")
    print("=" * 70)
    val_cmd = (
        "python3 student_resource/utils/validate_submission.py "
        f"--matching {final_match_path} "
        f"--candidate {final_cand_path} "
        "--test-dir student_resource/dataset/test"
    )
    print(f"Executing: {val_cmd}")
    val_res = os.system(val_cmd)
    if val_res == 0:
        import zipfile
        zip_path = "output/submission.zip"
        print(f"\nCreating submission zip: {zip_path}...")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(final_match_path, arcname="output/matching_results.tsv")
            zf.write(final_cand_path, arcname="output/candidate_pairs.tsv")
            # Also include root-level copies for portals that expect them at root
            zf.write(final_match_path, arcname="matching_results.tsv")
            zf.write(final_cand_path, arcname="candidate_pairs.tsv")
        print(f"Created {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.1f} MB)")
        print("\nAll tasks completed successfully! Ready for Portal upload.")
    else:
        print(f"\nVALIDATOR FAILED with exit code {val_res}!")


if __name__ == "__main__":
    main()
