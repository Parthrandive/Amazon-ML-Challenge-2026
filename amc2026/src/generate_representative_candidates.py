"""
generate_representative_candidates.py
Generates adaptive-depth candidate pairs with shared_key_count for a stratified,
representative sample of 80,000 Source 1 entities (India, US, France, unknown).
Output: outputs/train_candidates_adaptive.tsv
"""

import os
import sys
import time
import array
import gc
from collections import defaultdict, Counter
import pandas as pd

sys.path.append("src")
sys.path.append("amc2026/src")
from blocking import (
    get_tight_blocking_keys,
    get_fallback_blocking_keys,
    get_key_max_size
)

from rapidfuzz import fuzz

def main():
    print("=" * 60)
    print("GENERATING REPRESENTATIVE 80k ADAPTIVE CANDIDATES (K=50, RAPIDFUZZ)")
    print("=" * 60)

    out_file = "outputs/train_candidates_adaptive.tsv"
    s1_clean_path = "data/processed/source1_clean.tsv"
    s2_clean_path = "data/processed/source2_clean.tsv"
    s3_clean_path = "data/processed/source3_clean.tsv"

    # Target stratified sample counts (matching true 60% US / 40% India training distribution)
    sample_targets = {
        "US": 48000,
        "India": 32000,
    }
    K_CUTOFF = 50
    MAX_BLOCK_SIZE = 500

    print("Loading Source 1 and stratifying clean sample...")
    s1_df = pd.read_csv(
        s1_clean_path, sep="\t",
        usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]
    )
    s1_df["country"] = s1_df["country"].fillna("unknown")

    sample_dfs = []
    for c, target in sample_targets.items():
        c_df = s1_df[s1_df["country"] == c]
        n_avail = len(c_df)
        n_sample = min(target, n_avail)
        sampled = c_df.sample(n=n_sample, random_state=42)
        sample_dfs.append(sampled)
        print(f"  {c:10s}: Sampled {len(sampled):,} from {n_avail:,}")

    sampled_s1 = pd.concat(sample_dfs).reset_index(drop=True)
    print(f"Total sampled S1 entities: {len(sampled_s1):,}")

    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")

    t_start = time.time()
    total_processed = 0
    total_cands = 0

    # Process partitions: India, US
    for country in ["India", "US"]:
        print(f"\n--- Processing Partition: {country} ---")
        t_c = time.time()
        s1_sub = sampled_s1[sampled_s1["country"] == country]
        if s1_sub.empty:
            continue

        pool_id_list = []
        pool_names = []
        index = defaultdict(lambda: array.array("I"))

        for pool_name, path in [("Source 2", s2_clean_path), ("Source 3", s3_clean_path)]:
            for chunk in pd.read_csv(path, sep="\t", chunksize=500000,
                                     usecols=["entity_id", "country", "business_name_clean", "business_address_clean"]):
                country_chunk = chunk[chunk["country"] == country]
                for r in country_chunk.itertuples(index=False):
                    idx = len(pool_id_list)
                    pool_id_list.append(r.entity_id)
                    c_name = str(r.business_name_clean) if pd.notna(r.business_name_clean) else ""
                    pool_names.append(c_name)

                    keys = get_tight_blocking_keys(country, c_name, str(r.business_address_clean))
                    for k in keys:
                        lst = index[k]
                        max_cap = get_key_max_size(k)
                        if len(lst) < max_cap:
                            lst.append(idx)

        print(f"  Indexed {len(pool_id_list):,} {country} pool records ({len(index):,} keys) in {time.time() - t_c:.1f}s.")

        # Query sampled S1
        t_q = time.time()
        c_count = 0
        c_cands = 0
        with open(out_file, "a", encoding="utf-8") as f_out:
            lines = []
            for r in s1_sub.itertuples(index=False):
                s1_id = r.entity_id
                s1_name = str(r.business_name_clean) if pd.notna(r.business_name_clean) else ""

                keys = get_tight_blocking_keys(country, s1_name, str(r.business_address_clean))
                cand_counts = Counter()
                for k in keys:
                    cand_counts.update(index.get(k, ()))

                if not cand_counts:
                    lines.append(f"{s1_id}\t\n")
                    c_count += 1
                    continue

                if len(cand_counts) <= K_CUTOFF:
                    selected_ids = list(cand_counts.keys())
                else:
                    candidate_pool = list(cand_counts.keys()) if len(cand_counts) <= 2000 else [c for c, _ in cand_counts.most_common(2000)]
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
                    selected_ids = [int_id for int_id, _ in scored[:K_CUTOFF]]

                cand_str = ",".join(f"{pool_id_list[i]}:{cand_counts[i]}" for i in selected_ids)
                c_cands += len(selected_ids)
                lines.append(f"{s1_id}\t{cand_str}\n")
                c_count += 1

                if len(lines) >= 5000:
                    f_out.writelines(lines)
                    lines = []

            if lines:
                f_out.writelines(lines)

        print(f"  Streamed {c_count:,} {country} entities in {time.time() - t_q:.1f}s.")
        total_processed += c_count
        total_cands += c_cands

        del pool_id_list, pool_names, index
        gc.collect()

    print("\n" + "=" * 60)
    print(f"REPRESENTATIVE CANDIDATES COMPLETED in {time.time() - t_start:.2f}s")
    print(f"Total S1 entities: {total_processed:,}")
    print(f"Total candidates:  {total_cands:,} (avg {total_cands/total_processed:.1f} cands/entity)")
    print("=" * 60)

if __name__ == "__main__":
    main()
