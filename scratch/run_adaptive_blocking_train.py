"""
run_adaptive_blocking_train.py
Generates candidate pairs with adaptive depth uncapping and persists shared_key_count
for an 80,000 S1 training sample.
Output: outputs/train_candidates_adaptive.tsv (format: s1_id\tcid1:count1,cid2:count2,...)
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
from blocking import get_tight_blocking_keys, get_key_max_size

os.makedirs("outputs", exist_ok=True)

def main():
    print("=" * 60)
    print("GENERATING ADAPTIVE-DEPTH CANDIDATES WITH SHARED_KEY_COUNT (80k SAMPLE)")
    print("=" * 60)

    out_file = "outputs/train_candidates_adaptive.tsv"
    s1_clean_path = "data/processed/source1_clean.tsv"
    s2_clean_path = "data/processed/source2_clean.tsv"
    s3_clean_path = "data/processed/source3_clean.tsv"

    SAMPLE_ENTITIES = 80000
    K_CUTOFF = 20
    CHUNK_SIZE = 50000

    # 1. Load the first 80,000 S1 entities
    print(f"Loading first {SAMPLE_ENTITIES:,} entities from {s1_clean_path}...")
    s1_sample_df = pd.read_csv(s1_clean_path, sep="\t", nrows=SAMPLE_ENTITIES,
                               usecols=["entity_id", "country", "business_name_clean", "business_address_clean"])
    countries = list(s1_sample_df["country"].unique())
    print(f"Sample contains countries: {countries} ({len(s1_sample_df):,} total entities)")

    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")

    t_start = time.time()
    total_s1 = 0
    total_cands = 0

    for country in countries:
        print(f"\n--- Processing Partition: {country} ---")
        t_c = time.time()
        s1_country_df = s1_sample_df[s1_sample_df["country"] == country]
        country_s1_target = len(s1_country_df)
        print(f"Target S1 entities for {country}: {country_s1_target:,}")

        # Build adaptive-depth inverted index for this country from S2 and S3
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
                        # Adaptive depth: uncapped/high capacity for specific keys, capped at 500 for broad keys
                        max_cap = get_key_max_size(k)
                        if len(lst) < max_cap:
                            lst.append(idx)

        print(f"  Indexed {len(pool_id_list):,} {country} pool records ({len(index):,} keys) in {time.time() - t_c:.1f}s.")

        # Query S1 entities
        t_q = time.time()
        c_s1_count = 0
        c_cand_count = 0

        with open(out_file, "a", encoding="utf-8") as f_out:
            lines = []
            for r in s1_country_df.itertuples(index=False):
                s1_id = r.entity_id
                s1_name = str(r.business_name_clean) if pd.notna(r.business_name_clean) else ""
                s1_toks = set(s1_name.split())
                len1 = len(s1_name)
                s1_g1 = {s1_name[i:i + 3] for i in range(len1 - 2)} if len1 >= 3 else {s1_name}

                keys = get_tight_blocking_keys(country, s1_name, str(r.business_address_clean))
                cand_counts = Counter()
                for k in keys:
                    cand_counts.update(index.get(k, ()))

                if not cand_counts:
                    lines.append(f"{s1_id}\t\n")
                    c_s1_count += 1
                    continue

                if len(cand_counts) <= K_CUTOFF:
                    selected_ids = [c for c, _ in cand_counts.most_common(K_CUTOFF)]
                else:
                    top_cands = [c for c, _ in cand_counts.most_common(60)]
                    scored = []
                    for int_id in top_cands:
                        c_name = pool_names[int_id]
                        if s1_name == c_name:
                            sim = 1.0
                        else:
                            len2 = len(c_name)
                            g2 = {c_name[i:i + 3] for i in range(len2 - 2)} if len2 >= 3 else {c_name}
                            u = s1_g1 | g2
                            j_char = len(s1_g1 & g2) / len(u) if u else 0.0
                            toks2 = set(c_name.split())
                            u_tok = s1_toks | toks2
                            j_tok = len(s1_toks & toks2) / len(u_tok) if u_tok else 0.0
                            sim = 0.6 * j_char + 0.4 * j_tok

                        scored.append((int_id, sim + 0.15 * cand_counts[int_id]))

                    scored.sort(key=lambda x: x[1], reverse=True)
                    selected_ids = [int_id for int_id, _ in scored[:K_CUTOFF]]

                # Persist both candidate entity_id and shared_key_count (cid:count)
                cand_str = ",".join(f"{pool_id_list[i]}:{cand_counts[i]}" for i in selected_ids)
                c_cand_count += len(selected_ids)
                lines.append(f"{s1_id}\t{cand_str}\n")
                c_s1_count += 1

                if len(lines) >= 5000:
                    f_out.writelines(lines)
                    f_out.flush()
                    lines = []

            if lines:
                f_out.writelines(lines)
                f_out.flush()

        print(f"  Completed {country}: {c_s1_count:,} entities in {time.time() - t_q:.1f}s.")
        total_s1 += c_s1_count
        total_cands += c_cand_count

        del pool_id_list, pool_names, index
        gc.collect()

    print(f"\nAdaptive blocking completed in {time.time() - t_start:.2f}s.")
    print(f"Saved {total_s1:,} entities with shared_key_counts to {out_file}.")

if __name__ == "__main__":
    main()
