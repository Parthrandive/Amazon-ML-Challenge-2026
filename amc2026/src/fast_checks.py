"""
fast_checks.py
Runs two critical investigations:
1. Exact match recovery after normalization across all ground truth pairs.
2. Numeric ID-alignment / delta / order pattern analysis between S1 and S2/S3.
"""

import sys
import time
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

def check_1_exact_matches():
    print("=" * 60)
    print("CHECK 1: EXACT MATCH RECOVERY (DUCKDB)")
    print("=" * 60)
    t0 = time.time()
    con = duckdb.connect()

    query = """
    WITH gt AS (
        SELECT 
            source1_entity_id as s1_id, 
            unnest(string_split(matched_entity_ids, ',')) as target_id 
        FROM read_csv_auto('data/raw/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids != '' AND matched_entity_ids IS NOT NULL
    ),
    gt_filtered AS (
        SELECT * FROM gt WHERE target_id != '' AND target_id IS NOT NULL AND target_id != 'nan'
    ),
    s1 AS (
        SELECT entity_id, business_name, business_address, business_name_clean, business_address_clean
        FROM read_csv_auto('data/processed/source1_clean.tsv', delim='\t', header=True)
    ),
    pool AS (
        SELECT entity_id, business_name, business_address, business_name_clean, business_address_clean
        FROM read_csv_auto('data/processed/source2_clean.tsv', delim='\t', header=True)
        UNION ALL
        SELECT entity_id, business_name, business_address, business_name_clean, business_address_clean
        FROM read_csv_auto('data/processed/source3_clean.tsv', delim='\t', header=True)
    )
    SELECT 
        count(*) as total_gt_pairs,
        count(CASE WHEN s1.business_name = pool.business_name THEN 1 END) as raw_name_exact,
        count(CASE WHEN s1.business_name_clean = pool.business_name_clean THEN 1 END) as clean_name_exact,
        count(CASE WHEN s1.business_address = pool.business_address AND s1.business_address != '' THEN 1 END) as raw_addr_exact,
        count(CASE WHEN s1.business_address_clean = pool.business_address_clean AND s1.business_address_clean != '' THEN 1 END) as clean_addr_exact,
        count(CASE WHEN s1.business_name_clean = pool.business_name_clean AND s1.business_address_clean = pool.business_address_clean AND s1.business_address_clean != '' THEN 1 END) as both_clean_exact,
        count(CASE WHEN s1.business_name_clean = pool.business_name_clean OR (s1.business_address_clean = pool.business_address_clean AND s1.business_address_clean != '') THEN 1 END) as either_clean_exact
    FROM gt_filtered gt
    JOIN s1 ON gt.s1_id = s1.entity_id
    JOIN pool ON gt.target_id = pool.entity_id;
    """
    
    print("Executing exact match join in DuckDB...")
    df = con.execute(query).df()
    elapsed = time.time() - t0
    print(f"Executed query in {elapsed:.2f}s.\n")

    total = df["total_gt_pairs"][0]
    print(f"Total Evaluated Ground Truth Pairs: {total:,}")
    print(f"  Raw Business Name Exact Match:          {df['raw_name_exact'][0]:,d} ({df['raw_name_exact'][0]/total*100:.2f}%)")
    print(f"  Clean Business Name Exact Match:        {df['clean_name_exact'][0]:,d} ({df['clean_name_exact'][0]/total*100:.2f}%)")
    print(f"  Raw Business Address Exact Match:       {df['raw_addr_exact'][0]:,d} ({df['raw_addr_exact'][0]/total*100:.2f}%)")
    print(f"  Clean Business Address Exact Match:     {df['clean_addr_exact'][0]:,d} ({df['clean_addr_exact'][0]/total*100:.2f}%)")
    print(f"  BOTH Clean Name & Address Exact Match:  {df['both_clean_exact'][0]:,d} ({df['both_clean_exact'][0]/total*100:.2f}%)")
    print(f"  EITHER Clean Name OR Address Exact:     {df['either_clean_exact'][0]:,d} ({df['either_clean_exact'][0]/total*100:.2f}%)")
    con.close()


def check_2_id_alignment():
    print("\n" + "=" * 60)
    print("CHECK 2: NUMERIC ID ALIGNMENT & STRUCTURAL LEAK AUDIT")
    print("=" * 60)
    t0 = time.time()

    gt_file = "data/raw/train_ground_truth.tsv"
    print(f"Sampling 500,000 ground truth pairs from {gt_file}...")

    s1_ints = []
    cand_ints = []
    cand_sources = []

    count = 0
    with open(gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= 1 or not parts[1] or parts[1] == "nan":
                continue
            s1_str = parts[0]
            if not s1_str.startswith("S1-"):
                continue
            s1_val = int(s1_str[3:])

            for cid in parts[1].split(","):
                cid = cid.strip()
                if not cid or "-" not in cid:
                    continue
                src, num_str = cid.split("-", 1)
                try:
                    c_val = int(num_str)
                except ValueError:
                    continue
                s1_ints.append(s1_val)
                cand_ints.append(c_val)
                cand_sources.append(src)
                count += 1
                if count >= 500000:
                    break
            if count >= 500000:
                break

    s1_arr = np.array(s1_ints, dtype=np.int64)
    cand_arr = np.array(cand_ints, dtype=np.int64)
    src_arr = np.array(cand_sources)

    print(f"Collected {len(s1_arr):,} pairs in {time.time() - t0:.2f}s.")
    print(f"S1 ID range:   min={s1_arr.min():,}, max={s1_arr.max():,}")
    print(f"Cand ID range: min={cand_arr.min():,}, max={cand_arr.max():,}")

    # Check for S2 vs S3
    for target_src in ["S2", "S3"]:
        mask = src_arr == target_src
        s1_sub = s1_arr[mask]
        c_sub = cand_arr[mask]
        if len(s1_sub) == 0:
            continue
        print(f"\n--- Analysis for Source 1 vs {target_src} (N={len(s1_sub):,}) ---")
        
        # Pearson and Spearman correlation
        pearson_r, p_val = stats.pearsonr(s1_sub, c_sub)
        spearman_rho, _ = stats.spearmanr(s1_sub[:20000], c_sub[:20000])
        print(f"  Pearson Correlation r:  {pearson_r:+.6f}")
        print(f"  Spearman Rank rho:      {spearman_rho:+.6f}")

        # Deltas
        deltas = c_sub - s1_sub
        print(f"  Delta (Cand - S1): mean={deltas.mean():.2f}, std={deltas.std():.2f}")
        print(f"  Delta Percentiles: p01={np.percentile(deltas, 1):.0f}, p25={np.percentile(deltas, 25):.0f}, p50={np.percentile(deltas, 50):.0f}, p75={np.percentile(deltas, 75):.0f}, p99={np.percentile(deltas, 99):.0f}")

        # Check top most frequent deltas
        delta_series = pd.Series(deltas)
        top_deltas = delta_series.value_counts().head(5)
        print("  Top 5 most frequent exact deltas:")
        for d, cnt in top_deltas.items():
            print(f"    Delta = {d:+d}: count = {cnt} ({cnt/len(deltas)*100:.3f}%)")

        # Bitwise XOR check
        xors = s1_sub ^ c_sub
        print(f"  XOR (S1 ^ Cand): mean={xors.mean():.2f}, min={xors.min()}, max={xors.max()}")

        # Ratio check
        ratios = c_sub / np.maximum(s1_sub, 1)
        print(f"  Ratio (Cand / S1): p25={np.percentile(ratios, 25):.3f}, p50={np.percentile(ratios, 50):.3f}, p75={np.percentile(ratios, 75):.3f}")

        # Check modulo 10, 100, 1000
        for mod in [10, 100, 1000]:
            mod_match = np.mean((s1_sub % mod) == (c_sub % mod))
            print(f"  Modulo {mod:4d} match rate: {mod_match*100:.2f}% (Expected random: {100/mod:.2f}%)")


if __name__ == "__main__":
    check_1_exact_matches()
    check_2_id_alignment()
