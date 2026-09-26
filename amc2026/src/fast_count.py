import duckdb
import pandas as pd
import glob
import time
from collections import defaultdict

t0 = time.time()
print("Connecting to DuckDB...")
con = duckdb.connect()

print("Extracting overlapping keys using INTERSECT...")
overlap_df = con.execute("""
    WITH gt AS (
        SELECT 
            source1_entity_id, 
            unnest(string_split(matched_entity_ids, ',')) as target_id 
        FROM read_csv_auto('data/raw/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids != ''
    ),
    gt_filtered AS (
        SELECT * FROM gt WHERE target_id != '' AND target_id IS NOT NULL AND target_id != 'nan'
    ),
    s1_subset AS (
        SELECT s1.block_key 
        FROM read_parquet('output/s1_keys_*.parquet') s1
        INNER JOIN gt_filtered gt ON s1.entity_id = gt.source1_entity_id
    ),
    pool_subset AS (
        SELECT p.block_key 
        FROM read_parquet('output/pool_s*_keys_*.parquet') p
        INNER JOIN gt_filtered gt ON p.entity_id = gt.target_id
    )
    SELECT block_key FROM s1_subset
    INTERSECT
    SELECT block_key FROM pool_subset
""").df()

overlap_keys = set(overlap_df['block_key'])
print(f"Found {len(overlap_keys)} distinct overlapping keys.")

print("Scanning pool parquets for counts (Pandas)...")
key_counts = defaultdict(int)
for f in glob.glob("output/pool_s*_keys_*.parquet"):
    df = pd.read_parquet(f, columns=["block_key"])
    df = df[df["block_key"].isin(overlap_keys)]
    counts = df["block_key"].value_counts()
    for k, v in counts.items():
        key_counts[k] += v

valid_keys = {k for k, v in key_counts.items() if v <= 5000}
print(f"Found {len(valid_keys)} valid overlapping keys (count <= 5000).")

pd.DataFrame({"block_key": list(valid_keys)}).to_parquet("output/valid_keys.parquet")

print("Evaluating final recall in DuckDB...")
query = """
    WITH gt AS (
        SELECT 
            source1_entity_id, 
            unnest(string_split(matched_entity_ids, ',')) as target_id 
        FROM read_csv_auto('data/raw/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids != ''
    ),
    gt_filtered AS (
        SELECT * FROM gt WHERE target_id != '' AND target_id IS NOT NULL AND target_id != 'nan'
    ),
    s1_subset AS (
        SELECT s1.* 
        FROM read_parquet('output/s1_keys_*.parquet') s1
        INNER JOIN gt_filtered gt ON s1.entity_id = gt.source1_entity_id
    ),
    pool_subset AS (
        SELECT p.* 
        FROM read_parquet('output/pool_s*_keys_*.parquet') p
        INNER JOIN gt_filtered gt ON p.entity_id = gt.target_id
    ),
    matched_gt AS (
        SELECT DISTINCT gt.source1_entity_id, gt.target_id
        FROM gt_filtered gt
        INNER JOIN s1_subset s1 ON gt.source1_entity_id = s1.entity_id
        INNER JOIN pool_subset p ON s1.block_key = p.block_key AND gt.target_id = p.entity_id
        INNER JOIN read_parquet('output/valid_keys.parquet') v ON s1.block_key = v.block_key
        WHERE ((s1.country_extracted = p.country_extracted AND s1.country_extracted != 'unknown') 
               OR (s1.country_extracted = 'unknown' OR p.country_extracted = 'unknown')) 
        AND NOT (
            s1.block_key LIKE 'fallback_%' 
            AND s1.country_extracted != 'unknown' 
            AND p.country_extracted != 'unknown'
        )
    )
    SELECT 
        (SELECT count(*) FROM gt_filtered) as total_pairs,
        (SELECT count(*) FROM matched_gt) as matched_pairs
"""
res = con.execute(query).df()
total = res['total_pairs'][0]
matched = res['matched_pairs'][0]
recall = matched / total * 100 if total > 0 else 0

print(f"Audit completed in {time.time() - t0:.2f}s.")
print(f"Total Ground Truth Pairs: {total}")
print(f"Matched Candidate Pairs: {matched}")
print(f"Candidate Generation Recall: {recall:.2f}%")

