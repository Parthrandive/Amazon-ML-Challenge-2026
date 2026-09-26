import duckdb
import time

t0 = time.time()
print("Connecting to DuckDB...")
con = duckdb.connect()

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
    matched_gt AS (
        SELECT DISTINCT gt.source1_entity_id, gt.target_id
        FROM gt_filtered gt
        INNER JOIN read_parquet('output/s1_keys_*.parquet') s1 ON gt.source1_entity_id = s1.entity_id
        INNER JOIN read_parquet('output/valid_keys.parquet') v ON s1.block_key = v.block_key
        INNER JOIN read_parquet('output/pool_s*_keys_*.parquet') p ON gt.target_id = p.entity_id AND s1.block_key = p.block_key
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

