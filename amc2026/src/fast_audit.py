import duckdb
import time

def main():
    print("Connecting to DuckDB...", flush=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    
    t0 = time.time()
    
    print("Loading Ground Truth...", flush=True)
    con.execute("""
        CREATE OR REPLACE VIEW gt AS 
        SELECT 
            source1_entity_id, 
            unnest(string_split(matched_entity_ids, ',')) as target_id 
        FROM read_csv_auto('data/raw/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids != '';
        
        CREATE OR REPLACE VIEW gt_filtered AS 
        SELECT * FROM gt WHERE target_id != '' AND target_id IS NOT NULL AND target_id != 'nan';
    """)

    print("Extracting subset of S1 keys relevant to GT...", flush=True)
    con.execute("""
        CREATE OR REPLACE VIEW s1_subset AS
        SELECT s1.* 
        FROM read_parquet('output/s1_keys_*.parquet') s1
        INNER JOIN gt_filtered gt ON s1.entity_id = gt.source1_entity_id;
    """)

    print("Extracting subset of Pool keys relevant to GT...", flush=True)
    con.execute("""
        CREATE OR REPLACE VIEW pool_subset AS
        SELECT p.* 
        FROM read_parquet('output/pool_s*_keys_*.parquet') p
        INNER JOIN gt_filtered gt ON p.entity_id = gt.target_id;
    """)

    print("Extracting overlapping keys...", flush=True)
    con.execute("""
        CREATE OR REPLACE VIEW overlap_keys AS
        SELECT DISTINCT s1.block_key
        FROM s1_subset s1
        INNER JOIN pool_subset p ON s1.block_key = p.block_key;
    """)

    print("Computing counts for overlapping keys from the FULL pool...", flush=True)
    con.execute("""
        CREATE OR REPLACE VIEW valid_keys AS
        SELECT p.block_key
        FROM read_parquet('output/pool_s*_keys_*.parquet') p
        INNER JOIN overlap_keys o ON p.block_key = o.block_key
        GROUP BY p.block_key
        HAVING count(*) <= 5000;
    """)

    print("Evaluating recall...", flush=True)
    query = """
        WITH matched_gt AS (
            SELECT DISTINCT gt.source1_entity_id, gt.target_id
            FROM gt_filtered gt
            INNER JOIN s1_subset s1 ON gt.source1_entity_id = s1.entity_id
            INNER JOIN pool_subset p ON s1.block_key = p.block_key AND gt.target_id = p.entity_id
            INNER JOIN valid_keys v ON s1.block_key = v.block_key
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
    
    print(f"Audit completed in {time.time() - t0:.2f}s.", flush=True)
    print(f"Total Ground Truth Pairs: {total}", flush=True)
    print(f"Matched Candidate Pairs: {matched}", flush=True)
    print(f"Candidate Generation Recall: {recall:.2f}%", flush=True)

if __name__ == "__main__":
    main()
