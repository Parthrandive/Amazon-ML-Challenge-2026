import os
import time
import pandas as pd
import duckdb
from blocking import get_tight_blocking_keys, get_fallback_blocking_keys

def process_chunk(df: pd.DataFrame, chunk_idx: int, prefix: str):
    records = []
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country_extracted"] = df["country_extracted"].fillna("unknown")

    for row in df.itertuples(index=False):
        eid = row.entity_id
        c_ext = row.country_extracted
        name = row.business_name
        addr = row.business_address

        if c_ext in ["India", "US", "France"]:
            keys = get_tight_blocking_keys(c_ext, name, addr)
            keys.extend(get_fallback_blocking_keys(name))
        else:
            keys = get_fallback_blocking_keys(name)
        
        for k in keys:
            records.append((eid, k, c_ext))
            
    out_df = pd.DataFrame(records, columns=["entity_id", "block_key", "country_extracted"])
    out_file = f"output/{prefix}_keys_{chunk_idx}.parquet"
    out_df.to_parquet(out_file, index=False)
    return len(out_df)

def main():
    os.makedirs("output", exist_ok=True)

    print("Extracting keys for S1...")
    t0 = time.time()
    s1 = pd.read_csv("data/processed/source1_clean.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country_extracted"])
    s1_keys = process_chunk(s1, 0, "s1")
    print(f"S1 keys extracted in {time.time() - t0:.2f}s. Total keys: {s1_keys}")
    del s1
    
    print("Extracting keys for S2 & S3 (pool)...")
    t0 = time.time()
    s2 = pd.read_csv("data/processed/source2_clean.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country_extracted"])
    pool_keys = 0
    # Process S2 in chunks
    chunk_size = 1000000
    for i in range(0, len(s2), chunk_size):
        chunk = s2.iloc[i:i+chunk_size]
        pool_keys += process_chunk(chunk, i, "pool_s2")
    del s2
    
    s3 = pd.read_csv("data/processed/source3_clean.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country_extracted"])
    for i in range(0, len(s3), chunk_size):
        chunk = s3.iloc[i:i+chunk_size]
        pool_keys += process_chunk(chunk, i, "pool_s3")
    del s3

    print(f"Pool keys extracted in {time.time() - t0:.2f}s. Total keys: {pool_keys}")
    
    print("Connecting to DuckDB for candidate generation...")
    con = duckdb.connect()
    
    query = """
    WITH s1_all AS (
        SELECT * FROM read_parquet('output/s1_keys_*.parquet')
    ),
    pool_all AS (
        SELECT * FROM read_parquet('output/pool_s*_keys_*.parquet')
    ),
    pool_counts AS (
        SELECT block_key, count(*) as cnt
        FROM pool_all
        GROUP BY block_key
    ),
    capped_pool AS (
        SELECT p.block_key, p.entity_id as p_id, p.country_extracted as p_country
        FROM pool_all p
        JOIN pool_counts c ON p.block_key = c.block_key
        WHERE c.cnt <= 5000
    ),
    key_matches AS (
        SELECT 
            s1.entity_id as s1_id,
            p.p_id as p_id
        FROM s1_all s1
        INNER JOIN capped_pool p
            ON s1.block_key = p.block_key
        WHERE ((s1.country_extracted = p.p_country AND s1.country_extracted != 'unknown') 
               OR (s1.country_extracted = 'unknown' OR p.p_country = 'unknown')) 
        AND NOT (
            s1.block_key LIKE 'fallback_%' 
            AND s1.country_extracted != 'unknown' 
            AND p.p_country != 'unknown'
        )
    )
    SELECT 
        s1_id as source1_entity_id,
        string_agg(p_id, ',') as candidate_entity_ids
    FROM (
        SELECT s1_id, p_id
        FROM key_matches
        GROUP BY s1_id, p_id
    )
    GROUP BY s1_id
    """
    
    print("Executing DuckDB candidate join...")
    t2 = time.time()
    candidates = con.execute(query).df()
    print(f"Candidate pairs generated in {time.time() - t2:.2f}s. Total S1 matched: {len(candidates)}")
    
    candidates.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
    print("Saved output/candidate_pairs.tsv")

if __name__ == "__main__":
    main()
