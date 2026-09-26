import duckdb
import time

def main():
    print("Connecting to DuckDB...")
    con = duckdb.connect()
    
    # Enable jaro_winkler if needed, but it's built-in via text/string functions
    # DuckDB provides jaro_winkler_similarity(s1, s2), jaccard(s1, s2) and levenshtein(s1, s2)
    
    # We will read S1 and Pool into tables or just query parquet files.
    # To join candidates, we read candidate_pairs.tsv.
    # We explode candidate_pairs, join with S1, join with Pool, and compute features.
    
    query = """
    CREATE OR REPLACE VIEW cand_pairs AS
    SELECT 
        source1_entity_id,
        unnest(string_split(candidate_entity_ids, ',')) as candidate_entity_id
    FROM read_csv_auto('output/candidate_pairs.tsv', delim='\t', header=True)
    WHERE candidate_entity_ids != '';
    
    CREATE OR REPLACE VIEW s1_data AS
    SELECT 
        entity_id,
        coalesce(business_name_clean, '') as name_clean,
        coalesce(business_address_clean, '') as addr_clean
    FROM read_csv_auto('data/processed/source1_clean.tsv', delim='\t', header=True);
    
    CREATE OR REPLACE VIEW s2_data AS
    SELECT 
        entity_id,
        coalesce(business_name_clean, '') as name_clean,
        coalesce(business_address_clean, '') as addr_clean
    FROM read_csv_auto('data/processed/source2_clean.tsv', delim='\t', header=True);
    
    CREATE OR REPLACE VIEW s3_data AS
    SELECT 
        entity_id,
        coalesce(business_name_clean, '') as name_clean,
        coalesce(business_address_clean, '') as addr_clean
    FROM read_csv_auto('data/processed/source3_clean.tsv', delim='\t', header=True);
    
    CREATE OR REPLACE VIEW pool_data AS
    SELECT * FROM s2_data
    UNION ALL
    SELECT * FROM s3_data;
    
    CREATE OR REPLACE TABLE extracted_features AS
    SELECT 
        c.source1_entity_id,
        c.candidate_entity_id,
        
        -- Name features
        CASE WHEN s1.name_clean = p.name_clean THEN 1.0 ELSE 0.0 END as name_exact,
        jaro_winkler_similarity(s1.name_clean, p.name_clean) as name_jaro_winkler,
        jaccard(s1.name_clean, p.name_clean) as name_jaccard,
        abs(length(s1.name_clean) - length(p.name_clean)) as name_len_diff,
        CASE WHEN greatest(length(s1.name_clean), length(p.name_clean)) > 0 
             THEN least(length(s1.name_clean), length(p.name_clean))::DOUBLE / greatest(length(s1.name_clean), length(p.name_clean))
             ELSE 0.0 END as name_len_ratio,
             
        -- Address features
        CASE WHEN s1.addr_clean != '' AND p.addr_clean != '' THEN 1.0 ELSE 0.0 END as addr_both,
        CASE WHEN s1.addr_clean != '' AND p.addr_clean != '' THEN jaro_winkler_similarity(s1.addr_clean, p.addr_clean) ELSE 0.0 END as addr_jaro_winkler,
        CASE WHEN s1.addr_clean != '' AND p.addr_clean != '' THEN jaccard(s1.addr_clean, p.addr_clean) ELSE 0.0 END as addr_jaccard,
        
        -- Meta
        CASE WHEN c.candidate_entity_id LIKE 'S2-%' THEN 1.0 ELSE 0.0 END as is_s2,
        CASE WHEN c.candidate_entity_id LIKE 'S3-%' THEN 1.0 ELSE 0.0 END as is_s3
        
    FROM cand_pairs c
    INNER JOIN s1_data s1 ON c.source1_entity_id = s1.entity_id
    INNER JOIN pool_data p ON c.candidate_entity_id = p.entity_id;
    """
    
    print("Executing DuckDB feature extraction query...")
    t0 = time.time()
    con.execute(query)
    print(f"Features computed and saved to table extracted_features in {time.time() - t0:.2f}s.")
    
    con.execute("COPY extracted_features TO 'output/duckdb_features.parquet' (FORMAT PARQUET)")
    print("Saved features to output/duckdb_features.parquet")

if __name__ == "__main__":
    main()
