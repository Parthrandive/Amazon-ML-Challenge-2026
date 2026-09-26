import pandas as pd
from blocking import get_tight_blocking_keys, get_fallback_blocking_keys
import glob
import time
from collections import defaultdict

def main():
    t0 = time.time()
    print("Loading Ground Truth...")
    gt = pd.read_csv("data/raw/train_ground_truth.tsv", sep="\t")
    gt_pairs = []
    for row in gt.itertuples(index=False):
        s1 = row.source1_entity_id
        if pd.notna(row.matched_entity_ids) and row.matched_entity_ids != "":
            for tgt in str(row.matched_entity_ids).split(","):
                gt_pairs.append((s1, tgt.strip()))
                
    gt_pairs = [(s1, s2) for s1, s2 in gt_pairs if s1 and s2]
    print(f"Loaded {len(gt_pairs)} ground truth pairs.")
    
    s1_targets = {p[0] for p in gt_pairs}
    pool_targets = {p[1] for p in gt_pairs}
    
    print("Loading entity data for GT...")
    s1_df = pd.read_csv("data/processed/source1_clean.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country_extracted"])
    s1_df = s1_df[s1_df["entity_id"].isin(s1_targets)]
    
    pool_df = []
    for src in ["source2", "source3"]:
        df = pd.read_csv(f"data/processed/{src}_clean.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country_extracted"])
        pool_df.append(df[df["entity_id"].isin(pool_targets)])
    pool_df = pd.concat(pool_df)
    
    print("Generating keys for GT entities...")
    def get_keys(row):
        c_ext = str(row.country_extracted) if pd.notna(row.country_extracted) and str(row.country_extracted) != "nan" else "unknown"
        name = str(row.business_name) if pd.notna(row.business_name) and str(row.business_name) != "nan" else ""
        addr = str(row.business_address) if pd.notna(row.business_address) and str(row.business_address) != "nan" else ""
        if c_ext in ["India", "US", "France"]:
            keys = get_tight_blocking_keys(c_ext, name, addr)
            keys.extend(get_fallback_blocking_keys(name))
        else:
            keys = get_fallback_blocking_keys(name)
        return keys, c_ext

    s1_keys = defaultdict(list)
    s1_countries = {}
    for row in s1_df.itertuples(index=False):
        keys, c = get_keys(row)
        s1_keys[row.entity_id] = keys
        s1_countries[row.entity_id] = c
        
    pool_keys = defaultdict(list)
    pool_countries = {}
    for row in pool_df.itertuples(index=False):
        keys, c = get_keys(row)
        pool_keys[row.entity_id] = keys
        pool_countries[row.entity_id] = c
        
    # Find overlapping keys
    relevant_keys = set()
    for s1_id, p_id in gt_pairs:
        s1_k = set(s1_keys.get(s1_id, []))
        p_k = set(pool_keys.get(p_id, []))
        relevant_keys.update(s1_k & p_k)
        
    print(f"Found {len(relevant_keys)} distinct overlapping keys. Scanning parquets for frequencies...")
    
    key_counts = defaultdict(int)
    parquet_files = glob.glob("output/pool_s*_keys_*.parquet")
    for f in parquet_files:
        df = pd.read_parquet(f, columns=["block_key"])
        mask = df["block_key"].isin(relevant_keys)
        counts = df[mask]["block_key"].value_counts()
        for k, v in counts.items():
            key_counts[k] += v
            
    print("Evaluating final recall...")
    matched = 0
    for s1_id, p_id in gt_pairs:
        s1_k = s1_keys.get(s1_id, [])
        p_k = pool_keys.get(p_id, [])
        s1_c = s1_countries.get(s1_id, "unknown")
        p_c = pool_countries.get(p_id, "unknown")
        
        shared = set(s1_k) & set(p_k)
        valid_match = False
        for k in shared:
            if key_counts[k] <= 5000:
                if k.startswith("fallback_"):
                    if s1_c == "unknown" or p_c == "unknown":
                        valid_match = True
                        break
                else:
                    if (s1_c == p_c and s1_c != "unknown") or (s1_c == "unknown" or p_c == "unknown"):
                        valid_match = True
                        break
        if valid_match:
            matched += 1
            
    recall = matched / len(gt_pairs) * 100 if gt_pairs else 0
    print(f"Audit completed in {time.time() - t0:.2f}s.")
    print(f"Total Ground Truth Pairs: {len(gt_pairs)}")
    print(f"Matched Candidate Pairs: {matched}")
    print(f"Candidate Generation Recall: {recall:.2f}%")

if __name__ == "__main__":
    main()
