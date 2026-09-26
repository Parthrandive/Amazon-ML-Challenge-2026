import pandas as pd
import glob
import time
from collections import defaultdict

t0 = time.time()
print("Loading GT...")
gt = pd.read_csv('data/raw/train_ground_truth.tsv', sep='\t')
gt_pairs = []
for row in gt.itertuples(index=False):
    if pd.notna(row.matched_entity_ids) and row.matched_entity_ids != '':
        s1 = row.source1_entity_id
        for tgt in str(row.matched_entity_ids).split(','):
            gt_pairs.append((s1, tgt.strip()))

gt_s1_ids = {p[0] for p in gt_pairs}
gt_p_ids = {p[1] for p in gt_pairs}
print(f"Loaded {len(gt_pairs)} GT pairs.")

print("Loading valid keys...")
valid_keys = set(pd.read_parquet('output/valid_keys.parquet')['block_key'])
print(f"Loaded {len(valid_keys)} valid keys.")

print("Loading S1 keys...")
s1_keys_map = defaultdict(list)
df_s1 = pd.read_parquet('output/s1_keys_0.parquet')
df_s1 = df_s1[df_s1['entity_id'].isin(gt_s1_ids) & df_s1['block_key'].isin(valid_keys)]
for row in df_s1.itertuples(index=False):
    s1_keys_map[row.entity_id].append((row.block_key, row.country_extracted))

print("Loading Pool keys...")
pool_keys_map = defaultdict(list)
for f in glob.glob('output/pool_s*_keys_*.parquet'):
    df_p = pd.read_parquet(f)
    df_p = df_p[df_p['entity_id'].isin(gt_p_ids) & df_p['block_key'].isin(valid_keys)]
    for row in df_p.itertuples(index=False):
        pool_keys_map[row.entity_id].append((row.block_key, row.country_extracted))

print("Evaluating recall...")
matched = 0
for s1, p in gt_pairs:
    if s1 not in s1_keys_map or p not in pool_keys_map:
        continue
    
    s1_entries = s1_keys_map[s1]
    p_entries = pool_keys_map[p]
    
    # We want to see if any block_key matches and satisfies country rules
    match_found = False
    for k1, c1 in s1_entries:
        for k2, c2 in p_entries:
            if k1 == k2:
                # Country check
                c1 = "unknown" if pd.isna(c1) or c1 == "" else str(c1)
                c2 = "unknown" if pd.isna(c2) or c2 == "" else str(c2)
                
                if k1.startswith("fallback_"):
                    if c1 == "unknown" or c2 == "unknown":
                        match_found = True
                        break
                else:
                    if (c1 == c2 and c1 != "unknown") or (c1 == "unknown" or c2 == "unknown"):
                        match_found = True
                        break
        if match_found:
            break
            
    if match_found:
        matched += 1

recall = matched / len(gt_pairs) * 100 if gt_pairs else 0
print(f"Total GT Pairs: {len(gt_pairs)}")
print(f"Matched Pairs: {matched}")
print(f"Recall: {recall:.2f}%")
print(f"Time: {time.time() - t0:.2f}s")

