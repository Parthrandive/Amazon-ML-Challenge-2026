import pandas as pd
from typing import Dict, Set

def load_mapping_from_tsv(path: str, col_name: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    mapping = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        val = row.get(col_name)
        if pd.isna(val) or not str(val).strip():
            mapping[s1] = set()
        else:
            mapping[s1] = {x.strip() for x in str(val).split(",") if x.strip()}
    return mapping

def main():
    print("Loading ground truth...")
    gt_mapping = load_mapping_from_tsv("data/raw/train_ground_truth.tsv", "matched_entity_ids")
    
    print("Loading candidate pairs...")
    cand_mapping = load_mapping_from_tsv("output/candidate_pairs.tsv", "candidate_entity_ids")
    
    total_gt_matches = 0
    total_found_in_candidates = 0
    missing_gt_entities = 0
    
    for s1_id, gt_set in gt_mapping.items():
        if not gt_set:
            continue
            
        total_gt_matches += len(gt_set)
        
        cand_set = cand_mapping.get(s1_id, set())
        
        hits = len(gt_set & cand_set)
        total_found_in_candidates += hits
        
        if hits < len(gt_set):
            missing_gt_entities += 1

    recall = (total_found_in_candidates / total_gt_matches) * 100 if total_gt_matches > 0 else 0
    
    print(f"\n--- Candidate Generation Audit ---")
    print(f"Total S1 entities with GT matches: {len([k for k,v in gt_mapping.items() if v])}")
    print(f"Total GT individual matches: {total_gt_matches}")
    print(f"Total GT matches found in candidates: {total_found_in_candidates}")
    print(f"Candidate Recall: {recall:.2f}%")
    print(f"S1 entities with missed matches: {missing_gt_entities}")

if __name__ == "__main__":
    main()
