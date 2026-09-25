"""
generate_output.py
Generates the final matching_results.tsv from candidate pairs using the trained model.
Ensures strict compliance with Amazon ML Challenge 2026 formatting rules.
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("amc2026/src"))
from features import extract_pair_features, FEATURE_NAMES
from model import load_matcher_model


def score_candidates_and_generate_matches(
    candidate_pairs_file: str,
    s1_clean_file: str,
    s2_clean_file: str,
    s3_clean_file: str,
    model_path: str,
    output_matching_file: str,
    batch_entities: int = 5000
):
    """
    Reads candidate_pairs.tsv, scores candidates using trained LightGBM model,
    and writes matching_results.tsv using high-throughput batched matrix inference.
    """
    print(f"Loading model from {model_path}...")
    clf, threshold = load_matcher_model(model_path)
    print(f"Loaded model with tuned decision threshold: {threshold:.3f}")

    print("Loading Source 1 records...")
    t0 = time.time()
    s1_dict = {}
    with open(s1_clean_file, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        country_idx = header.index("country")
        name_idx = header.index("business_name_clean")
        addr_idx = header.index("business_address_clean")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_dict[parts[id_idx]] = {
                "entity_id": parts[id_idx],
                "country": parts[country_idx] if len(parts) > country_idx else "",
                "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
            }
    print(f"Loaded {len(s1_dict):,} S1 records in {time.time() - t0:.2f}s.")

    print("Collecting candidate IDs from candidate pairs...")
    needed_cand_ids = set()
    with open(candidate_pairs_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1]:
                for c in parts[1].split(","):
                    if c:
                        needed_cand_ids.add(c)
    print(f"Total distinct candidate pool records needed: {len(needed_cand_ids):,}")

    print("Loading pool records from S2 and S3...")
    t1 = time.time()
    pool_dict = {}
    needed = set(needed_cand_ids)
    for path in [s2_clean_file, s3_clean_file]:
        with open(path, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\n").split("\t")
            id_idx = header.index("entity_id")
            country_idx = header.index("country")
            name_idx = header.index("business_name_clean")
            addr_idx = header.index("business_address_clean")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[id_idx]
                if eid in needed:
                    pool_dict[eid] = {
                        "entity_id": eid,
                        "country": parts[country_idx] if len(parts) > country_idx else "",
                        "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                        "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                    }
                    needed.discard(eid)
                    if not needed:
                        break
    print(f"Loaded {len(pool_dict):,} candidate pool records in {time.time() - t1:.2f}s.")

    print(f"Scoring candidates and generating {output_matching_file}...")
    os.makedirs(os.path.dirname(output_matching_file), exist_ok=True)
    t2 = time.time()

    n_processed = 0
    n_matched = 0
    total_matches_count = 0

    with open(candidate_pairs_file, "r", encoding="utf-8") as f_in, \
         open(output_matching_file, "w", encoding="utf-8") as f_out:
        next(f_in)  # skip header
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

        batch_s1 = []
        batch_features = []
        entity_slices = []
        out_lines = []

        def flush_batch():
            nonlocal n_processed, n_matched, total_matches_count, batch_s1, batch_features, entity_slices, out_lines
            if not batch_s1:
                return

            if batch_features:
                X = np.array(batch_features, dtype=np.float32)
                probs = clf.predict_proba(X)[:, 1]
            else:
                probs = np.array([])

            for s1_id, start_idx, end_idx, cand_ids in entity_slices:
                n_processed += 1
                if start_idx == end_idx:  # no candidates or invalid
                    out_lines.append(f"{s1_id}\t\n")
                    continue

                cand_probs = [
                    (cid, probs[idx]) for idx, cid in zip(range(start_idx, end_idx), cand_ids)
                    if probs[idx] >= threshold
                ]
                if cand_probs:
                    cand_probs.sort(key=lambda x: x[1], reverse=True)
                    out_lines.append(f"{s1_id}\t{','.join(c for c, _ in cand_probs)}\n")
                    n_matched += 1
                    total_matches_count += len(cand_probs)
                else:
                    out_lines.append(f"{s1_id}\t\n")

            f_out.writelines(out_lines)
            out_lines = []
            batch_s1 = []
            batch_features = []
            entity_slices = []

        for line in f_in:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]

            s1_row = s1_dict.get(s1_id)
            if not s1_row or not cand_ids:
                entity_slices.append((s1_id, 0, 0, []))
                batch_s1.append(s1_id)
            else:
                start_idx = len(batch_features)
                valid_cands = []
                for rank, cid in enumerate(cand_ids):
                    cand_row = pool_dict.get(cid)
                    if cand_row:
                        feat = extract_pair_features(s1_row, cand_row, cand_rank=rank)
                        batch_features.append([feat[f] for f in FEATURE_NAMES])
                        valid_cands.append(cid)
                end_idx = len(batch_features)
                entity_slices.append((s1_id, start_idx, end_idx, valid_cands))
                batch_s1.append(s1_id)

            if len(batch_s1) >= batch_entities:
                flush_batch()
                if n_processed % 100000 == 0:
                    print(f"  Scored {n_processed:,} entities ({n_matched:,} with matches, {total_matches_count:,} total links)...")

        flush_batch()

    print(f"\nMatching generation complete in {time.time() - t2:.2f}s.")
    print(f"Total entities scored: {n_processed:,}")
    print(f"Entities with matches: {n_matched:,} ({n_matched/n_processed*100:.2f}%)")
    print(f"Singletons (no matches): {n_processed - n_matched:,} ({(n_processed - n_matched)/n_processed*100:.2f}%)")
    print(f"Total matched pairs: {total_matches_count:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate matching_results.tsv from candidates")
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--s1", default="data/processed/source1_clean.tsv", help="Path to source1_clean.tsv")
    parser.add_argument("--s2", default="data/processed/source2_clean.tsv", help="Path to source2_clean.tsv")
    parser.add_argument("--s3", default="data/processed/source3_clean.tsv", help="Path to source3_clean.tsv")
    parser.add_argument("--model", default="models/matcher_lgbm.pkl", help="Path to trained matcher model")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Path to output matching_results.tsv")
    args = parser.parse_args()

    score_candidates_and_generate_matches(
        candidate_pairs_file=args.candidates,
        s1_clean_file=args.s1,
        s2_clean_file=args.s2,
        s3_clean_file=args.s3,
        model_path=args.model,
        output_matching_file=args.output
    )
