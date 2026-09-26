"""
generate_output.py
Generates the final matching_results.tsv from candidate pairs using the trained LightGBM model.
Optimized for high-throughput batched matrix inference:
  - 32 features including rapidfuzz string similarity and TF-IDF weighted tokens
  - Country-calibrated two-tier thresholds (T1 primary match, T2 multi-match)
  - End-to-end Bipartite Conflict Resolution (Max-probability attribution, zero duplicate claims)
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("amc2026/src"))
from features import extract_pair_features, FEATURE_NAMES
from model import load_matcher_model_full


def score_candidates_and_generate_matches(
    candidate_pairs_file: str,
    s1_clean_file: str,
    s2_clean_file: str,
    s3_clean_file: str,
    model_path: str,
    output_matching_file: str,
    batch_entities: int = 10000
):
    """
    Reads candidate_pairs.tsv, scores candidates using trained LightGBM model,
    applies country-calibrated two-tier thresholds, resolves bipartite conflicts,
    and writes matching_results.tsv.
    """
    print(f"Loading model from {model_path}...")
    clf, threshold, country_thresholds, margin_params = load_matcher_model_full(model_path)
    print(f"Loaded model ({clf.n_features_} features) with default threshold: {threshold:.4f}")
    
    # Configure Two-Tier thresholds per country:
    # T1: primary admission (gatekeeper for singleton vs match)
    # T2: secondary admission (strict guard for subsequent multi-matches)
    # France is Latin-script with structured addresses, aligned with US
    t1_thresholds = {
        "India": country_thresholds.get("India", 0.6480),
        "US": country_thresholds.get("US", 0.8660),
        "France": country_thresholds.get("US", 0.8660),
    }
    t2_thresholds = {
        "India": 0.8000,
        "US": 0.9200,
        "France": 0.9200,
    }
    print(f"Active T1 (Primary) Thresholds:    {t1_thresholds}")
    print(f"Active T2 (Multi-match) Thresholds: {t2_thresholds}")

    print("Loading Source 1 records...")
    t0 = time.time()
    s1_dict = {}
    with open(s1_clean_file, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        id_idx = header.index("entity_id")
        country_idx = header.index("country") if "country" in header else header.index("country_extracted")
        name_idx = header.index("business_name_clean")
        addr_idx = header.index("business_address_clean")
        postal_idx = header.index("postal_code") if "postal_code" in header else -1
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_dict[parts[id_idx]] = {
                "entity_id": parts[id_idx],
                "country": parts[country_idx] if len(parts) > country_idx and parts[country_idx] else "unknown",
                "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                "postal_code": parts[postal_idx] if (postal_idx != -1 and len(parts) > postal_idx) else "",
            }
    print(f"Loaded {len(s1_dict):,} S1 records in {time.time() - t0:.2f}s.")

    print("Collecting candidate IDs from candidate pairs...")
    needed_cand_ids = set()
    with open(candidate_pairs_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1]:
                for item in parts[1].split(","):
                    cid = item.split(":", 1)[0] if ":" in item else item
                    if cid:
                        needed_cand_ids.add(cid)
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
            postal_idx = header.index("postal_code") if "postal_code" in header else -1
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[id_idx]
                if eid in needed:
                    pool_dict[eid] = {
                        "entity_id": eid,
                        "country": parts[country_idx] if len(parts) > country_idx else "",
                        "business_name_clean": parts[name_idx] if len(parts) > name_idx else "",
                        "business_address_clean": parts[addr_idx] if len(parts) > addr_idx else "",
                        "postal_code": parts[postal_idx] if (postal_idx != -1 and len(parts) > postal_idx) else "",
                    }
                    needed.discard(eid)
                    if not needed:
                        break
    print(f"Loaded {len(pool_dict):,} candidate pool records in {time.time() - t1:.2f}s.")

    print("Scoring candidate pairs with batched matrix inference...")
    os.makedirs(os.path.dirname(os.path.abspath(output_matching_file)), exist_ok=True)
    t2 = time.time()

    n_processed = 0
    raw_entity_matches: Dict[str, List[Tuple[str, float]]] = {}  # s1_id -> list of (cid, prob)
    s1_order: List[str] = []
    cand_claims = defaultdict(list)  # cid -> list of (s1_id, prob)

    with open(candidate_pairs_file, "r", encoding="utf-8") as f_in:
        next(f_in)  # skip header

        batch_s1 = []
        batch_features = []
        entity_slices = []

        def flush_batch():
            nonlocal n_processed, batch_s1, batch_features, entity_slices
            if not batch_s1:
                return

            if batch_features:
                X = np.array(batch_features, dtype=np.float32)
                probs = clf.predict_proba(X)[:, 1]
            else:
                probs = np.array([])

            for s1_id, start_idx, end_idx, cand_ids in entity_slices:
                n_processed += 1
                if start_idx == end_idx:
                    raw_entity_matches[s1_id] = []
                    continue

                s1_c = s1_dict.get(s1_id, {}).get("country", "US")
                t1 = t1_thresholds.get(s1_c, threshold)
                t2 = t2_thresholds.get(s1_c, 0.9200)

                # Pair candidate with predicted probability, sorted desc
                ranked_cands = sorted(
                    [(cid, float(probs[idx])) for idx, cid in zip(range(start_idx, end_idx), cand_ids)],
                    key=lambda x: x[1],
                    reverse=True
                )

                # Two-tier admission logic:
                # Rank 0 must pass T1. If not, reject all.
                # Rank >= 1 must pass T2.
                admitted = []
                for rank, (cid, p) in enumerate(ranked_cands):
                    if rank == 0:
                        if p >= t1:
                            admitted.append((cid, p))
                            cand_claims[cid].append((s1_id, p))
                        else:
                            break
                    else:
                        if p >= t2:
                            admitted.append((cid, p))
                            cand_claims[cid].append((s1_id, p))

                raw_entity_matches[s1_id] = admitted

            batch_s1 = []
            batch_features = []
            entity_slices = []

        for line in f_in:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            s1_order.append(s1_id)
            cand_str = parts[1] if len(parts) > 1 else ""
            cand_items = [c.strip() for c in cand_str.split(",") if c.strip()]

            s1_row = s1_dict.get(s1_id)
            if not s1_row or not cand_items:
                entity_slices.append((s1_id, 0, 0, []))
                batch_s1.append(s1_id)
            else:
                start_idx = len(batch_features)
                valid_cands = []
                for rank, item in enumerate(cand_items):
                    if ":" in item:
                        cid, count_str = item.split(":", 1)
                        count = float(count_str)
                    else:
                        cid = item
                        count = 1.0

                    cand_row = pool_dict.get(cid)
                    if cand_row:
                        feat = extract_pair_features(s1_row, cand_row, cand_rank=rank, shared_key_count=count)
                        batch_features.append([feat[f] for f in FEATURE_NAMES])
                        valid_cands.append(cid)
                end_idx = len(batch_features)
                entity_slices.append((s1_id, start_idx, end_idx, valid_cands))
                batch_s1.append(s1_id)

            if len(batch_s1) >= batch_entities:
                flush_batch()
                if n_processed % 100000 == 0:
                    elapsed = time.time() - t2
                    rate = n_processed / elapsed if elapsed > 0 else 0
                    print(f"  Scored {n_processed:,} entities | {rate:,.0f} entities/s")

        flush_batch()

    print(f"\nScoring completed in {time.time() - t2:.2f}s for {n_processed:,} entities.")

    # ------------------------------------------------------------------
    # Bipartite Conflict Resolution (Max-Probability Exclusive Assignment)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESOLVING BIPARTITE CONFLICTS (1-TO-1 CANDIDATE EXCLUSIVITY)")
    print("=" * 60)
    t3 = time.time()

    conflicted = {cid: claims for cid, claims in cand_claims.items() if len(claims) > 1}
    total_conflicted_links = sum(len(claims) for claims in conflicted.values())
    print(f"Conflicted candidates (claimed by >1 S1): {len(conflicted):,}")
    print(f"Total overlapping candidate links:       {total_conflicted_links:,}")

    # Determine highest-probability winner for each candidate
    cand_winner = {}
    for cid, claims in conflicted.items():
        sorted_claims = sorted(claims, key=lambda x: x[1], reverse=True)
        cand_winner[cid] = sorted_claims[0][0]  # winning s1_id

    # Filter out duplicate claims from losers and write final output
    print(f"Writing resolved matching results to {output_matching_file}...")
    dropped_count = 0
    n_matched = 0
    total_final_links = 0

    with open(output_matching_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        out_lines = []

        for s1_id in s1_order:
            cands = raw_entity_matches.get(s1_id, [])
            if not cands:
                out_lines.append(f"{s1_id}\t\n")
                continue

            filtered_cands = []
            for cid, prob in cands:
                if cid in cand_winner:
                    if cand_winner[cid] == s1_id:
                        filtered_cands.append(cid)
                    else:
                        dropped_count += 1
                else:
                    filtered_cands.append(cid)

            if filtered_cands:
                out_lines.append(f"{s1_id}\t{','.join(filtered_cands)}\n")
                n_matched += 1
                total_final_links += len(filtered_cands)
            else:
                out_lines.append(f"{s1_id}\t\n")

            if len(out_lines) >= 50000:
                f_out.writelines(out_lines)
                out_lines = []

        if out_lines:
            f_out.writelines(out_lines)

    print(f"\nBipartite resolution completed in {time.time() - t3:.2f}s!")
    print(f"Dropped duplicate false-positive claims: {dropped_count:,}")
    print(f"Total entities scored:   {len(s1_order):,}")
    print(f"Entities with matches:   {n_matched:,} ({n_matched/len(s1_order)*100:.2f}%)")
    print(f"Singletons (no matches): {len(s1_order) - n_matched:,} ({(len(s1_order) - n_matched)/len(s1_order)*100:.2f}%)")
    print(f"Total matched pairs:     {total_final_links:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate matching_results.tsv from candidates")
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv", help="Path to candidates file")
    parser.add_argument("--s1", default="data/processed/test_source1_clean.tsv", help="Path to test_source1_clean.tsv")
    parser.add_argument("--s2", default="data/processed/test_source2_clean.tsv", help="Path to test_source2_clean.tsv")
    parser.add_argument("--s3", default="data/processed/test_source3_clean.tsv", help="Path to test_source3_clean.tsv")
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
