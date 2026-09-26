"""Measure how much current blocking loss is attributable to country extraction.

The script deliberately derives country from raw name/address fields.  It does not use
the source-provided country label, which would conceal the failure this audit is meant
to measure.  Run it before rebuilding candidates:

    python3 audit_country_extraction.py
"""

import csv
import os
import sys
import tempfile
from collections import Counter

import pandas as pd

sys.path.append(os.path.abspath("amc2026/src"))
from preprocess import extract_country


GT_FILE = "data/raw/train_ground_truth.tsv"
CANDIDATES_FILE = "outputs/candidate_pairs.tsv"
SOURCES = (
    "data/raw/train_source1.tsv",
    "data/raw/train_source2.tsv",
    "data/raw/train_source3.tsv",
)
CHUNK_SIZE = 250_000


def load_candidates():
    """Return the current candidate set for every Source 1 entity."""
    candidates = {}
    with open(CANDIDATES_FILE, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            candidates[row["source1_entity_id"]] = frozenset(
                value for value in row["candidate_entity_ids"].split(",") if value
            )
    return candidates


def write_missing_edges(candidates, path):
    """Write missed edges and return their endpoint ID sets plus baseline totals."""
    s1_ids, pool_ids = set(), set()
    totals = Counter()
    with open(GT_FILE, newline="", encoding="utf-8") as gt, open(path, "w", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(gt, delimiter="\t")
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(("source1_entity_id", "matched_entity_id"))
        for row in reader:
            source1_id = row["source1_entity_id"]
            current = candidates.get(source1_id, frozenset())
            for matched_id in filter(None, row["matched_entity_ids"].split(",")):
                totals["true_links"] += 1
                if matched_id not in current:
                    writer.writerow((source1_id, matched_id))
                    s1_ids.add(source1_id)
                    pool_ids.add(matched_id)
                    totals["missing_links"] += 1
    return s1_ids, pool_ids, totals


def extract_needed_countries(s1_ids, pool_ids):
    """Stream source files and retain derived countries only for missed-edge endpoints."""
    countries = {}
    wanted = s1_ids | pool_ids
    for source in SOURCES:
        for chunk in pd.read_csv(
            source,
            sep="\t",
            usecols=["entity_id", "business_name", "business_address"],
            chunksize=CHUNK_SIZE,
        ):
            chunk = chunk[chunk["entity_id"].isin(wanted)]
            for row in chunk.itertuples(index=False):
                countries[row.entity_id] = extract_country(
                    row.business_address if isinstance(row.business_address, str) else "",
                    row.business_name if isinstance(row.business_name, str) else "",
                )
    missing = wanted - countries.keys()
    if missing:
        raise RuntimeError(f"Could not find {len(missing):,} missed-edge endpoints in the source files")
    return countries


def main():
    if not os.path.exists(CANDIDATES_FILE):
        raise SystemExit(f"Candidate file not found: {CANDIDATES_FILE}")
    print("Loading current candidate sets...")
    candidates = load_candidates()
    print(f"Loaded {len(candidates):,} candidate rows.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
        missing_file = tmp.name
    try:
        s1_ids, pool_ids, totals = write_missing_edges(candidates, missing_file)
        print(f"Current recall: {(totals['true_links'] - totals['missing_links']) / totals['true_links']:.2%}")
        print(f"Missing links: {totals['missing_links']:,}; unique endpoints: "
              f"{len(s1_ids):,} Source 1, {len(pool_ids):,} pool.")
        print("Deriving countries for missed-edge endpoints from raw addresses/names...")
        countries = extract_needed_countries(s1_ids, pool_ids)

        breakdown = Counter()
        pair_breakdown = Counter()
        with open(missing_file, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                left = countries[row["source1_entity_id"]]
                right = countries[row["matched_entity_id"]]
                breakdown["unknown_either"] += left == "unknown" or right == "unknown"
                breakdown["country_mismatch"] += left != right
                pair_breakdown[(left, right)] += 1

        missing = totals["missing_links"]
        print("\nCOUNTRY-EXTRACTION DIAGNOSIS FOR CURRENTLY MISSED TRUE LINKS")
        print(f"unknown on either endpoint: {breakdown['unknown_either']:,} "
              f"({breakdown['unknown_either'] / missing:.2%})")
        print(f"derived-country mismatch:    {breakdown['country_mismatch']:,} "
              f"({breakdown['country_mismatch'] / missing:.2%})")
        print("Top derived-country endpoint pairs:")
        for (left, right), count in pair_breakdown.most_common():
            print(f"  {left:7} -> {right:7} {count:,} ({count / missing:.2%})")
    finally:
        os.unlink(missing_file)


if __name__ == "__main__":
    main()
