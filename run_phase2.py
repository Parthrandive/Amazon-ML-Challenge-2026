"""
run_phase2.py
Re-runs preprocessing with updated transliteration pipeline, verifies residual counts,
and computes country breakdowns.
"""

import os
import sys
import time
import pandas as pd

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("amc2026/src"))
from preprocess import preprocess_source

sources = [
    ("Source 1", "data/raw/train_source1.tsv", "data/processed/source1_clean.tsv"),
    ("Source 2", "data/raw/train_source2.tsv", "data/processed/source2_clean.tsv"),
    ("Source 3", "data/raw/train_source3.tsv", "data/processed/source3_clean.tsv"),
]

country_splits = {}
residual_summaries = {}

for name, in_path, out_path in sources:
    print(f"\n{'='*30} Processing {name} {'='*30}")
    t0 = time.time()
    df = pd.read_csv(in_path, sep="\t")
    print(f"Loaded {len(df)} rows in {time.time() - t0:.2f}s.")

    t1 = time.time()
    df_clean = preprocess_source(df)
    print(f"Preprocessed {len(df_clean)} rows in {time.time() - t1:.2f}s.")

    t2 = time.time()
    df_clean.to_csv(out_path, sep="\t", index=False)
    print(f"Saved to {out_path} in {time.time() - t2:.2f}s.")

    total_rows = len(df_clean)
    residual_rows = df_clean[df_clean["translit_has_residual_script"] == True]
    non_latin_rows = df_clean[df_clean["name_script"] != "latin"]
    
    residual_summaries[name] = {
        "total_rows": total_rows,
        "non_latin_rows": len(non_latin_rows),
        "residual_rows": len(residual_rows),
    }

    country_counts = df_clean["country"].value_counts(dropna=False).to_dict()
    country_splits[name] = country_counts

    print(f"\n--- Summary for {name} ---")
    print(f"Total rows: {total_rows}")
    print(f"Country breakdown: {country_counts}")
    print(f"Non-latin script rows: {len(non_latin_rows)}")
    print(f"Residual script rows: {len(residual_rows)}")

print("\n\n" + "="*50)
print("FINAL RE-VERIFICATION SUMMARY:")
for name in ["Source 1", "Source 2", "Source 3"]:
    res = residual_summaries[name]
    cntry = country_splits[name]
    print(f"{name}:")
    print(f"  Country Split: {cntry}")
    print(f"  Total Rows: {res['total_rows']:,}")
    print(f"  Non-Latin Rows: {res['non_latin_rows']:,}")
    print(f"  Residual Rows: {res['residual_rows']}")
