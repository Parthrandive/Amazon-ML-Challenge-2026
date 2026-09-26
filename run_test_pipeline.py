"""
run_test_pipeline.py
End-to-End Pipeline for Amazon ML Challenge 2026 Test Set:
1. Preprocess test sources (test_source1, test_source2, test_source3) -> data/processed/test_source*_clean.tsv
2. Country-partitioned blocking (K=20) with adaptive-depth uncapping & shared_key_count persistence -> output/candidate_pairs.tsv
3. High-throughput batched model inference (LightGBM with shared_key_count at tuned threshold 0.6560) -> output/matching_results.tsv
4. Validate with student_resource/utils/validate_submission.py
5. Package final submission zip (submission.zip)
"""

import os
import sys
import time
import gc
import shutil
import array
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("amc2026/src"))

from preprocess import preprocess_source
from blocking import get_tight_blocking_keys, get_key_max_size
from generate_output import score_candidates_and_generate_matches

os.makedirs("output", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def stage1_preprocess_test():
    print("\n" + "=" * 60)
    print("STAGE 1: PREPROCESSING TEST DATASETS (India, US, France)")
    print("=" * 60)

    test_sources = [
        ("Test Source 1", "student_resource/dataset/test/test_source1.tsv", "data/processed/test_source1_clean.tsv"),
        ("Test Source 2", "student_resource/dataset/test/test_source2.tsv", "data/processed/test_source2_clean.tsv"),
        ("Test Source 3", "student_resource/dataset/test/test_source3.tsv", "data/processed/test_source3_clean.tsv"),
    ]

    for name, in_path, out_path in test_sources:
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000000:
            print(f"[{name}] Already preprocessed at {out_path} ({os.path.getsize(out_path):,} bytes). Skipping.")
            continue

        print(f"[{name}] Loading {in_path}...")
        t0 = time.time()
        df = pd.read_csv(in_path, sep="\t")
        print(f"  Loaded {len(df):,} rows in {time.time() - t0:.2f}s.")

        print(f"  Preprocessing (transliteration, diacritic stripping, suffix & address cleaning)...")
        t1 = time.time()
        clean_df = preprocess_source(df)
        print(f"  Preprocessed {len(clean_df):,} rows in {time.time() - t1:.2f}s.")

        print(f"  Saving to {out_path}...")
        t2 = time.time()
        clean_df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved in {time.time() - t2:.2f}s.")

        del df, clean_df
        gc.collect()


def stage2_blocking_test():
    print("\n" + "=" * 60)
    print("STAGE 2: COUNTRY-PARTITIONED ADAPTIVE BLOCKING ON TEST DATA (K=20)")
    print("=" * 60)

    out_clean_file = "output/candidate_pairs.tsv"
    out_counts_file = "output/candidate_pairs_with_counts.tsv"
    s1_clean_path = "data/processed/test_source1_clean.tsv"
    s2_clean_path = "data/processed/test_source2_clean.tsv"
    s3_clean_path = "data/processed/test_source3_clean.tsv"

    # Check distinct countries in test_source1 (raw file scan, no pandas needed)
    print("Detecting countries in test_source1...")
    country_set = set()
    s1_total = 0
    with open(s1_clean_path, "r", encoding="utf-8") as f:
        hdr = next(f).rstrip("\n").split("\t")
        c_idx = hdr.index("country")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > c_idx:
                country_set.add(parts[c_idx])
            s1_total += 1
    countries = sorted(country_set)
    print(f"Target countries in test set: {countries} ({s1_total:,} total entities)")

    K_CUTOFF = 20
    CHUNK_SIZE = 50000

    with open(out_clean_file, "w", encoding="utf-8") as f_clean, \
         open(out_counts_file, "w", encoding="utf-8") as f_counts:
        f_clean.write("source1_entity_id\tcandidate_entity_ids\n")
        f_counts.write("source1_entity_id\tcandidate_entity_ids\n")

    t_start = time.time()
    total_s1 = 0
    total_cands = 0

    for country in countries:
        print(f"\n--- Processing Partition: {country} ---")
        t_c = time.time()

        # Build adaptive-depth index for this country from S2 and S3
        # Uses raw file streaming instead of pandas for ~3-5x speedup
        pool_id_list = []
        pool_names = []
        index = defaultdict(lambda: array.array("I"))

        for pool_name, path in [("Source 2", s2_clean_path), ("Source 3", s3_clean_path)]:
            with open(path, "r", encoding="utf-8") as f:
                header = next(f).rstrip("\n").split("\t")
                id_idx = header.index("entity_id")
                country_idx = header.index("country")
                name_idx = header.index("business_name_clean")
                addr_idx = header.index("business_address_clean")
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= country_idx or parts[country_idx] != country:
                        continue
                    idx = len(pool_id_list)
                    pool_id_list.append(parts[id_idx])
                    c_name = parts[name_idx] if len(parts) > name_idx and parts[name_idx] not in ("", "nan", "None") else ""
                    pool_names.append(c_name)

                    c_addr = parts[addr_idx] if len(parts) > addr_idx else ""
                    keys = get_tight_blocking_keys(country, c_name, c_addr)
                    for k in keys:
                        lst = index[k]
                        max_cap = get_key_max_size(k)
                        if len(lst) < max_cap:
                            lst.append(idx)

        print(f"  Indexed {len(pool_id_list):,} {country} pool records ({len(index):,} keys) in {time.time() - t_c:.1f}s.")

        # Query S1 for this country — also raw file streaming
        t_q = time.time()
        c_s1_count = 0
        c_cand_count = 0

        with open(out_clean_file, "a", encoding="utf-8") as f_clean, \
             open(out_counts_file, "a", encoding="utf-8") as f_counts:
            clean_lines = []
            counts_lines = []

            with open(s1_clean_path, "r", encoding="utf-8") as f_s1:
                s1_header = next(f_s1).rstrip("\n").split("\t")
                s1_id_idx = s1_header.index("entity_id")
                s1_country_idx = s1_header.index("country")
                s1_name_idx = s1_header.index("business_name_clean")
                s1_addr_idx = s1_header.index("business_address_clean")

                for line in f_s1:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= s1_country_idx or parts[s1_country_idx] != country:
                        continue

                    s1_id = parts[s1_id_idx]
                    s1_name = parts[s1_name_idx] if len(parts) > s1_name_idx and parts[s1_name_idx] not in ("", "nan", "None") else ""
                    s1_addr = parts[s1_addr_idx] if len(parts) > s1_addr_idx else ""
                    s1_toks = set(s1_name.split())
                    len1 = len(s1_name)
                    s1_g1 = {s1_name[i:i + 3] for i in range(len1 - 2)} if len1 >= 3 else {s1_name}

                    keys = get_tight_blocking_keys(country, s1_name, s1_addr)
                    cand_counts = Counter()
                    for k in keys:
                        cand_counts.update(index.get(k, ()))

                    if not cand_counts:
                        clean_lines.append(f"{s1_id}\t\n")
                        counts_lines.append(f"{s1_id}\t\n")
                        c_s1_count += 1
                        continue

                    if len(cand_counts) <= K_CUTOFF:
                        selected_ids = [c for c, _ in cand_counts.most_common(K_CUTOFF)]
                    else:
                        top_cands = [c for c, _ in cand_counts.most_common(60)]
                        scored = []
                        for int_id in top_cands:
                            c_name = pool_names[int_id]
                            if s1_name == c_name:
                                sim = 1.0
                            else:
                                len2 = len(c_name)
                                g2 = {c_name[i:i + 3] for i in range(len2 - 2)} if len2 >= 3 else {c_name}
                                u = s1_g1 | g2
                                j_char = len(s1_g1 & g2) / len(u) if u else 0.0
                                toks2 = set(c_name.split())
                                u_tok = s1_toks | toks2
                                j_tok = len(s1_toks & toks2) / len(u_tok) if u_tok else 0.0
                                sim = 0.6 * j_char + 0.4 * j_tok

                            scored.append((int_id, sim + 0.15 * cand_counts[int_id]))

                        scored.sort(key=lambda x: x[1], reverse=True)
                        selected_ids = [int_id for int_id, _ in scored[:K_CUTOFF]]

                    cand_str_clean = ",".join(pool_id_list[i] for i in selected_ids)
                    cand_str_counts = ",".join(f"{pool_id_list[i]}:{cand_counts[i]}" for i in selected_ids)

                    c_cand_count += len(selected_ids)
                    clean_lines.append(f"{s1_id}\t{cand_str_clean}\n")
                    counts_lines.append(f"{s1_id}\t{cand_str_counts}\n")
                    c_s1_count += 1

                    if len(clean_lines) >= 10000:
                        f_clean.writelines(clean_lines)
                        f_clean.flush()
                        f_counts.writelines(counts_lines)
                        f_counts.flush()
                        clean_lines = []
                        counts_lines = []

                        if c_s1_count % 100000 == 0:
                            print(f"    Streamed {c_s1_count:,} {country} entities ({c_cand_count/c_s1_count:.1f} cands/entity)...", flush=True)

            if clean_lines:
                f_clean.writelines(clean_lines)
                f_clean.flush()
                f_counts.writelines(counts_lines)
                f_counts.flush()

        print(f"  Completed {country}: {c_s1_count:,} entities in {time.time() - t_q:.1f}s.")
        total_s1 += c_s1_count
        total_cands += c_cand_count

        del pool_id_list, pool_names, index
        gc.collect()


    print(f"\nAdaptive blocking completed in {time.time() - t_start:.2f}s.")
    print(f"Total S1 entities: {total_s1:,} | Total candidate pairs: {total_cands:,} (avg {total_cands/total_s1:.2f}/entity)")

    # Also symlink / copy to outputs/ for convenience
    os.makedirs("outputs", exist_ok=True)
    shutil.copyfile(out_clean_file, "outputs/candidate_pairs_test.tsv")


def stage3_inference():
    print("\n" + "=" * 60)
    print("STAGE 3: BATCHED MODEL INFERENCE ON TEST CANDIDATES (WITH SHARED_KEY_COUNT)")
    print("=" * 60)

    score_candidates_and_generate_matches(
        candidate_pairs_file="output/candidate_pairs.tsv",
        s1_clean_file="data/processed/test_source1_clean.tsv",
        s2_clean_file="data/processed/test_source2_clean.tsv",
        s3_clean_file="data/processed/test_source3_clean.tsv",
        model_path="models/matcher_lgbm.pkl",
        output_matching_file="output/matching_results.tsv",
        batch_entities=5000
    )


def stage4_validation():
    print("\n" + "=" * 60)
    print("STAGE 4: OFFICIAL COMPETITION SUBMISSION VALIDATOR")
    print("=" * 60)

    cmd = (
        "python3 student_resource/utils/validate_submission.py "
        "--matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv "
        "--test-dir student_resource/dataset/test"
    )
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print("ERROR: Validation failed! Please fix issues before packaging.")
        sys.exit(1)
    else:
        print("\n>>> VALIDATION SUCCESSFUL: output/matching_results.tsv PASSED ALL CHECKS! <<<")


def stage5_package_zip():
    print("\n" + "=" * 60)
    print("STAGE 5: PACKAGING FINAL SUBMISSION ARCHIVE (submission.zip)")
    print("=" * 60)

    staging_dir = "submission_staging"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    # 1. output/
    os.makedirs(os.path.join(staging_dir, "output"), exist_ok=True)
    shutil.copyfile("output/matching_results.tsv", os.path.join(staging_dir, "output", "matching_results.tsv"))
    shutil.copyfile("output/candidate_pairs.tsv", os.path.join(staging_dir, "output", "candidate_pairs.tsv"))

    # 2. code/business_entity_resolution/src and models
    code_src_dir = os.path.join(staging_dir, "code", "business_entity_resolution", "src")
    os.makedirs(code_src_dir, exist_ok=True)
    for fn in ["preprocess.py", "transliterate.py", "blocking.py", "features.py", "model.py", "evaluate.py", "generate_output.py", "token_idf.pkl"]:
        src_path = os.path.join("amc2026", "src", fn)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, os.path.join(code_src_dir, fn))

    code_models_dir = os.path.join(staging_dir, "code", "business_entity_resolution", "models")
    os.makedirs(code_models_dir, exist_ok=True)
    for fn in ["matcher_lgbm.pkl", "calibrator_isotonic.pkl", "token_idf.pkl"]:
        src_path = os.path.join("models", fn)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, os.path.join(code_models_dir, fn))

    # code README and requirements
    readme_src = "amc2026/README_SUBMISSION.md" if os.path.exists("amc2026/README_SUBMISSION.md") else "README.md"
    shutil.copyfile(readme_src, os.path.join(staging_dir, "code", "business_entity_resolution", "README.md"))
    with open(os.path.join(staging_dir, "code", "business_entity_resolution", "requirements.txt"), "w") as f:
        f.write("lightgbm>=4.0.0\npandas>=2.0.0\nnumpy>=1.24.0\nindic-transliteration>=2.3.0\nscikit-learn>=1.3.0\nrapidfuzz>=3.0.0\n")

    # 3. Documentation_template.md
    shutil.copyfile("student_resource/Documentation_template.md", os.path.join(staging_dir, "Documentation_template.md"))

    # 4. Create ZIP
    zip_out = "submission"
    shutil.make_archive(zip_out, "zip", staging_dir)
    zip_path = f"{zip_out}.zip"
    print(f"\nFinal submission archive created: {zip_path} ({os.path.getsize(zip_path):,} bytes)")

    # Clean up staging
    shutil.rmtree(staging_dir)
    print("Staging directory cleaned up.")


if __name__ == "__main__":
    t_all = time.time()
    stage1_preprocess_test()
    stage2_blocking_test()
    stage3_inference()
    stage4_validation()
    stage5_package_zip()
    print("\n" + "=" * 60)
    print(f"PIPELINE FULLY COMPLETE IN {time.time() - t_all:.2f}s!")
    print("=" * 60)
