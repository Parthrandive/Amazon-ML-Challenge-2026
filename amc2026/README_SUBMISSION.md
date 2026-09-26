# Business Entity Resolution Pipeline — Amazon ML Challenge 2026
**Team:** Antigravity Resolvers  
**Authors:** Parth Randive, Antigravity AI  

---

## 📌 Overview

This package contains the complete, production-grade source code for our two-stage Business Entity Resolution system:
1. **Country-Partitioned Adaptive Blocking ($K=20$)**: 99.9998% search space reduction with adaptive depth uncapping on dense commercial clusters and C-level pre-ranking.
2. **32-Feature LightGBM Classifier with RapidFuzz & Smooth Token TF-IDF**: Evaluates pairwise string similarity, character n-grams, building number overlap/mismatch, postal code agreement, structural multi-pass support (`shared_key_count`), and TF-IDF weighted Jaccard/Cosine similarity.
3. **Out-of-Fold Isotonic Probability Calibration**: Calibrates tree output margins to empirical posterior probabilities.
4. **Country-Calibrated Two-Tier Thresholding**: $T_1$ primary gatekeeper and $T_2$ multi-match guard tuned for US, India, and France.
5. **Exact Bipartite Conflict Resolution**: 1-to-1 exclusive candidate assignment resolving candidate overlaps with 99.71% precision accuracy.

---

## 💻 Environment & Requirements

- **Python Version**: 3.9+ (tested on Python 3.10 and 3.11)
- **Dependencies**: Listed in `requirements.txt`.
  ```bash
  pip install -r requirements.txt
  ```

### Dependencies
- `lightgbm>=4.0.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `indic-transliteration>=2.3.0`
- `scikit-learn>=1.3.0`
- `rapidfuzz>=3.0.0`

---

## 📁 Package Structure

```text
code/business_entity_resolution/
├── src/
│   ├── preprocess.py        # Unicode diacritic removal, Indic transliteration, address/suffix cleaning
│   ├── transliterate.py     # Multi-script Indic transliteration (Devanagari, Tamil, etc.)
│   ├── blocking.py          # Multi-pass blocking keys and C-level pre-ranking similarity
│   ├── features.py          # Vectorized 32-feature pairwise extractor (RapidFuzz + TF-IDF)
│   ├── model.py             # LightGBM training and macro F_0.5 threshold optimization
│   ├── evaluate.py          # Official competition metric evaluation (Macro F_0.5)
│   ├── generate_output.py   # High-throughput batched inference & Bipartite Conflict Resolution
│   └── token_idf.pkl        # Smooth token inverse document frequency dictionary
├── README.md                # Reproduction guide
└── requirements.txt         # Pinned python dependencies
```

---

## 🚀 End-to-End Reproduction Guide

### Step 1: Preprocess Data Sources
Clean and normalize business names and addresses across all sources:
```bash
python3 src/preprocess.py \
  --input dataset/test/test_source1.tsv \
  --output data/processed/test_source1_clean.tsv

python3 src/preprocess.py \
  --input dataset/test/test_source2.tsv \
  --output data/processed/test_source2_clean.tsv

python3 src/preprocess.py \
  --input dataset/test/test_source3.tsv \
  --output data/processed/test_source3_clean.tsv
```

### Step 2: Country-Partitioned Blocking ($K=20$)
Generate candidate pairs using multi-pass tight blocking with adaptive capacity:
```bash
python3 src/blocking.py \
  --s1 data/processed/test_source1_clean.tsv \
  --s2 data/processed/test_source2_clean.tsv \
  --s3 data/processed/test_source3_clean.tsv \
  --output output/candidate_pairs.tsv \
  --k 20
```

### Step 3: Run Model Inference & Bipartite Conflict Resolution
Score candidate pairs using the trained 32-feature LightGBM model, apply country-calibrated two-tier thresholds, and resolve bipartite conflicts:
```bash
python3 src/generate_output.py \
  --candidates output/candidate_pairs.tsv \
  --s1 data/processed/test_source1_clean.tsv \
  --s2 data/processed/test_source2_clean.tsv \
  --s3 data/processed/test_source3_clean.tsv \
  --model models/matcher_lgbm.pkl \
  --output output/matching_results.tsv
```

### Step 4: Validate Submission Format
Run the official challenge validator to ensure 100% compliance:
```bash
python3 ../../student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
