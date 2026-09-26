"""
features.py
Feature extraction for entity pairs (Source 1 reference vs Candidate).
Optimized for high-throughput pairwise scoring.
"""

from typing import Dict, List, Any, Optional, Set
import os
import math
import pickle
import re

# Load token IDF table for TF-IDF weighted features
_IDF_TABLE = None
_DEFAULT_IDF = 15.61

def _get_idf_table():
    global _IDF_TABLE, _DEFAULT_IDF
    if _IDF_TABLE is None:
        for path in [
            os.path.join(os.path.dirname(__file__), "token_idf.pkl"),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "token_idf.pkl"),
            "models/token_idf.pkl"
        ]:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        _IDF_TABLE = pickle.load(f)
                    _DEFAULT_IDF = _IDF_TABLE.get("__DEFAULT__", 15.61)
                    break
                except Exception:
                    pass
        if _IDF_TABLE is None:
            _IDF_TABLE = {}
    return _IDF_TABLE, _DEFAULT_IDF

# Use C-backed rapidfuzz when available (~48x faster than difflib.SequenceMatcher)
try:
    from rapidfuzz import fuzz as _rf_fuzz
    def _seq_ratio(a: str, b: str) -> float:
        """Full sequence similarity ratio via rapidfuzz (C-backed)."""
        return _rf_fuzz.ratio(a, b) / 100.0
    def _token_sort_ratio(a: str, b: str) -> float:
        """Token sort ratio via rapidfuzz (word-order invariant)."""
        return _rf_fuzz.token_sort_ratio(a, b) / 100.0
    def _token_set_ratio(a: str, b: str) -> float:
        """Token set ratio via rapidfuzz (subset and duplicate tolerant)."""
        return _rf_fuzz.token_set_ratio(a, b) / 100.0
    _USING_RAPIDFUZZ = True
except ImportError:
    import difflib
    def _seq_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()
    def _token_sort_ratio(a: str, b: str) -> float:
        s1 = " ".join(sorted(a.split()))
        s2 = " ".join(sorted(b.split()))
        return difflib.SequenceMatcher(None, s1, s2).ratio()
    def _token_set_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()
    _USING_RAPIDFUZZ = False

# Fast number extractor for addresses
NUMBER_REGEX = re.compile(r"\b\d+\b")

# Postal / PIN code extractor (India: 6 digits, US/France: 5 digits)
POSTAL_REGEX = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b")


def extract_numbers(text: Optional[str]) -> Set[str]:
    """Extracts normalized numeric tokens from text (e.g. house/street numbers)."""
    if not text or not isinstance(text, str):
        return set()
    nums = NUMBER_REGEX.findall(text)
    # Strip leading zeros so '0226' matches '226'
    return {n.lstrip("0") or "0" for n in nums}


def extract_postal_codes(text: Optional[str]) -> Set[str]:
    """Extracts normalized postal/PIN codes from address text."""
    if not text or not isinstance(text, str):
        return set()
    codes = set()
    for m in POSTAL_REGEX.findall(text):
        clean = m[:5] if ("-" in m and len(m) >= 5) else m
        codes.add(clean)
    return codes


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard index between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    u = tokens1 | tokens2
    return len(tokens1 & tokens2) / len(u) if u else 0.0


def extract_pair_features(
    s1_row: Dict[str, Any],
    cand_row: Dict[str, Any],
    cand_rank: int = 0,
    shared_key_count: int = 1
) -> Dict[str, float]:
    """
    Extracts discriminative features for an (S1, Candidate) pair.
    
    Features extracted:
    - Name exact match
    - Name SequenceMatcher ratio
    - Name token Jaccard & containment
    - Name character 3-gram Jaccard
    - Name first token match & prefix length
    - Name length differences & ratios
    - Address presence & SequenceMatcher ratio
    - Address token Jaccard
    - Address number exact match & mismatch penalty
    - Candidate source (S2 vs S3) & ranking metadata
    """
    name1 = str(s1_row.get("business_name_clean", "")).strip()
    name2 = str(cand_row.get("business_name_clean", "")).strip()

    addr1 = str(s1_row.get("business_address_clean", "")).strip()
    addr2 = str(cand_row.get("business_address_clean", "")).strip()
    if addr1 in ("nan", "None", ""): addr1 = ""
    if addr2 in ("nan", "None", ""): addr2 = ""

    cand_id = str(cand_row.get("entity_id", ""))

    # --- 1. Name Features ---
    len1 = len(name1)
    len2 = len(name2)
    max_len = max(len1, len2)
    min_len = min(len1, len2)

    name_exact = 1.0 if name1 and name1 == name2 else 0.0
    name_len_diff = float(abs(len1 - len2))
    name_len_ratio = (min_len / max_len) if max_len > 0 else 0.0

    # Prefix match
    prefix_len = 0
    for c1, c2 in zip(name1, name2):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    name_prefix_ratio = (prefix_len / max_len) if max_len > 0 else 0.0

    # Tokens
    toks1 = name1.split()
    toks2 = name2.split()
    set_toks1 = set(toks1)
    set_toks2 = set(toks2)

    name_tok_jaccard = token_jaccard(set_toks1, set_toks2)
    min_tok_count = min(len(set_toks1), len(set_toks2))
    name_tok_containment = (len(set_toks1 & set_toks2) / min_tok_count) if min_tok_count > 0 else 0.0

    # First token match
    first_tok_match = 1.0 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0.0

    # 3-gram Jaccard
    ng1 = char_ngrams(name1, 3)
    ng2 = char_ngrams(name2, 3)
    name_3gram_jaccard = token_jaccard(ng1, ng2)

    # Sequence similarity (rapidfuzz C-backed when available)
    name_seq_ratio = _seq_ratio(name1, name2)
    name_token_sort_ratio = _token_sort_ratio(name1, name2)
    name_token_set_ratio = _token_set_ratio(name1, name2)

    # TF-IDF Weighted Token Features (downweights ubiquitous generic tokens, upweights distinctive brand stems)
    idf_tab, def_idf = _get_idf_table()
    inter = set_toks1 & set_toks2
    if inter and idf_tab:
        w_inter = sum(idf_tab.get(t, def_idf) for t in inter)
        w_union = sum(idf_tab.get(t, def_idf) for t in (set_toks1 | set_toks2))
        name_tfidf_jaccard = (w_inter / w_union) if w_union > 0 else 0.0

        dot = sum(idf_tab.get(t, def_idf) ** 2 for t in inter)
        n1 = math.sqrt(sum(idf_tab.get(t, def_idf) ** 2 for t in set_toks1))
        n2 = math.sqrt(sum(idf_tab.get(t, def_idf) ** 2 for t in set_toks2))
        name_tfidf_cosine = (dot / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
    else:
        name_tfidf_jaccard = 0.0
        name_tfidf_cosine = 0.0

    # --- 2. Address Features ---
    has_addr1 = 1.0 if addr1 else 0.0
    has_addr2 = 1.0 if addr2 else 0.0
    addr_both = 1.0 if (has_addr1 and has_addr2) else 0.0

    if addr_both:
        addr_seq_ratio = _seq_ratio(addr1, addr2)
        addr_token_sort_ratio = _token_sort_ratio(addr1, addr2)
        addr_token_set_ratio = _token_set_ratio(addr1, addr2)
        addr_toks1 = set(addr1.split())
        addr_toks2 = set(addr2.split())
        addr_tok_jaccard = token_jaccard(addr_toks1, addr_toks2)

        # Address numbers
        nums1 = extract_numbers(addr1)
        nums2 = extract_numbers(addr2)
        both_have_nums = 1.0 if (nums1 and nums2) else 0.0
        num_overlap = 1.0 if (nums1 & nums2) else 0.0
        num_mismatch = 1.0 if (nums1 and nums2 and not (nums1 & nums2)) else 0.0

        # Address postal / PIN codes
        postals1 = extract_postal_codes(addr1)
        postals2 = extract_postal_codes(addr2)
        postal_both = 1.0 if (postals1 and postals2) else 0.0
        postal_match = 1.0 if (postals1 & postals2) else 0.0
        postal_mismatch = 1.0 if (postals1 and postals2 and not (postals1 & postals2)) else 0.0
    else:
        addr_seq_ratio = 0.0
        addr_token_sort_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_tok_jaccard = 0.0
        both_have_nums = 0.0
        num_overlap = 0.0
        num_mismatch = 0.0
        postal_both = 0.0
        postal_match = 0.0
        postal_mismatch = 0.0

    # --- 3. Country & Metadata Features ---
    c1 = str(s1_row.get("country", "")).strip()
    c2 = str(cand_row.get("country", "")).strip()
    country_match = 1.0 if (c1 and c2 and c1 == c2) else (0.5 if (not c1 or not c2) else 0.0)
    country_is_us = 1.0 if c1 == "US" else 0.0
    country_is_india = 1.0 if c1 == "India" else 0.0
    country_is_france = 1.0 if c1 == "France" else 0.0

    # Candidate source (S2 vs S3) & ranking metadata
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return {
        "name_exact": name_exact,
        "name_seq_ratio": name_seq_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_tok_jaccard": name_tok_jaccard,
        "name_tok_containment": name_tok_containment,
        "name_first_tok_match": first_tok_match,
        "name_3gram_jaccard": name_3gram_jaccard,
        "name_tfidf_jaccard": name_tfidf_jaccard,
        "name_tfidf_cosine": name_tfidf_cosine,
        "name_prefix_ratio": name_prefix_ratio,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_both": addr_both,
        "addr_seq_ratio": addr_seq_ratio,
        "addr_token_sort_ratio": addr_token_sort_ratio,
        "addr_token_set_ratio": addr_token_set_ratio,
        "addr_tok_jaccard": addr_tok_jaccard,
        "addr_both_have_nums": both_have_nums,
        "addr_num_overlap": num_overlap,
        "addr_num_mismatch": num_mismatch,
        "addr_postal_both": postal_both,
        "addr_postal_match": postal_match,
        "addr_postal_mismatch": postal_mismatch,
        "country_match": country_match,
        "country_is_us": country_is_us,
        "country_is_india": country_is_india,
        "country_is_france": country_is_france,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "cand_rank": float(cand_rank),
        "shared_key_count": float(shared_key_count),
    }


FEATURE_NAMES = list(extract_pair_features(
    {"business_name_clean": "test", "business_address_clean": "123 main st", "entity_id": "S1-1", "country": "US"},
    {"business_name_clean": "test corp", "business_address_clean": "123 main st", "entity_id": "S2-1", "country": "US"}
).keys())


if __name__ == "__main__":
    s1 = {"business_name_clean": "starbucks coffee", "business_address_clean": "100 pine st seattle wa 98101", "entity_id": "S1-0001", "country": "US"}
    s2 = {"business_name_clean": "starbucks coffee co", "business_address_clean": "100 pine street suite 200 98101", "entity_id": "S2-0005", "country": "US"}
    feats = extract_pair_features(s1, s2, cand_rank=0, shared_key_count=3)
    print("Extracted Features:")
    for k, v in feats.items():
        print(f"  {k:25s}: {v:.4f}")
    assert len(feats) == len(FEATURE_NAMES)
    print(f"\nTotal features: {len(FEATURE_NAMES)}")
    print("features.py test PASSED!")
