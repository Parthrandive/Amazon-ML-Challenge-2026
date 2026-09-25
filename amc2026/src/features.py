"""
features.py
Feature extraction for entity pairs (Source 1 reference vs Candidate).
Optimized for high-throughput pairwise scoring.
"""

from typing import Dict, List, Any, Optional, Set
import difflib
import re

# Fast number extractor for addresses
NUMBER_REGEX = re.compile(r"\b\d+\b")


def extract_numbers(text: Optional[str]) -> Set[str]:
    """Extracts normalized numeric tokens from text (e.g. house/street numbers)."""
    if not text or not isinstance(text, str):
        return set()
    nums = NUMBER_REGEX.findall(text)
    # Strip leading zeros so '0226' matches '226'
    return {n.lstrip("0") or "0" for n in nums}


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

    # Sequence similarity (quick_ratio is very fast)
    sm = difflib.SequenceMatcher(None, name1, name2)
    name_seq_ratio = sm.ratio()

    # --- 2. Address Features ---
    has_addr1 = 1.0 if addr1 else 0.0
    has_addr2 = 1.0 if addr2 else 0.0
    addr_both = 1.0 if (has_addr1 and has_addr2) else 0.0

    if addr_both:
        addr_seq_ratio = difflib.SequenceMatcher(None, addr1, addr2).quick_ratio()
        addr_toks1 = set(addr1.split())
        addr_toks2 = set(addr2.split())
        addr_tok_jaccard = token_jaccard(addr_toks1, addr_toks2)

        # Address numbers
        nums1 = extract_numbers(addr1)
        nums2 = extract_numbers(addr2)
        both_have_nums = 1.0 if (nums1 and nums2) else 0.0
        num_overlap = 1.0 if (nums1 & nums2) else 0.0
        num_mismatch = 1.0 if (nums1 and nums2 and not (nums1 & nums2)) else 0.0
    else:
        addr_seq_ratio = 0.0
        addr_tok_jaccard = 0.0
        both_have_nums = 0.0
        num_overlap = 0.0
        num_mismatch = 0.0

    # --- 3. Metadata & Source Features ---
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return {
        "name_exact": name_exact,
        "name_seq_ratio": name_seq_ratio,
        "name_tok_jaccard": name_tok_jaccard,
        "name_tok_containment": name_tok_containment,
        "name_first_tok_match": first_tok_match,
        "name_3gram_jaccard": name_3gram_jaccard,
        "name_prefix_ratio": name_prefix_ratio,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_both": addr_both,
        "addr_seq_ratio": addr_seq_ratio,
        "addr_tok_jaccard": addr_tok_jaccard,
        "addr_both_have_nums": both_have_nums,
        "addr_num_overlap": num_overlap,
        "addr_num_mismatch": num_mismatch,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "cand_rank": float(cand_rank),
        "shared_key_count": float(shared_key_count),
    }


FEATURE_NAMES = list(extract_pair_features(
    {"business_name_clean": "test", "business_address_clean": "123 main st", "entity_id": "S1-1"},
    {"business_name_clean": "test corp", "business_address_clean": "123 main st", "entity_id": "S2-1"}
).keys())


if __name__ == "__main__":
    s1 = {"business_name_clean": "starbucks coffee", "business_address_clean": "100 pine st seattle wa 98101", "entity_id": "S1-0001"}
    s2 = {"business_name_clean": "starbucks coffee co", "business_address_clean": "100 pine street suite 200", "entity_id": "S2-0005"}
    feats = extract_pair_features(s1, s2, cand_rank=0, shared_key_count=3)
    print("Extracted Features:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")
    assert len(feats) == len(FEATURE_NAMES)
    print("\nfeatures.py test PASSED!")
