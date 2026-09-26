"""
blocking.py
Tightened Composite Multi-Pass Disjunctive Blocking for Amazon ML Challenge 2026.
Designed for 95%+ effective recall with tight block sizes to prevent block flooding.
"""

import os
import sys
import re
from typing import Set, List, Tuple, Optional

# High frequency corporate & industry descriptors (suppressed from single-token keys)
COMMON_WORDS = {
    "services", "service", "solutions", "solution", "technologies", "technology",
    "enterprises", "enterprise", "group", "holdings", "industries", "industry",
    "consulting", "consultants", "trading", "associates", "international",
    "management", "global", "systems", "corporation", "company", "india", "tech",
    "care", "health", "dental", "medical", "clinic", "center", "centre", "hospital",
    "store", "shop", "mart", "bazaar", "builders", "developers", "properties",
    "automotive", "motors", "auto", "logistics", "transport", "food", "restaurant",
    "cafe", "hotel", "pharma", "pharmacy", "products", "commercial", "agency",
    "venture", "ventures", "corp", "incorporated"
}

import unicodedata

# Stopwords & titles to strip from start of business name
LEADING_TITLES = {
    "the", "a", "an", "m", "s", "dr", "smt", "shri", "sri", "mr", "mrs",
    "inc", "llc", "pvt", "ltd", "co", "m/s", "ínc",
    "sci", "sarl", "sas", "sa", "eurl", "snc", "le", "la", "les", "l", "d"
}

# Generic street/location words to skip when extracting distinctive street tokens
IGNORE_STREET_WORDS = {
    "street", "road", "avenue", "drive", "lane", "apartment", "unit", "floor",
    "india", "us", "delhi", "maharashtra", "karnataka", "texas", "california",
    "ohio", "nc", "tn", "il", "ca", "ny", "po", "box", "st", "rd", "ave",
    "dr", "apt", "fl", "no", "near", "opp", "behind", "c", "o",
    "france", "paris", "rue", "boulevard", "allee", "chemin", "place", "nord", "cedex", "gironde"
}


def clean_accents(text: str) -> str:
    """Replaces accented characters with base Latin characters."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def get_name_tokens(name: Optional[str]) -> List[str]:
    """Extracts clean alphanumeric tokens from business name, stripping leading titles."""
    if not isinstance(name, str) or not name.strip():
        return []
    name_clean = clean_accents(name.lower())
    tokens = re.findall(r"[a-z0-9]+", name_clean)
    while tokens and tokens[0] in LEADING_TITLES:
        tokens = tokens[1:]
    return tokens


def get_address_tokens(addr: Optional[str]) -> Tuple[List[str], List[str]]:
    """
    Extracts normalized address numbers (leading zeros stripped) and distinctive street words.
    """
    if not isinstance(addr, str) or not addr.strip() or addr == "nan":
        return [], []

    addr_clean = clean_accents(addr.lower())
    raw_nums = re.findall(r"\b[a-z]{0,2}\d{1,5}[a-z]?\b", addr_clean)
    clean_nums = []
    for n in raw_nums:
        m = re.match(r"^([a-z]*)0*(\d+)([a-z]*)$", n)
        if m:
            clean_nums.append(f"{m.group(1)}{m.group(2)}{m.group(3)}")
        else:
            clean_nums.append(n)

    all_words = re.findall(r"[a-z]+", addr_clean)
    street_words = [w for w in all_words if w not in IGNORE_STREET_WORDS and len(w) >= 3]

    return clean_nums, street_words


def get_tight_blocking_keys(
    country: str,
    name: Optional[str],
    addr: Optional[str]
) -> List[str]:
    """
    Generates tightened composite blocking keys:
    1. Pass 1: Name First Significant Token (4-char prefix & phonetic initial)
    2. Pass 2: Address Number + Distinctive Street Word (always composite)
    3. Pass 3: 2-Token Shingles (t0_t1 pair, t0_t2 pair) & non-generic secondary token
    """
    tokens = get_name_tokens(name)
    nums, street_words = get_address_tokens(addr)
    keys = set()

    # Pass 1: Name primary token
    if tokens:
        t0 = tokens[0]
        t0_ph = "k" + t0[1:] if t0.startswith("c") else t0
        p4 = t0[:4] if len(t0) >= 4 else t0
        p3_ph = t0_ph[:3] if len(t0_ph) >= 3 else t0_ph
        keys.add(f"{country}_n1_{p4}")
        keys.add(f"{country}_n1_{p3_ph}")

        # Pass 3: 2-token shingles
        if len(tokens) >= 2:
            pair = "_".join(sorted([tokens[0][:3], tokens[1][:3]]))
            keys.add(f"{country}_np_{pair}")
            t1 = tokens[1]
            if t1 not in COMMON_WORDS and len(t1) >= 4:
                keys.add(f"{country}_n2_{t1[:4]}")
        if len(tokens) >= 3:
            pair2 = "_".join(sorted([tokens[0][:3], tokens[2][:3]]))
            keys.add(f"{country}_np_{pair2}")

    # Pass 2: Address composite keys (always number + street word)
    if nums and street_words:
        for num in nums[:2]:
            for sw in street_words[:3]:
                keys.add(f"{country}_as_{num}_{sw[:4]}")

    return list(keys)


def get_key_max_size(key: str) -> int:
    """
    Adaptive block depth: Specific composite keys have high/uncapped capacity,
    broad single-token keys are strictly capped to prevent flooding.
    """
    if "_as_" in key:  # Address composite: number + street word (exact building/house)
        return 10000   # essentially uncapped
    elif "_np_" in key:  # 2-token shingles
        return 5000    # high capacity
    else:  # Broad single-token prefix keys (n1_, n2_, fallback_)
        return 500     # strictly capped to prevent flooding


def get_fallback_blocking_keys(name: Optional[str]) -> List[str]:
    """Country-agnostic, name-only keys for records whose country is unknown.

    These keys are deliberately narrower than normal blocking: they are a safety net
    for an uncertain country, not a second all-pairs pass.  The caller still caps each
    bucket and applies the same relevance ranking used by country partitions.
    """
    tokens = get_name_tokens(name)
    if not tokens:
        return []
    primary = tokens[0]
    phonetic = "k" + primary[1:] if primary.startswith("c") else primary
    keys = {
        f"fallback_n1_{primary[:4]}",
        f"fallback_n1_{phonetic[:3]}",
    }
    if len(tokens) >= 2:
        keys.add(f"fallback_np_{'_'.join(sorted((primary[:3], tokens[1][:3])))}")
    if len(tokens) >= 3:
        keys.add(f"fallback_np_{'_'.join(sorted((primary[:3], tokens[2][:3])))}")
    return list(keys)


def fast_combined_similarity(
    s1_name: str,
    s2_name: str,
    s1_toks_set: Set[str],
    s1_nums_set: Set[str],
    s2_nums_set: Set[str]
) -> float:
    """
    Lightweight, C-level string similarity pre-ranker.
    Combines character 3-gram Jaccard, token Jaccard, and address number match bonus/penalty.
    """
    if s1_name and s1_name == s2_name:
        name_sim = 1.0
    else:
        # Character 3-gram Jaccard
        len1 = len(s1_name)
        len2 = len(s2_name)
        g1 = {s1_name[i:i + 3] for i in range(len1 - 2)} if len1 >= 3 else {s1_name}
        g2 = {s2_name[i:i + 3] for i in range(len2 - 2)} if len2 >= 3 else {s2_name}
        u = g1 | g2
        j_char = len(g1 & g2) / len(u) if u else 0.0

        # Word Token Jaccard
        toks2 = set(s2_name.split())
        u_tok = s1_toks_set | toks2
        j_tok = len(s1_toks_set & toks2) / len(u_tok) if u_tok else 0.0

        name_sim = 0.6 * j_char + 0.4 * j_tok

    # Address Number Match Bonus / Mismatch Penalty
    addr_bonus = 0.0
    if s1_nums_set and s2_nums_set:
        if s1_nums_set & s2_nums_set:
            addr_bonus = 0.35  # Confirmatory building/house number match
        else:
            addr_bonus = -0.15 # Conflicting numbers on same street

    return name_sim + addr_bonus
