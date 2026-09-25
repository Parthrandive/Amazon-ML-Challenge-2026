"""
preprocess.py
Module for entity resolution preprocessing: name normalization, transliteration,
address standardization, and residual script checking.
"""

import re
from typing import Optional
import pandas as pd
from transliterate import detect_script, transliterate_name

import unicodedata

def strip_accents(text: str) -> str:
    """Converts accented Latin characters to plain ASCII base characters (e.g. é -> e, à -> a, ç -> c)."""
    if not text:
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

# ---------------------------------------------------------
# Regex patterns for English and French legal suffixes
# ---------------------------------------------------------
LEGAL_SUFFIXES_EN = [
    # French corporate forms (spaced, dotted, and concatenated)
    r"societe\s+anonyme",
    r"societe\s+par\s+actions\s+simplifiee",
    r"societe\s+a\s+responsabilite\s+limitee",
    r"s\s*a\s*r\s*l",
    r"s\s*a\s*s",
    r"s\s*c\s*i",
    r"e\s*u\s*r\s*l",
    r"s\s*n\s*c",
    r"groupe",
    r"s\s*a",
    # English forms
    r"private\s+limited",
    r"pvt\s+limited",
    r"pvt\s+ltd",
    r"limited\s+liability\s+company",
    r"limited\s+liability\s+partnership",
    r"incorporated",
    r"corporation",
    r"private",
    r"limited",
    r"company",
    r"corp",
    r"ltd",
    r"llc",
    r"llp",
    r"inc",
    r"co",
    r"pc",
]
EN_LEGAL_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(LEGAL_SUFFIXES_EN) + r")\b\s*$", re.IGNORECASE
)

# ---------------------------------------------------------
# Regex patterns for transliterated Indic suffix variants
# ---------------------------------------------------------
TRANSLIT_SUFFIXES = [
    r"praiveta\s+limiteda",
    r"praiveta\s+limited",
    r"praivet\s+limiteda",
    r"praivet\s+limited",
    r"bhiraivedh\s+limidhedh",
    r"bhiraivedh\s+limidhed",
    r"praivet\s+limided",
    r"praiveta\s+limideda",
    r"praivet\s+limite[dt]",
    r"praive[tT]\s+limi[tT][eèé]?[dD]",
    r"limiteda",
    r"limidhedh",
    r"limidhed",
    r"limiteḍa",
    r"limite[dt]a?",
    r"elaelapi",
    r"pra\s+li",
    r"praiveta",
    r"praivet",
    r"kampani",
    r"kamapani",
    r"korporesana",
    r"korporeshana",
]
TRANSLIT_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(TRANSLIT_SUFFIXES) + r")\b\s*$", re.IGNORECASE
)

# ---------------------------------------------------------
# Non-Latin Unicode pattern to detect residual script
# ---------------------------------------------------------
NON_LATIN_RE = re.compile(r"[\u0600-\u0D7F\u200B-\u200D\u3000-\uD7AF\u4E00-\u9FFF]")

# ---------------------------------------------------------
# Address abbreviation standardizers
# ---------------------------------------------------------
ADDR_PUNCT_RE = re.compile(r"[^\w\s,\u0900-\u0D7F]")
ST_RE = re.compile(r"\bst\b\.?", re.IGNORECASE)
RD_RE = re.compile(r"\brd\b\.?", re.IGNORECASE)
AVE_RE = re.compile(r"\bave\b\.?", re.IGNORECASE)
DR_RE = re.compile(r"\bdr\b\.?", re.IGNORECASE)
APT_RE = re.compile(r"\bapt\b\.?", re.IGNORECASE)
BVD_RE = re.compile(r"\b(bd|bld|blvd)\b\.?", re.IGNORECASE)
RUE_RE = re.compile(r"\br\b\.?", re.IGNORECASE)
ALL_RE = re.compile(r"\ball\b\.?", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
NAME_PUNCT_RE = re.compile(r"[^\w\s\u0900-\u0D7F]")


def normalize_name(name: Optional[str]) -> str:
    """
    lowercase, strip accents/diacritics, strip punctuation, collapse whitespace,
    strip legal suffixes (English & French forms).
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    s = strip_accents(name.lower())
    s = re.sub(r"&", " and ", s)
    s = NAME_PUNCT_RE.sub(" ", s).replace("_", " ")
    s = WHITESPACE_RE.sub(" ", s).strip()
    while True:
        m = EN_LEGAL_SUFFIX_RE.search(s)
        if m:
            s = s[:m.start()].strip()
        else:
            break
    return s


def normalize_translit_suffix(text: Optional[str]) -> str:
    """
    Strip transliterated suffix variants.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    s = text.strip()
    while True:
        m = TRANSLIT_SUFFIX_RE.search(s)
        if m:
            s = s[:m.start()].strip()
        else:
            break
    return s


def normalize_address(address: Optional[str]) -> str:
    """
    lowercase, strip accents/diacritics, strip punctuation except commas,
    standardize abbreviations (St/Rd/Ave/Dr/Apt/Bd/Rue/Allee), collapse whitespace.
    """
    if not isinstance(address, str) or not address.strip():
        return ""
    s = strip_accents(address.lower())
    s = ADDR_PUNCT_RE.sub(" ", s).replace("_", " ")
    s = ST_RE.sub("street", s)
    s = RD_RE.sub("road", s)
    s = AVE_RE.sub("avenue", s)
    s = DR_RE.sub("drive", s)
    s = APT_RE.sub("apartment", s)
    s = BVD_RE.sub("boulevard", s)
    s = RUE_RE.sub("rue", s)
    s = ALL_RE.sub("allee", s)
    return WHITESPACE_RE.sub(" ", s).strip()


def has_residual_script(text: Optional[str]) -> bool:
    """
    Check if a text still contains non-Latin Unicode characters after conversion.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(NON_LATIN_RE.search(text))


def preprocess_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe:
    - Retain raw columns: entity_id, business_name, business_address, country
    - Add:
      - name_script
      - business_name_translit
      - translit_has_residual_script
      - business_name_clean
      - business_address_clean
    """
    result = df.copy()

    # --- 1. Process Names efficiently via unique values ---
    raw_names = result["business_name"].fillna("").astype(str)
    unique_names = pd.Series(raw_names.unique())

    # Script detection
    script_map = {n: detect_script(n) for n in unique_names}

    # Transliteration
    translit_map = {}
    for n in unique_names:
        sc = script_map[n]
        translit_map[n] = transliterate_name(n, sc)

    # Check residual script on transliterated names
    residual_map = {n: has_residual_script(translit_map[n]) for n in unique_names}

    # Name cleaning
    clean_name_map = {}
    for n in unique_names:
        trans = translit_map[n]
        norm = normalize_name(trans)
        clean = normalize_translit_suffix(norm)
        clean_name_map[n] = clean

    # Map back to dataframe
    result["name_script"] = raw_names.map(script_map)
    result["business_name_translit"] = raw_names.map(translit_map)
    result["translit_has_residual_script"] = raw_names.map(residual_map)
    result["business_name_clean"] = raw_names.map(clean_name_map)

    # --- 2. Process Addresses efficiently via unique values ---
    raw_addrs = result["business_address"].fillna("").astype(str)
    unique_addrs = pd.Series(raw_addrs.unique())

    # Address normalization
    clean_addr_map = {a: normalize_address(a) for a in unique_addrs}
    result["business_address_clean"] = raw_addrs.map(clean_addr_map)

    return result
