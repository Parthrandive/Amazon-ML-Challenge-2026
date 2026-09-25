"""
postal.py
Country-specific postal code extraction and normalization:
- US: 5-digit base ZIP code (stripping +4 extension)
- India: 6-digit PIN code (100000 - 999999)
- France: 5-digit postal code (01000 - 98999)
"""

import re
from typing import Optional

US_POSTAL_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
INDIA_POSTAL_RE = re.compile(r"\b([1-9]\d{5})\b")
FRANCE_POSTAL_RE = re.compile(r"\b(0[1-9]\d{3}|[1-8]\d{4}|9[0-8]\d{3})\b")

POSTAL_MISSING = "POSTAL_MISSING"


def is_valid_postal(code: Optional[str]) -> bool:
    """Returns True if code is non-empty and not missing sentinel."""
    return bool(code and str(code).strip() not in (POSTAL_MISSING, "nan", "None", ""))


def extract_postal_code(address: Optional[str], country: Optional[str]) -> str:
    """
    Extracts and normalizes country-specific postal code from address string.
    Returns explicit POSTAL_MISSING when not found or invalid.
    """
    if not isinstance(address, str) or not address.strip() or address == "nan":
        return POSTAL_MISSING

    country_str = str(country).strip().lower() if country else ""

    if country_str in ("us", "usa", "united states"):
        matches = US_POSTAL_RE.findall(address)
        if matches:
            return matches[-1]
    elif country_str in ("india", "in", "ind"):
        matches = INDIA_POSTAL_RE.findall(address)
        if matches:
            return matches[-1]
    elif country_str in ("france", "fr", "fra"):
        matches = FRANCE_POSTAL_RE.findall(address)
        if matches:
            return matches[-1]
    else:
        # Dynamic fallback
        in_match = INDIA_POSTAL_RE.findall(address)
        if in_match:
            return in_match[-1]
        us_fr_match = US_POSTAL_RE.findall(address)
        if us_fr_match:
            return us_fr_match[-1]

    return POSTAL_MISSING
