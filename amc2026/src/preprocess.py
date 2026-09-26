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

# Country extraction is intentionally independent of the supplied `country` column.
# Source labels are useful for evaluation, but are not available/reliable enough to be a
# blocking key in production.  Keep the aliases here rather than an inspected sample so
# that a newly encountered state spelling does not silently become `unknown`.
US_STATE_ALIASES = {
    "alabama": "US", "alaska": "US", "arizona": "US", "arkansas": "US",
    "california": "US", "calif": "US", "cal": "US", "colorado": "US",
    "connecticut": "US", "conn": "US", "delaware": "US", "florida": "US",
    "fla": "US", "georgia": "US", "hawaii": "US", "idaho": "US",
    "illinois": "US", "ill": "US", "indiana": "US", "iowa": "US",
    "kansas": "US", "kentucky": "US", "louisiana": "US", "maine": "US",
    "maryland": "US", "massachusetts": "US", "mass": "US", "michigan": "US",
    "minnesota": "US", "minn": "US", "mississippi": "US", "missouri": "US",
    "montana": "US", "nebraska": "US", "nevada": "US", "new hampshire": "US",
    "new jersey": "US", "new mexico": "US", "new york": "US", "north carolina": "US",
    "north dakota": "US", "ohio": "US", "oklahoma": "US", "oregon": "US",
    "pennsylvania": "US", "penn": "US", "rhode island": "US", "south carolina": "US",
    "south dakota": "US", "tennessee": "US", "texas": "US", "utah": "US",
    "vermont": "US", "virginia": "US", "washington": "US", "west virginia": "US",
    "wisconsin": "US", "wyoming": "US", "district of columbia": "US",
    "washington dc": "US", "puerto rico": "US", "guam": "US",
    "american samoa": "US", "northern mariana islands": "US",
    "us virgin islands": "US", "virgin islands": "US",
}

# Official/common native-script forms for all 28 states and 8 union territories.
# Several states use more than one script in real address data, so variants are included.
INDIA_NATIVE_STATE_ALIASES = (
    "ఆంధ్ర ప్రదేశ్", "ఆంధ్రప్రదేశ్", "आंध्र प्रदेश", "अरुणाचल प्रदेश", "অসম", "असम", "বিহার", "बिहार", "छत्तीसगढ़", "छत्तीसगढ",
    "गोवा", "ગુજરાત", "हरियाणा", "ਹਰਿਆਣਾ", "हिमाचल प्रदेश", "झारखंड", "ঝাড়খণ্ড", "ঝারখণ্ড", "झारखण्ड", "ಕರ್ನಾಟಕ",
    "കേരളം", "मध्य प्रदेश", "महाराष्ट्र", "মণিপুর", "मणिपुर", "मेघालय", "মেঘালয়",
    "মিজোরাম", "मिजोरम", "नागालैंड", "নাগাল্যান্ড", "ꯃꯅꯤꯄꯨꯔ", "ଓଡ଼ିଶା", "ଓଡିଶା", "ओडिशा",
    "ਪੰਜਾਬ", "ਪੰਜਾਬੀ", "राजस्थान", "सिक्किम", "தமிழ்நாடு", "தமிழ் நாடு", "తెలంగాణ",
    "త్రిపుర", "ত্রিপুরা", "उत्तर प्रदेश", "उत्तराखंड", "पश्चिम बंगाल", "পশ্চিমবঙ্গ",
    "আন্দামান ও নিকোবর", "চণ্ডীগড়", "चंडीगढ़", "ਚੰਡੀਗੜ੍ਹ", "दादरा और नगर हवेली", "દાદરા અને નગર હવેલી",
    "दमन और दीव", "દમણ અને દીવ", "दिल्ली", "जम्मू और कश्मीर", "जम्मू कश्मीर",
    "लद्दाख", "ലക്ഷദ്വീപ്", "पुदुचेरी", "புதுச்சேரி",
)
INDIA_LATIN_STATE_ALIASES = (
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "uttaranchal", "west bengal", "andaman and nicobar", "chandigarh", "dadra and nagar haveli",
    "daman and diu", "delhi", "new delhi", "nct of delhi", "jammu and kashmir", "ladakh",
    "lakshadweep", "puducherry", "pondicherry",
)

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA", "ID", "IL",
    "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND",
    "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY", "DC", "PR", "GU", "AS", "MP", "VI",
}
INDIA_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH",
    "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

FRENCH_ADDRESS_RE = re.compile(
    r"\b(?:rue|allee|chemin|impasse|quai|cedex|france|"
    r"paris|lyon|marseille|toulouse|bordeaux|lille|nantes|strasbourg|nice|rennes|montpellier|"
    r"gironde|ile de france|hauts de seine|seine saint denis|val de marne)\b",
    re.IGNORECASE,
)
FRENCH_NAME_RE = re.compile(r"\b(?:sarl|sas|sasu|sci|eurl|snc|societe|societe anonyme)\b", re.IGNORECASE)

# Compile the large state dictionaries once.  Per-alias searches are far too expensive
# when preprocessing multi-million-row source files.
INDIA_NATIVE_STATE_RE = re.compile("|".join(re.escape(alias) for alias in INDIA_NATIVE_STATE_ALIASES))
INDIA_LATIN_STATE_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(alias) for alias in INDIA_LATIN_STATE_ALIASES) + r")(?![a-z0-9])",
    re.IGNORECASE,
)
US_STATE_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(alias) for alias in US_STATE_ALIASES) + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def extract_country(address: Optional[str], business_name: Optional[str] = None) -> str:
    """Infer India, US, France, or ``unknown`` from address/name evidence only.

    Strong geographic tokens win over postal-shape fallbacks.  France is checked before
    the five-digit ZIP fallback because French postcodes also have five digits.
    """
    address_text = "" if not isinstance(address, str) else address
    name_text = "" if not isinstance(business_name, str) else business_name
    combined = f"{address_text} {name_text}".casefold()

    if INDIA_NATIVE_STATE_RE.search(combined):
        return "India"
    latin_text = strip_accents(combined)
    if INDIA_LATIN_STATE_RE.search(latin_text):
        return "India"
    if US_STATE_RE.search(latin_text):
        return "US"

    # Two-letter state codes only count in an address-style segment; this avoids treating
    # ordinary words such as "in" as Indiana.
    code_tokens = re.findall(r"(?:^|[,\s])([A-Za-z]{2})(?=\s*(?:,|\d{5}(?:-\d{4})?|$))", address_text)
    for code in code_tokens:
        normalized_code = code.upper()
        india_code = normalized_code in INDIA_STATE_CODES
        us_code = normalized_code in US_STATE_CODES
        if india_code and us_code:
            # AP/AS/IN/MP are valid in both systems.  The nearby postal shape is the
            # only safe tie-breaker; otherwise leave the record for later evidence.
            if re.search(r"(?<!\d)\d{6}(?!\d)", address_text):
                return "India"
            if re.search(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)", address_text):
                return "US"
        elif india_code:
            return "India"
        elif us_code:
            return "US"

    normalized_address = strip_accents(address_text.casefold())
    normalized_name = strip_accents(name_text.casefold())
    if FRENCH_ADDRESS_RE.search(normalized_address) or FRENCH_NAME_RE.search(normalized_name):
        return "France"

    # Tertiary signal only: use it after all country-specific evidence above.
    if re.search(r"(?<!\d)\d{6}(?!\d)", address_text):
        return "India"
    if re.search(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)", address_text):
        return "US"
    return "unknown"


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
      - country_extracted (address/name-derived; never copied from the supplied label)
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

    # Country is derived from supplied label when present, or extracted from address/name
    if "country" in result.columns:
        clean_country = result["country"].fillna("").astype(str).str.strip()
        has_label = clean_country.isin(["India", "US", "France"])
        extracted = clean_country.copy()
        unresolved = ~has_label
        if unresolved.any():
            address_country_map = {address: extract_country(address) for address in unique_addrs}
            extracted_unresolved = raw_addrs[unresolved].map(address_country_map)
            extracted.loc[unresolved] = extracted_unresolved
            unknown_names = raw_names[extracted == "unknown"].unique()
            name_country_map = {name: extract_country("", name) for name in unknown_names}
            still_unresolved = extracted == "unknown"
            extracted.loc[still_unresolved] = raw_names.loc[still_unresolved].map(name_country_map)
    else:
        address_country_map = {address: extract_country(address) for address in unique_addrs}
        extracted = raw_addrs.map(address_country_map)
        unknown_names = raw_names[extracted == "unknown"].unique()
        name_country_map = {name: extract_country("", name) for name in unknown_names}
        unresolved = extracted == "unknown"
        extracted.loc[unresolved] = raw_names.loc[unresolved].map(name_country_map)
    result["country_extracted"] = extracted

    # Extract postal codes after country resolution so country-specific formats
    # are interpreted consistently even when the input country label is absent.
    address_country_pairs = list(zip(result["business_address_clean"], result["country_extracted"].fillna("")))
    postal_map = {pair: extract_postal_code(pair[0], pair[1]) for pair in set(address_country_pairs)}
    result["postal_code"] = [postal_map[pair] for pair in address_country_pairs]

    return result
