"""
transliterate.py
Module for script detection and Indic script transliteration into Latin phonetic form.
"""

from typing import Optional, Tuple
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Unicode ranges and corresponding sanscript scheme
SCRIPT_RANGES = [
    (0x0900, 0x097F, "Devanagari", sanscript.DEVANAGARI),
    (0x0980, 0x09FF, "Bengali", sanscript.BENGALI),
    (0x0A00, 0x0A7F, "Gurmukhi", sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, "Gujarati", sanscript.GUJARATI),
    (0x0B00, 0x0B7F, "Oriya", sanscript.ORIYA),
    (0x0B80, 0x0BFF, "Tamil", sanscript.TAMIL),
    (0x0C00, 0x0C7F, "Telugu", sanscript.TELUGU),
    (0x0C80, 0x0CFF, "Kannada", sanscript.KANNADA),
    (0x0D00, 0x0D7F, "Malayalam", sanscript.MALAYALAM),
]

SCRIPT_MAP = {name.lower(): scheme for _, _, name, scheme in SCRIPT_RANGES}

# Pre-pass: strip zero-width characters and invisible formatting marks
PRE_STRIP_TRANS = str.maketrans("", "", "\u200b\u200c\u200d\ufeff")

# Fallback character map for loanword vowels, special chillus, and missing glyphs in itrans
FALLBACK_MAP = {
    "\u0949": "o",   # DEVANAGARI VOWEL SIGN CANDRA O (e.g. प्रॉपर्टीज -> propartija)
    "\u0911": "o",   # DEVANAGARI LETTER CANDRA O (e.g. ऑल -> ol)
    "\u0945": "e",   # DEVANAGARI VOWEL SIGN CANDRA E
    "\u090d": "e",   # DEVANAGARI LETTER CANDRA E
    "\u0972": "c",   # DEVANAGARI LETTER CANDRA A
    "\u0ba9": "n",   # TAMIL LETTER NNNA (e.g. பிசினஸ் -> bhijhinas)
    "\u09bc": "",    # BENGALI SIGN NUKTA
    "\u0a3c": "",    # GURMUKHI SIGN NUKTA
    "\u0b3c": "",    # ORIYA SIGN NUKTA
    "\u0b71": "w",   # ORIYA LETTER WA (e.g. ସଫ୍ଟୱେୟାର୍ -> saphtaweyar)
    "\u0d7c": "r",   # MALAYALAM LETTER CHILLU RR
    "\u0d7b": "n",   # MALAYALAM LETTER CHILLU N
    "\u0d7d": "l",   # MALAYALAM LETTER CHILLU L
    "\u0d7a": "n",   # MALAYALAM LETTER CHILLU NN
    "\u0d7e": "l",   # MALAYALAM LETTER CHILLU LL
    "\u0d57": "au",  # MALAYALAM AU LENGTH MARK
}


def detect_script(text: Optional[str]) -> str:
    """
    Unicode-range check for Indic scripts (Devanagari, Bengali, Gurmukhi,
    Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam), else latin.
    """
    if not isinstance(text, str) or not text.strip():
        return "latin"

    if text.isascii():
        return "latin"

    for ch in text:
        cp = ord(ch)
        for start, end, name, _ in SCRIPT_RANGES:
            if start <= cp <= end:
                return name

    return "latin"


def transliterate_name(text: Optional[str], script: Optional[str] = None) -> str:
    """
    Use indic-transliteration to convert non-Latin scripts to Latin phonetic form (ITRANS).
    Applies:
    1. Pre-pass: strips zero-width non-joiners/joiners and invisible marks.
    2. Transliteration using sanscript.ITRANS for the detected script.
    3. Targeted post-fallback: maps any remaining loanword vowels/chillus to Latin.
    Returns unchanged if already Latin.
    """
    if not isinstance(text, str) or not text.strip() or text.isascii():
        return "" if text is None else str(text)

    # Pre-pass: strip zero-width characters
    clean_text = text.translate(PRE_STRIP_TRANS)

    if script is None:
        script = detect_script(clean_text)

    script_key = script.lower()
    if script_key == "latin" or script_key not in SCRIPT_MAP:
        return text

    scheme = SCRIPT_MAP[script_key]
    try:
        t = transliterate(clean_text, scheme, sanscript.ITRANS)
    except Exception:
        t = clean_text

    # Targeted post-fallback on any remaining non-ASCII character
    if not t.isascii():
        t = "".join(FALLBACK_MAP.get(ch, ch) for ch in t)

    return t
