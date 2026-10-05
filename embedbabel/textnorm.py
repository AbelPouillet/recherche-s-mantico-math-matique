"""Normalisation Unicode, graphèmes et segmentation IPA (sans dépendance externe)."""
from __future__ import annotations

import unicodedata

TIE_BARS = {"\u035c", "\u0361"}
# Marques prosodiques qui ne sont pas des phonèmes
IPA_IGNORED = set("ˈˌ.|‖ \t/[]()")
SYLLABLE_SEP = "."


def normalize(text: str, form: str = "NFC") -> str:
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"forme de normalisation inconnue : {form}")
    return unicodedata.normalize(form, text)


def graphemes(text: str) -> list[str]:
    """Groupes « caractère de base + signes combinants » (approximation des
    clusters de graphèmes étendus ; suffisant pour latin, grec, arabe, hébreu)."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", text):
        if out and unicodedata.category(ch).startswith("M"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def ipa_tokens(ipa: str) -> list[str]:
    """Découpe une transcription IPA en symboles : base + diacritiques/modificateurs
    (ʰ ʲ ː ̃ …) ; les ligatures (t͡s) sont conservées ; accents/frontières ignorés."""
    tokens: list[str] = []
    glue_next = False
    for ch in unicodedata.normalize("NFC", ipa):
        if ch in IPA_IGNORED:
            glue_next = False
            continue
        cat = unicodedata.category(ch)
        if tokens and (cat in ("Mn", "Mc", "Me", "Lm", "Sk") or ch in TIE_BARS or glue_next):
            tokens[-1] += ch
            glue_next = ch in TIE_BARS
        else:
            tokens.append(ch)
            glue_next = False
    return tokens


def ipa_syllables(ipa: str) -> list[str] | None:
    """Syllabes IPA uniquement si la source contient des frontières explicites ('.').
    Sinon None : on ne syllabifie jamais par règle inventée."""
    if SYLLABLE_SEP not in ipa:
        return None
    return [s for s in (p.strip("/[] ˈˌ") for p in ipa.split(SYLLABLE_SEP)) if s]
