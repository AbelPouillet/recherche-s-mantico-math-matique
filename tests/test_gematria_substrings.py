import pytest

from embedbabel.gematria import get_alphabet, load_config
from embedbabel.substrings import contiguous_substrings, count_substring_occurrences, prefixes
from embedbabel.textnorm import graphemes, ipa_syllables, ipa_tokens


def test_hebrew_standard_and_final_forms():
    he = get_alphabet("he")
    assert he.size == 22
    assert he.gematria("אבג") == 6
    assert he.gematria("שלום") == 300 + 30 + 6 + 40          # ם final -> מ
    assert he.gematria("שָׁלוֹם") == he.gematria("שלום")      # niqqud optionnel supprimé


def test_greek_isopsephy_and_accents():
    el = get_alphabet("el")
    assert el.size == 24
    assert el.gematria("αβγ") == 6
    assert el.gematria("ς") == el.gematria("σ") == 200
    assert el.gematria("άλφα") == el.gematria("αλφα") == 1 + 30 + 500 + 1


def test_arabic_abjad_folding():
    ar = get_alphabet("ar")
    assert ar.size == 28
    assert ar.gematria("أبجد") == 1 + 2 + 3 + 4
    assert ar.gematria("بِسْمِ") == 2 + 60 + 40            # tashkīl supprimé
    assert ar.letter_value("غ") == 1000


def test_latin_ordinal_and_pythagorean():
    fr = get_alphabet("fr")
    assert fr.size == 26
    assert fr.gematria("abc") == 6
    assert fr.letters_of("Écureuil") == list("ecureuil")
    assert fr.gematria("j", "pythagorean") == 1 and fr.gematria("i", "pythagorean") == 9
    assert fr.letters_of("œuf") == list("oeuf")


def test_spanish_keeps_enye_and_german_sharp_s():
    assert get_alphabet("es").letters_of("año") == list("año")
    assert get_alphabet("es").size == 27
    assert get_alphabet("de").letters_of("Straße") == list("strasse")


def test_every_system_covers_alphabet():
    cfg = load_config()
    for lang, spec in cfg["languages"].items():
        alpha = get_alphabet(lang, cfg)
        for system in alpha.system_names:
            for c in alpha.letters:
                assert alpha.letter_value(c, system) > 0


def test_unknown_language_and_non_alphabetic():
    with pytest.raises(KeyError):
        get_alphabet("xx")
    zh = get_alphabet("zh")
    assert not zh.alphabetic and zh.letters_of("你好a") == ["你", "好"]
    with pytest.raises(ValueError):
        zh.gematria("你")


def test_prefixes_and_substrings():
    assert prefixes(list("mai")) == ["m", "ma", "mai"]
    subs = list(contiguous_substrings(list("abc")))
    assert [s for _, s in subs] == ["a", "ab", "abc", "b", "bc", "c"]
    assert [s for _, s in contiguous_substrings(list("abc"), 2)] == ["a", "ab", "b", "bc", "c"]
    assert count_substring_occurrences(6) == 21
    assert count_substring_occurrences(6, 2) == 6 + 5
    assert all(len(list(contiguous_substrings(list("x" * n), 3))) == count_substring_occurrences(n, 3)
               for n in range(1, 8))


def test_graphemes_and_ipa():
    assert graphemes("e\u0301a") == ["é", "a"]
    assert ipa_tokens("ˈe.ky.ʁœj") == ["e", "k", "y", "ʁ", "œ", "j"]
    assert ipa_tokens("t͡sa") == ["t͡s", "a"]
    assert ipa_tokens("aː") == ["aː"]
    assert ipa_tokens("ɑ̃") == ["ɑ̃"]
    assert ipa_syllables("e.ky.ʁœj") == ["e", "ky", "ʁœj"]
    assert ipa_syllables("ekyʁœj") is None
