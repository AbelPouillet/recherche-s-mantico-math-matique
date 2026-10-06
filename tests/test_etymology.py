from pathlib import Path

import pytest

import fetch_etymology as fe

FIX = Path(__file__).parent / "fixtures" / "etymology"


def edges_for(name, lang, title, **kw):
    return fe.parse_etymology((FIX / name).read_text(encoding="utf-8"), lang, title, **kw)


def as_set(edges):
    return {(e.src_lang, e.src_text, e.dst_lang, e.dst_text, e.type) for e in edges}


def test_ecureuil_chain():
    edges = edges_for("fr__écureuil.wikitext", "fr", "écureuil")
    assert as_set(edges) == {
        ("fro", "escurel", "fr", "écureuil", "sound_change"),
        ("la", "sciurus", "fro", "escurel", "sound_change"),
        ("grc", "σκίουρος", "la", "sciurus", "borrowing"),
        ("en", "squirrel", "fr", "écureuil", "cognate"),
    }
    assert all(e.source.startswith("en.wiktionary.org/wiki/") and e.template for e in edges)
    assert all(e.rule == "" for e in edges)      # aucune règle sans source documentée


def test_squirrel_borrowing_from_old_french():
    edges = edges_for("en__squirrel.wikitext", "en", "squirrel")
    assert ("fro", "escurel", "enm", "squirel", "borrowing") in as_set(edges)
    assert ("enm", "squirel", "en", "squirrel", "sound_change") in as_set(edges)


def test_documented_rule_is_attached_only_with_source():
    rules = {("fro", "fr", "escurel", "écureuil"): ("règle de test", "réf. de test")}
    edges = edges_for("fr__écureuil.wikitext", "fr", "écureuil", rules=rules)
    e = next(e for e in edges if e.src_text == "escurel")
    assert e.rule == "règle de test" and "réf. de test" in e.source


def test_rule_without_source_rejected(tmp_path):
    f = tmp_path / "r.tsv"
    f.write_text("src_lang\tdst_lang\tsrc_text\tdst_text\trule\tsource\nfro\tfr\ta\tb\tr\t\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fe.load_rules(f)


def test_contraction_blend_and_der_option():
    wt = ("==English==\n===Etymology===\n{{contraction|en|do|not}} {{blend|en|smoke|fog}} "
          "{{der|en|la|foo}}\n")
    edges = fe.parse_etymology(wt, "en", "x")
    assert {e.type for e in edges} == {"contraction", "contact_blend"}
    edges = fe.parse_etymology(wt, "en", "x", include_der=True)
    assert any(e.type == "borrowing" and e.confidence == 0.5 for e in edges)


def test_no_etymology_section_gives_no_edges():
    assert fe.parse_etymology("==English==\n===Noun===\nfoo\n", "en", "foo") == []


def test_cli_writes_tsv(tmp_path):
    out = tmp_path / "edges.tsv"
    assert fe.main(["--wikitext-dir", str(FIX), "--out", str(out), "--rules", str(tmp_path / "none.tsv")]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == fe.EDGE_FIELDS and len(lines) == 1 + 6
