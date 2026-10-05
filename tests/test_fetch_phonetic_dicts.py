import fetch_phonetic_dicts as fd


def test_parse_wikipron(tmp_path):
    f = tmp_path / "a.tsv"
    f.write_text("squirrel\ts k w ɝ l\nsquirrel\ts k w ɝ l\nx\t\n", encoding="utf-8")
    assert list(fd.parse_wikipron(f)) == [("squirrel", "skwɝl")] * 2


def test_parse_ipadict_variants(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("écureuil\t/ekʏʁœj/, /ekyʁœj/\n", encoding="utf-8")
    assert list(fd.parse_ipadict(f)) == [("écureuil", "ekʏʁœj"), ("écureuil", "ekyʁœj")]


def test_build_lang_offline_dedup(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "dict"
    for source, fname, content in (("wikipron", "fra_latn_broad.tsv", "a\ta\na\ta\n"),
                                   ("ipa-dict", "fr_QC.txt", "a\t/a/\n")):
        (raw / source).mkdir(parents=True)
        (raw / source / fname).write_text(content, encoding="utf-8")
    n = fd.build_lang("fr", raw, out, {"wikipron": "x", "ipa-dict": "x"}, True, {})
    lines = (out / "fr.tsv").read_text(encoding="utf-8").splitlines()
    assert n == 2 and lines[0] == "word\tipa\tvariety\tsource"
    assert {l.split("\t")[2] for l in lines[1:]} == {"fr-FR", "fr-QC"}


def test_all_requested_languages_configured():
    assert {"fr", "en", "de", "es", "ar", "he", "el", "zh", "yue"} <= set(fd.SOURCES)
    assert {v for v, _, _ in fd.SOURCES["fr"]} == {"fr-FR", "fr-QC"}
    assert {v for v, _, _ in fd.SOURCES["en"]} == {"en-US", "en-UK"}
