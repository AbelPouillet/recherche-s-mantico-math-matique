import build_graph as bg
from embedbabel.gematria import get_alphabet


def run(rows, lang="fr", k=3):
    sink = bg.CountSink()
    stats = bg.build_language(lang, rows, get_alphabet(lang), k, sink)
    return sink, stats


ROWS = [("maison", "mɛzɔ̃", "fr-FR", "t"), ("mai", "mɛ", "fr-FR", "t"), ("mai", "me", "fr-FR", "t")]


def test_shared_substrings_and_counts():
    sink, stats = run(ROWS)
    # sous-chaînes (len 2..3) distinctes de « maison » : 5 + 4 ; « mai » : ma, ai, mai (ma, ai, mai déjà vus)
    assert stats["substrings"] == 9
    assert sink.nodes["Letter"] == 26
    assert sink.nodes["Word"] == 2                       # mai (2 IPA) et maison
    assert sink.nodes["Pronunciation"] == 3
    assert sink.rels["NEXT_LETTER"] == 25
    assert sink.rels["PRONOUNCED_AS"] == 3
    assert sink.rels["COMPOSED_OF"] == 9 * 2 + 6 + 3     # DAG (2 par sous-chaîne) + lettres des mots
    assert sink.rels["PREFIX_OF"] == 3 + 2               # maison : m, ma, mai (len <= 3) ; mai : m, ma


def test_max_substring_len_reduces_size():
    big, _ = run(ROWS, k=6)
    small, _ = run(ROWS, k=2)
    assert small.nodes["Substring"] < big.nodes["Substring"]
    assert small.rels["SUBSTRING_OF"] < big.rels["SUBSTRING_OF"]


def test_non_alphabetic_language_uses_observed_chars():
    sink, _ = run([("你好", "ni˧˥ xɑʊ̯˨˩˦", "zh-cmn", "t")], lang="zh", k=2)
    assert sink.nodes["Letter"] == 2 and sink.rels.get("NEXT_LETTER", 0) == 0


def test_etymology_loading(tmp_path):
    tsv = tmp_path / "e.tsv"
    tsv.write_text("src_lang\tsrc_text\tdst_lang\tdst_text\ttype\trule\tconfidence\tsource\ttemplate\t"
                   "evidence\tretrieved\tlicense\n"
                   "fro\tescurel\tfr\técureuil\tsound_change\t\t1.0\ts\tinh\te\t2026-01-01\tl\n",
                   encoding="utf-8")
    sink = bg.CountSink()
    n = bg.load_etymology(tsv, sink, {("fr", "écureuil", "fr-FR")})
    assert n == 1 and sink.rels["SOUND_CHANGE"] == 1 and sink.rels["SAME_FORM"] == 1
