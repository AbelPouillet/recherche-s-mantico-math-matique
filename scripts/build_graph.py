#!/usr/bin/env python3
"""Construit le graphe Neo4j des dictionnaires : lettres (feuilles) -> sous-chaînes
(branches partagées) -> mots, avec prononciations IPA et arêtes étymologiques.

Modes de sortie (combinables) : --dry-run (comptes seulement), --export-csv DIR
(neo4j-admin import), chargement Neo4j par lots UNWIND/MERGE (par défaut).
Voir graph/README.md pour l'explosion combinatoire (--max-substring-len).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedbabel.gematria import Alphabet, get_alphabet, load_config  # noqa: E402
from embedbabel.substrings import contiguous_substrings, prefixes  # noqa: E402
from embedbabel.textnorm import ipa_syllables  # noqa: E402

# label -> (propriétés clés, {propriété: type CSV})
NODE_SPECS = {
    "Letter": (["lang", "char"], {"alphabet_index": "int", "gematria_systems": "string[]",
                                  "gematria_values": "int[]"}),
    "Substring": (["lang", "text"], {"length": "int", "gematria": "int"}),
    "Word": (["lang", "text", "variety"], {"ipa": "string[]", "period": "string"}),
    "Pronunciation": (["lang", "ipa", "variety"], {}),
    "Syllable": (["lang", "ipa"], {"text": "string"}),
}
# propriété qui distingue plusieurs arêtes de même type entre deux nœuds
REL_MERGE_PROP = {"COMPOSED_OF": "pos", "SUBSTRING_OF": "start", "HAS_SYLLABLE": "pos"}
REL_PROP_TYPES = {"pos": "int", "start": "int", "confidence": "float"}
ETYMOLOGY_TYPES = {"cognate": "COGNATE", "borrowing": "BORROWING", "sound_change": "SOUND_CHANGE",
                   "contraction": "CONTRACTION", "contact_blend": "CONTACT_BLEND"}


def node_id(label: str, key: dict) -> str:
    return "|".join(str(key[k]) for k in NODE_SPECS[label][0])


# --------------------------------------------------------------------------- sinks
class Sink:
    def node(self, label: str, key: dict, props: dict | None = None) -> None: ...
    def rel(self, typ: str, sl: str, sk: dict, dl: str, dk: dict, props: dict | None = None) -> None: ...
    def close(self) -> None: ...


class CountSink(Sink):
    """Compte nœuds/arêtes par type (dry-run) ; ne stocke rien."""
    def __init__(self) -> None:
        self.nodes: Counter = Counter()
        self.rels: Counter = Counter()

    def node(self, label, key, props=None):
        self.nodes[label] += 1

    def rel(self, typ, sl, sk, dl, dk, props=None):
        self.rels[typ] += 1


class Neo4jSink(Sink):
    def __init__(self, uri: str, user: str, password: str, batch_size: int, database: str | None):
        from neo4j import GraphDatabase  # import tardif : inutile en dry-run
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.batch = batch_size
        self.buf: dict[tuple, list] = defaultdict(list)

    @staticmethod
    def _match(var: str, label: str, side: str) -> str:
        keys = NODE_SPECS[label][0]
        return f"({var}:{label} {{" + ", ".join(f"{k}: r.{side}.{k}" for k in keys) + "})"

    def _query(self, kind: tuple) -> str:
        if kind[0] == "node":
            label = kind[1]
            keys = NODE_SPECS[label][0]
            pat = ", ".join(f"{k}: r.key.{k}" for k in keys)
            return f"UNWIND $rows AS r MERGE (n:{label} {{{pat}}}) SET n += r.props"
        _, typ, sl, dl = kind
        mp = REL_MERGE_PROP.get(typ)
        rel = f"[e:{typ} {{{mp}: r.props.{mp}}}]" if mp else f"[e:{typ}]"
        return (f"UNWIND $rows AS r MATCH {self._match('a', sl, 's')} MATCH {self._match('b', dl, 'd')} "
                f"MERGE (a)-{rel}->(b) SET e += r.props")

    def _flush(self, kind: tuple) -> None:
        rows = self.buf.pop(kind, [])
        if rows:
            query = self._query(kind)
            with self.driver.session(database=self.database) as session:
                session.execute_write(lambda tx: tx.run(query, rows=rows).consume())

    def _add(self, kind: tuple, row: dict) -> None:
        self.buf[kind].append(row)
        if len(self.buf[kind]) >= self.batch:
            if kind[0] == "rel":   # MATCH exige que tous les nœuds en attente existent déjà
                self.flush_nodes()
            self._flush(kind)

    def flush_nodes(self) -> None:
        for kind in [k for k in list(self.buf) if k[0] == "node"]:
            self._flush(kind)

    def node(self, label, key, props=None):
        self._add(("node", label), {"key": key, "props": _clean(props)})

    def rel(self, typ, sl, sk, dl, dk, props=None):
        self._add(("rel", typ, sl, dl), {"s": sk, "d": dk, "props": _clean(props)})

    def flush_all(self) -> None:
        self.flush_nodes()
        for kind in list(self.buf):
            self._flush(kind)

    def close(self) -> None:
        self.flush_all()
        self.driver.close()

    def apply_schema(self, path: Path) -> None:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.strip().startswith("//")]
        stmts = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
        with self.driver.session(database=self.database) as session:
            for s in stmts:
                session.run(s).consume()


class CsvSink(Sink):
    """Fichiers pour `neo4j-admin database import full` (en-têtes séparés)."""
    def __init__(self, out_dir: Path):
        self.dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[tuple, tuple] = {}
        self.seen_nodes: set[tuple[str, str]] = set()

    def _writer(self, kind: tuple, columns: list[str], header: list[str], name: str):
        if kind not in self.files:
            with open(self.dir / f"{name}_header.csv", "w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow(header)
            fh = open(self.dir / f"{name}.csv", "w", encoding="utf-8", newline="")
            self.files[kind] = (fh, csv.writer(fh), columns)
        return self.files[kind]

    @staticmethod
    def _fmt(v):
        if isinstance(v, (list, tuple)):
            return ";".join(str(x) for x in v)
        return "" if v is None else v

    def node(self, label, key, props=None):
        nid = node_id(label, key)
        if (label, nid) in self.seen_nodes:
            return
        self.seen_nodes.add((label, nid))
        keys, ptypes = NODE_SPECS[label]
        pcols = list(ptypes)
        header = [f"id:ID({label})"] + keys + [f"{p}:{t}" for p, t in ptypes.items()] + [":LABEL"]
        _, w, _ = self._writer(("node", label), pcols, header, f"nodes_{label}")
        props = props or {}
        w.writerow([nid] + [key[k] for k in keys] + [self._fmt(props.get(p)) for p in pcols] + [label])

    def rel(self, typ, sl, sk, dl, dk, props=None):
        props = props or {}
        pcols = sorted(props)
        kind = ("rel", typ, sl, dl)
        header = [f":START_ID({sl})", f":END_ID({dl})"] + [
            f"{p}:{REL_PROP_TYPES[p]}" if p in REL_PROP_TYPES else p for p in pcols] + [":TYPE"]
        _, w, cols = self._writer(kind, pcols, header, f"rels_{typ}_{sl}_{dl}")
        w.writerow([node_id(sl, sk), node_id(dl, dk)] + [self._fmt(props.get(p)) for p in cols] + [typ])

    def close(self) -> None:
        for fh, _, _ in self.files.values():
            fh.close()


class Tee(Sink):
    """Envoie à plusieurs sinks et compte (rapport de taille)."""
    def __init__(self, *sinks: Sink):
        self.sinks = sinks

    def node(self, *a, **k):
        for s in self.sinks:
            s.node(*a, **k)

    def rel(self, *a, **k):
        for s in self.sinks:
            s.rel(*a, **k)

    def close(self):
        for s in self.sinks:
            s.close()


def _clean(props: dict | None) -> dict:
    return {k: v for k, v in (props or {}).items() if v is not None}


# ----------------------------------------------------------------- construction
def read_dict(path: Path, varieties: set[str] | None = None, limit: int | None = None):
    with open(path, encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if limit is not None and i >= limit:
                break
            if varieties and row["variety"] not in varieties:
                continue
            yield row["word"], row["ipa"], row["variety"], row["source"]


def build_language(lang: str, rows, alphabet: Alphabet, max_len: int, sink: Sink,
                   system: str | None = None) -> dict:
    """Construit le graphe d'une langue dans `sink`. Retourne des statistiques."""
    words: dict[tuple[str, str], list[str]] = {}
    for text, ipa, variety, _src in rows:
        ipas = words.setdefault((text, variety), [])
        if ipa not in ipas:
            ipas.append(ipa)
    gem = (lambda letters: alphabet.gematria(letters, system)) if alphabet.systems else (lambda l: None)

    def lett(text: str):
        return "Letter" if len(text) == 1 else "Substring"

    def keyof(text: str):
        return {"lang": lang, "char": text} if len(text) == 1 else {"lang": lang, "text": text}

    # --- passe 1 : unités des mots, ensemble partagé des sous-chaînes
    units_of: dict[tuple[str, str], list[str]] = {}
    substrings: set[str] = set()
    letters_seen: set[str] = set()
    for (text, variety) in words:
        units = alphabet.letters_of(text)
        if not units:
            continue
        units_of[(text, variety)] = units
        letters_seen.update(units)
        for _, s in contiguous_substrings(units, max_len):
            if len(s) > 1:
                substrings.add(s)

    # --- feuilles : lettres de l'alphabet (ou caractères observés si non alphabétique)
    if alphabet.alphabetic:
        for i, c in enumerate(alphabet.letters):
            sink.node("Letter", {"lang": lang, "char": c}, {
                "alphabet_index": i, "gematria_systems": alphabet.system_names or None,
                "gematria_values": [alphabet.letter_value(c, s) for s in alphabet.system_names] or None})
        for a, b in zip(alphabet.letters, alphabet.letters[1:]):
            sink.rel("NEXT_LETTER", "Letter", {"lang": lang, "char": a}, "Letter", {"lang": lang, "char": b})
    else:
        for c in sorted(letters_seen):
            sink.node("Letter", {"lang": lang, "char": c}, {})

    # --- branches : sous-chaînes partagées entre mots (DAG)
    for s in sorted(substrings):
        sink.node("Substring", {"lang": lang, "text": s}, {"length": len(s), "gematria": gem(list(s))})
    for s in sorted(substrings):
        for pos, child in ((0, s[:-1]), (1, s[1:])):
            sink.rel("COMPOSED_OF", "Substring", {"lang": lang, "text": s}, lett(child), keyof(child),
                     {"pos": pos})

    # --- mots, prononciations, liens
    pron_seen: set[tuple[str, str]] = set()
    for (text, variety), ipas in words.items():
        wkey = {"lang": lang, "text": text, "variety": variety}
        sink.node("Word", wkey, {"ipa": ipas})
        for ipa in ipas:
            pkey = {"lang": lang, "ipa": ipa, "variety": variety}
            if (ipa, variety) not in pron_seen:
                pron_seen.add((ipa, variety))
                sink.node("Pronunciation", pkey, {})
                syl = ipa_syllables(ipa)
                for pos, sy in enumerate(syl or []):
                    sink.node("Syllable", {"lang": lang, "ipa": sy}, {})
                    sink.rel("HAS_SYLLABLE", "Pronunciation", pkey, "Syllable", {"lang": lang, "ipa": sy},
                             {"pos": pos})
            sink.rel("PRONOUNCED_AS", "Word", wkey, "Pronunciation", pkey, {})
        units = units_of.get((text, variety))
        if not units:
            continue
        for pos, c in enumerate(units):
            sink.rel("COMPOSED_OF", "Word", wkey, "Letter", {"lang": lang, "char": c}, {"pos": pos})
        for p in prefixes(units)[:-1]:
            if len(p) <= max_len:
                sink.rel("PREFIX_OF", lett(p), keyof(p), "Word", wkey, {})
        for start, s in contiguous_substrings(units, max_len):
            if len(s) < len(units):
                sink.rel("SUBSTRING_OF", lett(s), keyof(s), "Word", wkey, {"start": start})
    return {"words": len(words), "indexed_words": len(units_of), "substrings": len(substrings)}


def load_etymology(path: Path, sink: Sink, known_words: set[tuple[str, str]]) -> int:
    n = 0
    ends: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            typ = ETYMOLOGY_TYPES.get(r["type"])
            if not typ:
                continue
            for lg, tx in ((r["src_lang"], r["src_text"]), (r["dst_lang"], r["dst_text"])):
                if (lg, tx) not in ends:
                    ends.add((lg, tx))
                    sink.node("Word", {"lang": lg, "text": tx, "variety": ""}, {})
            props = {"source": r["source"], "confidence": float(r["confidence"] or 0),
                     "rule": r["rule"] or None, "template": r["template"],
                     "retrieved": r["retrieved"], "license": r["license"]}
            sink.rel(typ, "Word", {"lang": r["src_lang"], "text": r["src_text"], "variety": ""},
                     "Word", {"lang": r["dst_lang"], "text": r["dst_text"], "variety": ""}, props)
            n += 1
    for lg, tx in ends:
        for variety in sorted(v for (l2, t2, v) in known_words if (l2, t2) == (lg, tx)):
            sink.rel("SAME_FORM", "Word", {"lang": lg, "text": tx, "variety": ""},
                     "Word", {"lang": lg, "text": tx, "variety": variety}, {})
    return n


def format_report(report: dict) -> str:
    lines = []
    for lang, r in report.items():
        lines.append(f"[{lang}] mots={r['words']} indexés={r['indexed_words']} "
                     f"sous-chaînes distinctes={r['substrings']} "
                     f"| nœuds={sum(r['nodes'].values())} arêtes={sum(r['rels'].values())}")
        lines.append("   nœuds : " + ", ".join(f"{k}={v}" for k, v in sorted(r["nodes"].items())))
        lines.append("   arêtes : " + ", ".join(f"{k}={v}" for k, v in sorted(r["rels"].items())))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dict-dir", type=Path, default=ROOT / "data" / "dict")
    p.add_argument("--config", type=Path, default=None, help="config/gematria_systems.yaml par défaut")
    p.add_argument("--langs", nargs="+", help="langues à traiter (défaut : tous les data/dict/*.tsv)")
    p.add_argument("--varieties", nargs="+", help="restreindre à ces variétés (ex. fr-FR en-US)")
    p.add_argument("--max-substring-len", type=int, default=4,
                   help="longueur maximale des sous-chaînes indexées (défaut 4) ; croissance quadratique sinon")
    p.add_argument("--max-words", type=int, help="ne lire que les N premières lignes de chaque dictionnaire")
    p.add_argument("--system", help="système de gématrie (défaut : premier système de la langue)")
    p.add_argument("--etymology", type=Path, help="TSV produit par fetch_etymology.py")
    p.add_argument("--dry-run", action="store_true", help="aucune base : seulement le rapport de taille")
    p.add_argument("--export-csv", type=Path, help="écrit des CSV pour neo4j-admin import")
    p.add_argument("--apply-schema", action="store_true", help="exécuter graph/schema.cypher avant le chargement")
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--report", type=Path, help="écrit le rapport de taille en JSON")
    p.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--database", default=os.environ.get("NEO4J_DATABASE"))
    args = p.parse_args(argv)

    if args.max_substring_len < 1:
        p.error("--max-substring-len doit être >= 1")
    cfg = load_config(args.config)
    langs = args.langs or sorted(f.stem for f in args.dict_dir.glob("*.tsv"))
    if not langs:
        p.error(f"aucun dictionnaire dans {args.dict_dir} (lancer scripts/fetch_phonetic_dicts.py)")

    neo = None
    sinks: list[Sink] = []
    if args.export_csv:
        csv_sink = CsvSink(args.export_csv)
        sinks.append(csv_sink)
    if not args.dry_run and not args.export_csv:
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            p.error("NEO4J_PASSWORD non défini (ou utiliser --dry-run / --export-csv)")
        neo = Neo4jSink(args.uri, args.user, password, args.batch_size, args.database)
        if args.apply_schema:
            neo.apply_schema(ROOT / "graph" / "schema.cypher")
        sinks.append(neo)

    report: dict = {}
    known: set[tuple[str, str, str]] = set()
    for lang in langs:
        path = args.dict_dir / f"{lang}.tsv"
        if not path.exists():
            p.error(f"{path} introuvable")
        counter = CountSink()
        sink = Tee(counter, *sinks)
        rows = list(read_dict(path, set(args.varieties) if args.varieties else None, args.max_words))
        known.update((lang, t, v) for t, _i, v, _s in rows)
        stats = build_language(lang, rows, get_alphabet(lang, cfg), args.max_substring_len, sink, args.system)
        stats.update(nodes=dict(counter.nodes), rels=dict(counter.rels))
        report[lang] = stats
    if args.etymology:
        counter = CountSink()
        n = load_etymology(args.etymology, Tee(counter, *sinks), known)
        report["_etymology"] = {"words": 0, "indexed_words": 0, "substrings": 0,
                                "nodes": dict(counter.nodes), "rels": dict(counter.rels)}
    for s in sinks:
        s.close()
    print(format_report(report))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
