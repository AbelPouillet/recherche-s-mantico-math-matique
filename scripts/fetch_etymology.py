#!/usr/bin/env python3
"""Extrait des arêtes étymologiques (cognate, borrowing, sound_change, contraction,
contact_blend) à partir des modèles de la section « Etymology » du Wiktionnaire anglais.

Principe : AUCUNE règle n'est déduite librement. Chaque arête vient d'un modèle de
wikitexte explicite ({{inh}}, {{bor}}, {{cog}}, {{contraction}}, {{blend}}) et porte sa
provenance (page, langue, modèle, extrait). Le champ `rule` n'est rempli que si une règle
documentée existe dans data/etymology/rules.tsv (avec sa source) ; sinon il reste vide.

Modes :
  --wikitext-dir DIR : lit des fichiers <lang>__<titre>.wikitext locaux (hors-ligne, tests)
  --words fr:écureuil en:squirrel ... : récupère via l'API MediaWiki (réseau requis),
                                        met en cache dans data/raw/etymology/
Sortie : data/etymology/edges.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI_API = "https://en.wiktionary.org/w/api.php"
LICENSE = "CC BY-SA (texte Wiktionary)"

# Correspondance code Wiktionary -> nom de section de langue (liste minimale)
SECTION_NAMES = {"fr": "French", "en": "English", "de": "German", "es": "Spanish",
                 "ar": "Arabic", "he": "Hebrew", "el": "Greek", "zh": "Chinese",
                 "la": "Latin", "fro": "Old French", "enm": "Middle English"}

# Famille de modèles -> type d'arête. Correspondance PRÉSUMÉE : à valider contre la
# documentation des modèles Wiktionary (vérification impossible depuis l'environnement
# de développement, accès réseau bloqué). `der` est exclu par défaut car ambigu.
LINEAGE = {"inh": ("sound_change", 1.0), "inh+": ("sound_change", 1.0)}
BORROWING = {"bor": 1.0, "bor+": 1.0, "lbor": 1.0, "slbor": 1.0, "sbor": 1.0}
DERIVED = {"der": 0.5, "der+": 0.5}
COGNATE = {"cog", "cognate", "cog+"}
CONTRACTION = {"contraction", "contracted"}
BLEND = {"blend", "bl"}

EDGE_FIELDS = ["src_lang", "src_text", "dst_lang", "dst_text", "type", "rule",
               "confidence", "source", "template", "evidence", "retrieved", "license"]


@dataclass
class Edge:
    src_lang: str
    src_text: str
    dst_lang: str
    dst_text: str
    type: str
    rule: str
    confidence: float
    source: str
    template: str
    evidence: str
    retrieved: str = ""
    license: str = LICENSE


def parse_templates(text: str) -> list[tuple[str, list[str], dict[str, str], str]]:
    """Modèles {{nom|a|b|k=v}} de premier niveau -> (nom, positionnels, nommés, texte brut)."""
    out, i, n = [], 0, len(text)
    while i < n - 1:
        if text.startswith("{{", i):
            depth, j = 1, i + 2
            while j < n - 1 and depth:
                if text.startswith("{{", j):
                    depth += 1; j += 2
                elif text.startswith("}}", j):
                    depth -= 1; j += 2
                else:
                    j += 1
            if depth:
                break
            raw = text[i:j]
            parts, buf, d = [], "", 0
            inner = raw[2:-2]
            for k, ch in enumerate(inner):
                if inner.startswith("{{", k) or inner.startswith("[[", k):
                    d += 1
                if inner.startswith("}}", k) or inner.startswith("]]", k):
                    d -= 1
                if ch == "|" and d == 0:
                    parts.append(buf); buf = ""
                else:
                    buf += ch
            parts.append(buf)
            name = parts[0].strip()
            pos, named = [], {}
            for part in parts[1:]:
                m = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=(.*)$", part, re.S)
                if m and not part.lstrip().startswith("["):
                    named[m.group(1)] = m.group(2).strip()
                else:
                    pos.append(part.strip())
            out.append((name, pos, named, raw))
            i = j
        else:
            i += 1
    return out


def etymology_section(wikitext: str, lang_code: str) -> str:
    """Texte de la (des) section(s) `Etymology` de la langue demandée."""
    name = SECTION_NAMES.get(lang_code)
    if name is None:
        raise KeyError(f"nom de section inconnu pour « {lang_code} » (ajouter à SECTION_NAMES)")
    m = re.search(rf"^==\s*{re.escape(name)}\s*==\s*$(.*?)(?=^==[^=]|\Z)", wikitext, re.M | re.S)
    if not m:
        return ""
    chunks = re.findall(r"^===\s*Etymology(?:\s+\d+)?\s*===\s*$(.*?)(?=^===[^=]|\Z)",
                        m.group(1), re.M | re.S)
    return "\n".join(chunks)


def _clean(term: str) -> str:
    term = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", term).strip()
    return term


def load_rules(path: Path) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    """Règles documentées : (src_lang, dst_lang, src_text, dst_text) -> (règle, source)."""
    rules: dict = {}
    if not path.exists():
        return rules
    with open(path, encoding="utf-8") as fh:
        rows = csv.DictReader((l for l in fh if l.strip() and not l.startswith("#")), delimiter="\t")
        for r in rows:
            if not r.get("source"):
                raise ValueError(f"règle sans source refusée : {r}")
            rules[(r["src_lang"], r["dst_lang"], r["src_text"], r["dst_text"])] = (r["rule"], r["source"])
    return rules


def parse_etymology(wikitext: str, lang: str, title: str, rules: dict | None = None,
                    include_der: bool = False, retrieved: str = "") -> list[Edge]:
    rules = rules or {}
    section = etymology_section(wikitext, lang)
    source = f"en.wiktionary.org/wiki/{urllib.parse.quote(title)}#{SECTION_NAMES[lang]}"
    edges: list[Edge] = []
    known: dict[str, str] = {lang: title}   # dernier terme connu par langue (chaîne d'ancêtres)

    def add(src_lang, src_text, dst_lang, dst_text, typ, conf, template, raw):
        if not src_text or src_text == "-" or not dst_text:
            return
        rule, rsrc = rules.get((src_lang, dst_lang, src_text, dst_text), ("", ""))
        edges.append(Edge(src_lang, src_text, dst_lang, dst_text, typ, rule, conf,
                          source + (f" ; règle : {rsrc}" if rsrc else ""), template,
                          raw[:200], retrieved))

    for name, pos, named, raw in parse_templates(section):
        base = name.lower()
        if base in LINEAGE or base in BORROWING or (include_der and base in DERIVED):
            if len(pos) < 3:
                continue
            dst_l, src_l, term = pos[0], pos[1], _clean(pos[2])
            dst_t = known.get(dst_l)
            if dst_t is None:
                continue
            if base in LINEAGE:
                typ, conf = LINEAGE[base]
            elif base in BORROWING:
                typ, conf = "borrowing", BORROWING[base]
            else:
                typ, conf = "borrowing", DERIVED[base]
            add(src_l, term, dst_l, dst_t, typ, conf, name, raw)
            known[src_l] = term
        elif base in COGNATE:
            if len(pos) >= 2 and _clean(pos[1]):
                # arête symétrique, orientée par ordre lexicographique de (lang, texte)
                a, b = (lang, title), (pos[0], _clean(pos[1]))
                a, b = sorted([a, b])
                add(a[0], a[1], b[0], b[1], "cognate", 1.0, name, raw)
        elif base in CONTRACTION:
            parts = [_clean(x) for x in pos[1:] if _clean(x)]
            for part in parts:
                add(pos[0] if pos else lang, part, lang, title, "contraction", 1.0, name, raw)
        elif base in BLEND:
            parts = [_clean(x) for x in pos[1:] if _clean(x)]
            for part in parts:
                add(pos[0] if pos else lang, part, lang, title, "contact_blend", 0.5, name, raw)
    return edges


def fetch_wikitext(lang: str, title: str, cache_dir: Path, user_agent: str, delay: float) -> str:
    cache = cache_dir / f"{lang}__{title}.wikitext"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    params = urllib.parse.urlencode({
        "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "titles": title, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(f"{WIKI_API}?{params}", headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    page = data["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"page absente : {title}")
    text = page["revisions"][0]["slots"]["main"]["content"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    time.sleep(delay)
    return text


def write_edges(edges: list[Edge], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, uniq = set(), []
    for e in edges:
        key = (e.src_lang, e.src_text, e.dst_lang, e.dst_text, e.type)
        if key not in seen:
            seen.add(key); uniq.append(e)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EDGE_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for e in uniq:
            w.writerow(asdict(e))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--wikitext-dir", type=Path, help="dossier de fichiers <lang>__<titre>.wikitext")
    src.add_argument("--words", nargs="+", metavar="LANG:MOT", help="ex. fr:écureuil en:squirrel (réseau)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "etymology" / "edges.tsv")
    p.add_argument("--rules", type=Path, default=ROOT / "data" / "etymology" / "rules.tsv")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "raw" / "etymology")
    p.add_argument("--include-der", action="store_true",
                   help="inclure {{der}} comme borrowing (confiance 0.5, ambigu)")
    p.add_argument("--user-agent", default="embedbabel-research/0.1 (https://github.com/AbelPouillet)")
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args(argv)

    rules = load_rules(args.rules)
    today = date.today().isoformat()
    items: list[tuple[str, str, str]] = []
    if args.wikitext_dir:
        for f in sorted(args.wikitext_dir.glob("*__*.wikitext")):
            lang, title = f.stem.split("__", 1)
            items.append((lang, title, f.read_text(encoding="utf-8")))
    else:
        for w in args.words:
            lang, title = w.split(":", 1)
            items.append((lang, title, fetch_wikitext(lang, title, args.cache_dir,
                                                      args.user_agent, args.delay)))
    edges: list[Edge] = []
    for lang, title, text in items:
        edges += parse_etymology(text, lang, title, rules, args.include_der, today)
    write_edges(edges, args.out)
    print(f"{len(edges)} arêtes -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
