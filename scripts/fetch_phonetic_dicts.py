#!/usr/bin/env python3
"""Télécharge et normalise des dictionnaires mot -> IPA.

Sources réellement utilisées (vérifiées, voir data/SOURCES.md) :
  - WikiPron (CUNY-CL/wikipron, dossier data/scrape/tsv) : données issues de Wiktionary.
  - open-dict-data/ipa-dict : uniquement pour fr-QC (seule source trouvée pour cette variété).

Sorties :
  data/raw/<source>/<fichier>   téléchargements bruts (gitignorés) + data/raw/manifest.json
  data/dict/<lang>.tsv          colonnes : word, ipa, variety, source

Aucune donnée n'est inventée : une variété sans source vérifiée n'est pas produite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WIKIPRON_URL = "https://raw.githubusercontent.com/CUNY-CL/wikipron/{ref}/data/scrape/tsv/{file}"
IPADICT_URL = "https://raw.githubusercontent.com/open-dict-data/ipa-dict/{ref}/data/{file}"

# lang -> liste de (variété, source, fichier distant)
SOURCES: dict[str, list[tuple[str, str, str]]] = {
    "fr": [("fr-FR", "wikipron", "fra_latn_broad.tsv"),
           ("fr-QC", "ipa-dict", "fr_QC.txt")],
    "en": [("en-US", "wikipron", "eng_latn_us_broad.tsv"),
           ("en-UK", "wikipron", "eng_latn_uk_broad.tsv")],
    "de": [("de", "wikipron", "deu_latn_broad.tsv")],
    "es": [("es-ES", "wikipron", "spa_latn_ca_broad.tsv"),
           ("es-419", "wikipron", "spa_latn_la_broad.tsv")],
    "ar": [("ar", "wikipron", "ara_arab_broad.tsv")],
    "he": [("he", "wikipron", "heb_hebr_broad.tsv")],
    "el": [("el", "wikipron", "ell_grek_broad.tsv")],
    "zh": [("zh-cmn", "wikipron", "cmn_hani_standard_broad.tsv")],
    "yue": [("zh-yue", "wikipron", "yue_hani_standard_broad.tsv")],
}

# Limites connues, affichées par --coverage (voir data/SOURCES.md pour le détail)
CAVEATS = {
    "fr-FR": "WikiPron `fra` sans étiquette de dialecte (standard) ; étiquette fr-FR = convention du projet.",
    "fr-QC": "ipa-dict : généré par règles (qc-ipa), « très expérimental » selon ses auteurs ; absent de WikiPron.",
    "ar": "WikiPron `ara` : un seul fichier, sans variété précisée ; pas de dialectes.",
    "he": "WikiPron `heb` : ~6,8 k lignes seulement ; plusieurs transcriptions par mot (variantes).",
    "zh-cmn": "Clés en caractères Han (surtout 1 caractère/entrée), IPA avec chiffres de ton ; pas de pinyin fourni.",
    "zh-yue": "Idem mandarin : clés en caractères Han ; pas de romanisation fournie.",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "embedbabel-research/0.1"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"échec du téléchargement de {url} : {exc}") from exc
            time.sleep(2 * attempt)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def parse_wikipron(path: Path):
    """Lignes `mot<TAB>phonèmes séparés par des espaces` -> (mot, IPA sans espaces)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip():
                continue
            ipa = nfc("".join(parts[1].split()))
            if ipa:
                yield nfc(parts[0].strip()), ipa


def parse_ipadict(path: Path):
    """Lignes `mot<TAB>/ipa/` (variantes séparées par « , ») -> (mot, IPA)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip():
                continue
            for variant in parts[1].split(","):
                ipa = nfc(variant.strip().strip("/").strip())
                if ipa:
                    yield nfc(parts[0].strip()), ipa


PARSERS = {"wikipron": parse_wikipron, "ipa-dict": parse_ipadict}
URLS = {"wikipron": WIKIPRON_URL, "ipa-dict": IPADICT_URL}


def build_lang(lang: str, raw_dir: Path, dict_dir: Path, ref: dict[str, str],
               offline: bool, manifest: dict) -> int:
    seen: set[tuple[str, str, str]] = set()
    rows: list[tuple[str, str, str, str]] = []
    for variety, source, fname in SOURCES[lang]:
        local = raw_dir / source / fname
        url = URLS[source].format(ref=ref[source], file=fname)
        if not local.exists():
            if offline:
                raise FileNotFoundError(f"--offline : {local} absent")
            print(f"[{lang}] téléchargement {url}", file=sys.stderr)
            download(url, local)
            manifest[f"{source}/{fname}"] = {
                "url": url, "sha256": sha256_file(local), "bytes": local.stat().st_size,
                "retrieved": date.today().isoformat()}
        tag = f"{source}:{fname.rsplit('.', 1)[0]}"
        for word, ipa in PARSERS[source](local):
            key = (word, ipa, variety)
            if key not in seen:
                seen.add(key)
                rows.append((word, ipa, variety, tag))
    rows.sort()
    dict_dir.mkdir(parents=True, exist_ok=True)
    out = dict_dir / f"{lang}.tsv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write("word\tipa\tvariety\tsource\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"[{lang}] {len(rows)} lignes -> {out}", file=sys.stderr)
    return len(rows)


def print_coverage() -> None:
    print("Langue  Variété   Source     Fichier                         Réserves")
    for lang, entries in SOURCES.items():
        for variety, source, fname in entries:
            print(f"{lang:<7} {variety:<9} {source:<10} {fname:<31} {CAVEATS.get(variety, '')}")
    print("\nNon couvert (aucune source vérifiée n'a été utilisée) : dialectes arabes, "
          "hébreu vocalisé à grande échelle, pinyin/jyutping, grec ancien/moyen, "
          "périodes historiques (le champ `period` reste vide).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--langs", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--dict-dir", type=Path, default=ROOT / "data" / "dict")
    p.add_argument("--wikipron-ref", default="master", help="branche/tag/commit de CUNY-CL/wikipron")
    p.add_argument("--ipadict-ref", default="master", help="branche/tag/commit de open-dict-data/ipa-dict")
    p.add_argument("--offline", action="store_true", help="n'utiliser que data/raw déjà présent")
    p.add_argument("--coverage", action="store_true", help="afficher couverture/réserves et quitter")
    args = p.parse_args(argv)

    if args.coverage:
        print_coverage()
        return 0
    manifest_path = args.raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    refs = {"wikipron": args.wikipron_ref, "ipa-dict": args.ipadict_ref}
    for lang in args.langs:
        build_lang(lang, args.raw_dir, args.dict_dir, refs, args.offline, manifest)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
