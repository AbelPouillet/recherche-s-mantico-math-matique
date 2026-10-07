"""Registre d'exécution **hors des fichiers suivis par git**.

Pourquoi ce module existe
------------------------
`bench/registry/<id>.json` est une **entrée** du harnais : il fige la définition d'une version, son
hash et l'historique de ses exécutions passées. Avant, l'état RAPPORT y ajoutait chaque nouvelle
exécution : un simple `python -m bench.run` **modifiait un fichier versionné**, ce qui interdit toute
CI honnête (gate « arbre propre » impossible, jobs parallèles en conflit, artefacts non
reproductibles).

Désormais `bench.registry` est en lecture seule pour le lanceur : seul `--register` l'écrit.
Chaque exécution dépose son propre enregistrement **dans son dossier de sortie** :

    <out>/run_record.json

Aucune écriture partagée, donc aucune course entre jobs parallèles, et l'agrégation devient une
opération explicite (`python -m bench.ledger collect --root bench/runs`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .trace import canonical_json

RECORD_NAME = "run_record.json"


def record_path(out_dir: Path) -> Path:
    return Path(out_dir) / RECORD_NAME


def write_record(out_dir: Path, summary: dict) -> Path:
    """Écrit l'enregistrement d'exécution du dossier `out_dir`. Idempotent : même contenu, même octets."""
    path = record_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return path


def read_record(out_dir: Path) -> dict | None:
    path = record_path(out_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def iter_records(root: Path):
    """Tous les enregistrements sous `root`, triés par dossier (donc indépendant de l'ordre du disque)."""
    for path in sorted(Path(root).rglob(RECORD_NAME)):
        try:
            yield path.parent.name, json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue


def collect(root: Path) -> dict:
    """Agrège les enregistrements par harnais. Retourne {harness: [exécutions triées]}."""
    by_harness: dict[str, list[dict]] = {}
    for run_name, record in iter_records(root):
        record = {**record, "run": run_name}
        by_harness.setdefault(record.get("harness", "?"), []).append(record)
    return {h: sorted(r, key=lambda e: (e.get("seed", -1), e.get("task", ""), e["run"]))
            for h, r in sorted(by_harness.items())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.ledger", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="agrège les enregistrements d'exécution sous un dossier")
    c.add_argument("--root", type=Path, default=Path("bench/runs"))
    args = p.parse_args(argv)
    data = collect(args.root)
    if not data:
        print(f"aucun {RECORD_NAME} sous {args.root}", file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
