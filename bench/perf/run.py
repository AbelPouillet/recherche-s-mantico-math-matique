"""Mesure de la pipeline d'inférence — point d'entrée.

    python -m bench.perf.run --config bench/perf/matrix.example.json --out artifacts/perf/2026-10-07
    python -m bench.perf.run --config bench/perf/matrix.example.json --dry-run
    python -m bench.perf.run --models bench/models/llamacpp-local.json --host http://127.0.0.1:8080 \
        --workloads decode-512,prefill-4k --repetitions 3 --out artifacts/perf/strata

Ce que ce harnais est, et n'est pas
----------------------------------
Il mesure le **débit, la latence et la mémoire** de la pipeline d'inférence sous une matrice de
configuration. Il ne produit aucun jugement de qualité : c'est le rôle du harnais H1
(`python -m bench.run`). Les deux rapports sont séparés volontairement.

Trois règles de méthode, appliquées par défaut :

1. **nonce unique par requête** — sinon un cache de prompt fait passer un préfixe réutilisé pour un
   prefill mesuré ;
2. **streaming** — sinon il n'existe aucun instant « premier token » à mesurer ;
3. **`cache_n` publié** — une ligne dont le prefill a été rejoué est marquée `replayed_prefill` et
   comptée à part dans l'agrégat.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from . import matrix
from .resources import host_summary


def config_from_host(host: str, label: str = "host") -> dict:
    """Config minimale ciblant un serveur **déjà lancé** (Strata, llama-server, vLLM…)."""
    return {"engines": [{"id": label, "kind": "host", "params": {"host": host}}],
            "workloads": ["decode-128"], "repetitions": 3, "warmups": 1, "seed": 0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.perf.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="fichier JSON de matrice (voir matrix.example.json)")
    p.add_argument("--host", help="cible un serveur déjà lancé, sans fichier de config")
    p.add_argument("--label", default="serveur", help="nom du moteur ciblé par --host")
    p.add_argument("--workloads", help="charges séparées par des virgules (défaut : celles du fichier)")
    p.add_argument("--engines", help="identifiants de moteurs à retenir, séparés par des virgules")
    p.add_argument("--repetitions", type=int, help="répétitions par cellule et par charge")
    p.add_argument("--warmups", type=int, help="requêtes de chauffe avant les mesures")
    p.add_argument("--gpu-index", type=int, default=0, help="GPU échantillonné par nvidia-smi")
    p.add_argument("--out", type=Path, help="dossier de sortie (défaut : artifacts/perf/<date>)")
    p.add_argument("--dry-run", action="store_true", help="n'exécute rien : imprime le plan")
    p.add_argument("--list-workloads", action="store_true", help="liste les charges prédéfinies")
    args = p.parse_args(argv)

    if args.list_workloads:
        for name, spec in sorted(matrix.WORKLOADS.items()):
            print(f"{name:14} prompt~{spec['prompt_tokens']:>6} tokens, sortie {spec['max_tokens']:>4} tokens")
        return 0

    if args.config:
        # `utf-8-sig` : les outils Windows (PowerShell, Notepad) écrivent volontiers un BOM, et un
        # fichier de configuration ne doit pas faire échouer une campagne pour cette raison.
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    elif args.host:
        config = config_from_host(args.host, args.label)
    else:
        p.error("--config ou --host est requis (--list-workloads pour voir les charges)")

    out = args.out or Path("artifacts") / "perf" / datetime.date.today().isoformat()
    report = matrix.run_matrix(
        config, out, dry_run=args.dry_run,
        only_engines=args.engines.split(",") if args.engines else None,
        only_workloads=args.workloads.split(",") if args.workloads else None,
        repetitions=args.repetitions, warmups=args.warmups, gpu_index=args.gpu_index)

    if args.dry_run:
        print(json.dumps({"host": report["host"], "plan": report["plan"]}, indent=2, ensure_ascii=False))
        return 0

    paths = matrix.write_report(report, out)
    print(f"\n{matrix.markdown_table(report)}\n")
    print(f"rapport  : {paths['report']}\nrésumé   : {paths['summary']}")
    failed = [c["cell"] for c in report["cells"] if c.get("error")]
    if failed:
        print(f"\ncellules en échec : {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
