"""Boucle d'auto-amélioration et auto-réglage — point d'entrée.

    # 1. ce que les tours ont donné, à partir d'issues EXTERNES
    python -m bench.selfimprove.run score --decisions ~/.dsh/embedbabel/dsh-turns.jsonl \
        --outcomes artifacts/outcomes.jsonl

    # 2. décision pré-déclarée : promouvoir ou rejeter
    python -m bench.selfimprove.run accept --baseline artifacts/base.json \
        --candidate artifacts/cand.json --margin 5

    # 3. auto-réglage piloté par les ressources locales (une passe)
    python -m bench.selfimprove.run tune --config artifacts/perf/config-local.json \
        --space artifacts/space.json --workload decode-512 --metric decode_tok_s \
        --vram-budget-mib 23000 --gpu-temp-max-c 85 --margin 5 --max-trials 8

    # 4. mode continu : détecte la dérive et retente le voisinage
    python -m bench.selfimprove.run tune --config artifacts/perf/config-local.json \
        --space artifacts/space.json --workload decode-512 --metric decode_tok_s \
        --vram-budget-mib 23000 --watch --rounds 0 --interval 900
    python -m bench.selfimprove.run envelope

`space.json` : `{"ngl": [99, 40, 20], "context": [8192, 32768], "kv_type": ["f16", "q8_0"],
"batch": [2048, 4096], "concurrency": [1, 2]}` — seuls les axes listés dans `tune.TUNABLE_AXES`
sont acceptés.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import records as records_mod
from . import score as scoring
from . import tune as tuning


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cmd_score(args) -> int:
    loaded = records_mod.load(args.decisions, args.outcomes)
    guard = tuning.dataset_guard(loaded)
    print(json.dumps({"dataset_hash": records_mod.dataset_hash(loaded), **guard},
                     indent=2, ensure_ascii=False))
    if args.split:
        train, holdout = records_mod.split(loaded, seed=args.seed)
        print(f"\nentraînement : {len(train)} tours · tenus à l'écart : {len(holdout)} tours")
    return 0 if guard["aggregate"].get("metrics") else 1


def cmd_accept(args) -> int:
    decision = scoring.compare(_load_json(args.baseline), _load_json(args.candidate),
                               margin_pct=args.margin)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if decision["decision"] == "promouvoir" else 1


def _build_tuner(args) -> tuning.Tuner:
    rails = tuning.Rails(vram_budget_mib=args.vram_budget_mib,
                         gpu_temp_max_c=args.gpu_temp_max_c,
                         min_free_ram_mib=args.min_free_ram_mib)
    return tuning.Tuner(_load_json(args.config), _load_json(args.space), args.workload, args.metric,
                        rails=rails, margin_pct=args.margin, min_reps=args.repetitions,
                        seed=args.seed, out_dir=args.out, gpu_index=args.gpu_index)


def cmd_tune(args) -> int:
    tuner = _build_tuner(args)
    state = tuner.staleness(tuner.load_profile())
    print(json.dumps({"enveloppe": tuning.envelope(args.gpu_index), "profil": state},
                     indent=2, ensure_ascii=False))
    if args.dry_run:
        profile = tuner.load_profile()
        axes = (profile or {}).get("axes") or tuner._default_axes()
        print(json.dumps({"incumbent proposé": axes,
                          "voisinage": tuner.neighbourhood(axes)[:args.max_trials]},
                         indent=2, ensure_ascii=False))
        return 0
    if args.watch:
        tuner.watch(rounds=args.rounds, interval_s=args.interval, max_trials=args.max_trials)
        return 0
    result = tuner.run_round(max_trials=args.max_trials)
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2, ensure_ascii=False))
    return 0 if result.get("decision") else 1


def cmd_envelope(args) -> int:
    print(json.dumps(tuning.envelope(args.gpu_index), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.selfimprove.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="agrège des tours à partir d'issues externes")
    s.add_argument("--decisions", type=Path, required=True, help="journal JSONL du plugin DSH")
    s.add_argument("--outcomes", type=Path, help="fichier JSONL d'issues (sans lui : rien à scorer)")
    s.add_argument("--split", action="store_true", help="affiche aussi la séparation tenue à l'écart")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_score)

    a = sub.add_parser("accept", help="décision pré-déclarée entre deux agrégats")
    a.add_argument("--baseline", type=Path, required=True)
    a.add_argument("--candidate", type=Path, required=True)
    a.add_argument("--margin", type=float, default=5.0, help="marge exigée en %% (défaut 5)")
    a.set_defaults(func=cmd_accept)

    t = sub.add_parser("tune", help="auto-réglage piloté par les ressources locales")
    t.add_argument("--config", type=Path, required=True, help="matrice de base (bench/perf)")
    t.add_argument("--space", type=Path, required=True, help="valeurs admissibles par axe")
    t.add_argument("--workload", default="decode-512")
    t.add_argument("--metric", default="decode_tok_s", choices=sorted(tuning.OBJECTIVES))
    t.add_argument("--vram-budget-mib", type=int)
    t.add_argument("--gpu-temp-max-c", type=float)
    t.add_argument("--min-free-ram-mib", type=int)
    t.add_argument("--margin", type=float, default=5.0)
    t.add_argument("--repetitions", type=int, default=3)
    t.add_argument("--max-trials", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--gpu-index", type=int, default=0)
    t.add_argument("--out", type=Path, default=Path("artifacts") / "tune")
    t.add_argument("--watch", action="store_true", help="tours successifs (détection de dérive)")
    t.add_argument("--rounds", type=int, default=0, help="0 = jusqu'à interruption")
    t.add_argument("--interval", type=float, default=900.0, help="secondes entre deux tours")
    t.add_argument("--dry-run", action="store_true", help="n'exécute rien : montre le voisinage")
    t.set_defaults(func=cmd_tune)

    e = sub.add_parser("envelope", help="environnement matériel courant (invalide un profil)")
    e.add_argument("--gpu-index", type=int, default=0)
    e.set_defaults(func=cmd_envelope)


    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
