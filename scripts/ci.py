"""CI locale — rejoue exactement ce que fait le workflow, sur la machine courante.

    python scripts/ci.py              # suite complète + gate d'herméticité
    python scripts/ci.py --quick      # suite du harnais seulement (sans numpy/neo4j/qdrant)
    python scripts/ci.py --keep       # conserve le dossier de sortie du run hors ligne

Pourquoi ce script existe
------------------------
Une CI ne se vérifie pas en la poussant. Ce script exécute **le même code** que
`.github/workflows/ci.yml`, donc il n'y a pas de dérive possible entre ce qu'on croit testé et ce
qui l'est. Le point sensible qu'il vérifie est l'**herméticité** : un run du harnais ne doit
modifier aucun fichier suivi par git. Sinon le gate « arbre propre » est impossible et deux jobs
parallèles se marchent dessus.

Code de sortie : 0 si tout passe, 1 sinon.

Le plugin DeepSeek Harness (`dsh/embedbabel-dsh/`) est du JavaScript : il est vérifié par
`node --test`, lancé **depuis** `tests/test_bench_guard.py` (donc par la suite pytest ci-dessus, sans
étape supplémentaire). Si `node` est absent, ce test est ignoré avec une raison explicite.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_SUITE = ["tests/test_bench_schema.py", "tests/test_bench_harness.py", "tests/test_bench_context.py",
               "tests/test_bench_catalog.py", "tests/test_bench_live.py", "tests/test_bench_golden.py",
               "integrations"]


def run_step(title: str, cmd: list[str], cwd: Path = ROOT) -> bool:
    print(f"\n=== {title} ===\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace")
    ok = proc.returncode == 0
    print(f"--- {title} : {'OK' if ok else 'ÉCHEC'}", flush=True)
    return ok


def git_status() -> str:
    """État de l'arbre de travail, normalisé (vide = arbre propre)."""
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return "\n".join(sorted(line for line in proc.stdout.splitlines() if line.strip()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts/ci.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="ne lance que la suite du harnais")
    p.add_argument("--keep", action="store_true", help="conserve le dossier du run hors ligne")
    args = p.parse_args(argv)

    py = sys.executable
    targets = BENCH_SUITE if args.quick else []
    results: dict[str, bool] = {}

    results["tests"] = run_step("Suite de tests", [py, "-m", "pytest", "-q", *targets])

    before = git_status()
    out = Path(tempfile.mkdtemp(prefix="embedbabel-ci-")) if not args.keep else ROOT / "artifacts" / "ci-run"
    try:
        results["run hors ligne"] = run_step(
            "Run hors ligne du harnais (empreinte dorée)",
            [py, "-m", "bench.run", "--models", "bench/models/offline-mock.json",
             "--task", "v2-complet", "--budget", "3000", "--seed", "0", "--out", str(out)])

        after = git_status()
        results["herméticité"] = (before == after)
        print("\n=== Herméticité ===")
        print(f"$ git status --porcelain   (avant le run)")
        print(before or "(arbre propre)")
        if before != after:
            print("\nFICHIERS MODIFIÉS PAR LE RUN :")
            for line in sorted(set(after.splitlines()) - set(before.splitlines())):
                print(f"  {line}")
            print("Un run du harnais ne doit jamais écrire dans un fichier suivi par git.")
    finally:
        if not args.keep:
            shutil.rmtree(out, ignore_errors=True)

    print("\n================ RÉSUMÉ ================")
    for name, ok in results.items():
        print(f"  {'OK   ' if ok else 'ÉCHEC'}  {name}")
    failed = [n for n, ok in results.items() if not ok]
    print("========================================")
    if failed:
        print(f"CI en échec : {', '.join(failed)}")
        return 1
    print("CI verte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
