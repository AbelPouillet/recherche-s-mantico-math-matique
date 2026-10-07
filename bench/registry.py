"""Registre versionné des harnais de bench — **entrée en lecture seule pour le lanceur**.

Les définitions sources sont dans bench/definitions/*.def.json ; `register` les
inscrit dans bench/registry/<name>-<version>.json :
  {"definition": {...}, "content_hash": "...", "runs": [{...}]}
Le hash couvre la définition ET le contenu du prompt référencé : modifier l'un
ou l'autre sans changer `version` est refusé (pas d'écrasement silencieux).

`runs` est un **historique figé** des exécutions antérieures à l'introduction du registre
d'exécution (`bench/ledger.py`). Le lanceur n'écrit plus ici : un run dépose son
enregistrement dans son propre dossier de sortie, sinon un simple `python -m bench.run`
salirait un fichier suivi par git et aucune CI ne pourrait exiger un arbre propre.
Seul `--register` écrit dans ce dossier.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .trace import digest

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = Path(__file__).resolve().parent / "registry"

REQUIRED = ("name", "version", "prompt", "preferences")


def harness_id(defn: dict) -> str:
    return f"{defn['name']}-{defn['version']}"


def _extra_files(defn: dict) -> list[str]:
    """Fichiers couverts par le hash en plus du prompt : documentation et paquets des tâches."""
    paths = [defn["doc"]] if "doc" in defn else []
    for task in defn.get("tasks", {}).values():
        paths += [p["path"] for p in task.get("packets", [])]
    return sorted(set(paths) - {defn["prompt"]})


def _sha(path: Path) -> str:
    # fins de ligne normalisées : git (core.autocrlf) réécrit LF <-> CRLF selon la machine,
    # le hash ne doit pas en dépendre
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def compute_hash(defn: dict, root: Path = ROOT) -> str:
    payload = {"definition": defn, "prompt_sha256": _sha(root / defn["prompt"])}
    extra = _extra_files(defn)
    if extra:  # absent des définitions 0.1.0 : leur hash reste inchangé
        payload["extra_sha256"] = {p: _sha(root / p) for p in extra}
    return digest(payload)


def _path(hid: str, registry_dir: Path | None) -> Path:
    return (registry_dir or REGISTRY_DIR) / f"{hid}.json"


def _write(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load(hid: str, registry_dir: Path | None = None) -> dict:
    return json.loads(_path(hid, registry_dir).read_text(encoding="utf-8"))


def register(defn: dict, registry_dir: Path | None = None, root: Path = ROOT) -> dict:
    missing = [k for k in REQUIRED if k not in defn]
    if missing:
        raise ValueError(f"définition incomplète, clés manquantes : {missing}")
    hid, new_hash = harness_id(defn), compute_hash(defn, root)
    if _path(hid, registry_dir).exists():
        entry = load(hid, registry_dir)
        if entry["content_hash"] != new_hash:
            raise ValueError(f"{hid} existe avec un contenu différent : incrémentez `version`")
        return entry
    entry = {"definition": defn, "content_hash": new_hash, "runs": []}
    _write(_path(hid, registry_dir), entry)
    return entry


def verify(hid: str, registry_dir: Path | None = None, root: Path = ROOT) -> bool:
    entry = load(hid, registry_dir)
    return compute_hash(entry["definition"], root) == entry["content_hash"]


def list_harnesses(registry_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (registry_dir or REGISTRY_DIR).glob("*.json"))
