"""Registre versionné des harnais de bench.

Les définitions sources sont dans bench/definitions/*.def.json ; `register` les
inscrit dans bench/registry/<name>-<version>.json :
  {"definition": {...}, "content_hash": "...", "runs": [{...}]}
Le hash couvre la définition ET le contenu du prompt référencé : modifier l'un
ou l'autre sans changer `version` est refusé (pas d'écrasement silencieux).
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


def compute_hash(defn: dict, root: Path = ROOT) -> str:
    prompt = (root / defn["prompt"]).read_bytes()
    return digest({"definition": defn, "prompt_sha256": hashlib.sha256(prompt).hexdigest()})


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


def record_run(hid: str, summary: dict, registry_dir: Path | None = None) -> None:
    entry = load(hid, registry_dir)
    if summary not in entry["runs"]:  # idempotent : relancer à l'identique n'ajoute rien
        entry["runs"].append(summary)
        _write(_path(hid, registry_dir), entry)


def list_harnesses(registry_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (registry_dir or REGISTRY_DIR).glob("*.json"))
