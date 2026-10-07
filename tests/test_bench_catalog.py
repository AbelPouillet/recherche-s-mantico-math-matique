"""Catalogue des harnais versionnés (bench/harnesses) : nommage, documentation, inscription au registre."""
import json
import re
from pathlib import Path

import pytest

from bench import registry

ROOT = Path(__file__).resolve().parent.parent
HARNESSES = ROOT / "bench" / "harnesses"
DEFS = sorted(HARNESSES.glob("*/*/harness.def.json"))
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
FAMILY = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def test_catalog_is_not_empty_and_has_its_documents():
    assert DEFS
    assert (HARNESSES / "README.md").read_text(encoding="utf-8").strip()
    assert (HARNESSES / "CHANGELOG.md").read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("path", DEFS, ids=lambda p: f"{p.parent.parent.name}-{p.parent.name}")
def test_each_harness_is_named_documented_and_registered(path):
    defn = json.loads(path.read_text(encoding="utf-8"))
    family, version = path.parent.parent.name, path.parent.name
    hid = f"{family}-{version}"
    assert FAMILY.match(family) and SEMVER.match(version)
    assert (defn["name"], defn["version"]) == (family, version)
    assert defn["doc"] == f"bench/harnesses/{family}/{version}/HARNESS.md"
    doc = (ROOT / defn["doc"]).read_text(encoding="utf-8")
    assert doc.startswith(f"# {family} {version}\n") and hid in doc
    assert hid in (HARNESSES / "README.md").read_text(encoding="utf-8")
    assert f"### {version}" in (HARNESSES / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = registry.load(hid)
    assert entry["definition"] == defn
    assert registry.verify(hid), "un fichier couvert par le hash a changé : créer une nouvelle version"


def test_legacy_0_1_0_hash_is_unchanged_by_the_new_hash_rules():
    assert registry.verify("embedbabel-bench-0.1.0")
