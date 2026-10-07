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


def test_major_version_changes_nothing_the_model_sees():
    """1.0.0 est une version MAJEURE (la façon de mesurer change) mais ne doit pas changer le contexte.

    C'est ce qui rend la comparabilité vérifiable plutôt qu'affirmée : quand tous les paquets tiennent
    dans la limite, le texte envoyé au modèle est octet pour octet celui de 0.4.0. Seule la
    *required*-ness du paquet « projet » a changé, et le tri des paquets est stable.
    """
    from bench import context

    texts = {}
    for version in ("0.4.0", "1.0.0"):
        defn = json.loads((HARNESSES / "embedbabel-bench" / version / "harness.def.json")
                          .read_text(encoding="utf-8"))
        packets = context.load_packets(defn["tasks"]["v2-complet"]["packets"], ROOT)
        ctx, text = context.build_context(packets, 32768, reserve=4000)
        assert ctx["ok"] and not ctx["left_out"], f"{version} : tout doit tenir pour ce test"
        texts[version] = (text, [p["packet"] for p in ctx["loaded"]])
    assert texts["0.4.0"] == texts["1.0.0"], (
        "1.0.0 ne doit pas modifier le contexte quand tout tient ; sinon la comparabilité est rompue")


def test_the_project_description_is_now_required_so_a_model_cannot_be_graded_blind():
    """Le prompt de sortie renvoie à PROMPT.md : le rendre optionnel notait des analyses à l'aveugle."""
    old = json.loads((HARNESSES / "embedbabel-bench" / "0.4.0" / "harness.def.json")
                     .read_text(encoding="utf-8"))
    new = json.loads((HARNESSES / "embedbabel-bench" / "1.0.0" / "harness.def.json")
                     .read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in new["tasks"]["v2-complet"]["packets"]}
    assert by_id["projet"]["required"] is True
    assert all(p["id"] != "projet" or not p.get("required")
               for p in old["tasks"]["v2-complet"]["packets"])
