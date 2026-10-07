"""Arbre de raisonnement : étapes, sous-étapes, appels de plugin, conclusions.

Déterministe : identifiants hiérarchiques ("1", "1.2", "1.2.1"), aucune horloge.
Sérialisable en JSON canonique, donc comparable octet par octet à graine égale.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field

KINDS = ("step", "plugin_call", "conclusion")


@dataclass
class Node:
    id: str
    kind: str
    title: str
    data: dict = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title,
                "data": self.data, "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(d["id"], d["kind"], d["title"], d.get("data", {}),
                   [cls.from_dict(c) for c in d.get("children", [])])


class TraceBuilder:
    def __init__(self, title: str = "bench") -> None:
        self.root = Node("0", "step", title)
        self._stack: list[Node] = [self.root]

    def _add(self, kind: str, title: str, data: dict | None) -> Node:
        parent = self._stack[-1]
        prefix = "" if parent.id == "0" else parent.id + "."
        node = Node(f"{prefix}{len(parent.children) + 1}", kind, title, dict(data or {}))
        parent.children.append(node)
        return node

    @contextmanager
    def step(self, title: str, **data):
        node = self._add("step", title, data)
        self._stack.append(node)
        try:
            yield node
        finally:
            self._stack.pop()

    def plugin_call(self, plugin: str, action: str, **data) -> Node:
        return self._add("plugin_call", f"{plugin}:{action}", {"plugin": plugin, "action": action, **data})

    def conclusion(self, title: str, **data) -> Node:
        return self._add("conclusion", title, data)

    def to_dict(self) -> dict:
        return self.root.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> "TraceBuilder":
        tb = cls(d["title"])
        tb.root = Node.from_dict(d)
        tb._stack = [tb.root]
        return tb


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def validate_tree(d: dict, _seen: set | None = None) -> list[str]:
    """Erreurs de structure : id manquant/dupliqué, genre inconnu, ids enfants incohérents."""
    errors: list[str] = []
    seen = _seen if _seen is not None else set()
    nid = d.get("id")
    if not nid:
        errors.append("nœud sans id")
    elif nid in seen:
        errors.append(f"id dupliqué : {nid}")
    seen.add(nid)
    if d.get("kind") not in KINDS:
        errors.append(f"{nid}: genre inconnu {d.get('kind')!r}")
    for i, child in enumerate(d.get("children", []), 1):
        prefix = "" if nid == "0" else f"{nid}."
        if child.get("id") != f"{prefix}{i}":
            errors.append(f"{child.get('id')}: id attendu {prefix}{i}")
        errors.extend(validate_tree(child, seen))
    return errors


def walk(d: dict):
    yield d
    for c in d.get("children", []):
        yield from walk(c)
