"""Adaptateurs de modèle factices : reproductibles (graine), sans réseau, sans horloge.

Le comportement vient du code, jamais d'un appel de modèle. Chaque étape est une
fonction PURE de (graine, numéro d'étape, sorties des étapes précédentes) : c'est ce
qui permet à un second adaptateur de reprendre à l'étape n depuis le seul compte rendu.

  honnete    limite et budget corrects
  optimiste  annonce un budget 3 fois trop bas (et conclut GO, tout « solide »)
  bavard     dépasse sa limite de contexte déclarée
  casse_*    sorties invalides (champ manquant, tag hors vocabulaire, texte hors JSON)
"""
from __future__ import annotations

import random

from .budget import CHARS_PER_TOKEN, STEP_NAMES, est_tokens
from .live import LlamaCppAdapter, ManualAdapter, OllamaAdapter
from .schema import TAGS
from .trace import canonical_json

AXES = ("word~etymon", "etymon~phoneme", "word~phoneme")
OPERATORS = ("Advance", "Merge", "Compress", "ProjectToLLM", "S_t")
BASELINES = ("caractères", "byte-level", "tokenizer natif", "sous-chaînes sans gématrie", "trie",
             "features aléatoires", "embeddings appris", "petit pré-encodeur", "sans EmbedBabel")
CONTROLS = ("permutation", "dictionnaire bruité", "langue OOD", "scramble numérique")
INTEGRATION = ("pré-tokenizer", "tokenizer", "prefill", "draft", "KV-cache", "couche externe")
CLAIM_TAGS = ("TESTABLE", "PLAUSIBLE", "SPECULATIF")


def _produce(step: int, prior: dict, info: dict, optimistic: bool) -> dict:
    """Contenu d'une étape. Pure : dépend de info['seed'], de `step` et de `prior`."""
    rng = random.Random(f"{info['seed']}:{step}")
    if step == 1:
        n_ops = 3 + rng.randint(0, 2)
        return {
            "meta": {"model": info["model"], "seed": info["seed"], "schema": "v2-conv-0.2"},
            "executive_verdict": {"decision": "GO" if optimistic else "NO-GO",
                                  "summary": "gain non démontré, H0 conservée" if not optimistic
                                  else "gain prometteur"},
            "formalization": {"graph": "G=(V,E,R)", "operators": list(OPERATORS[:n_ops])},
        }
    if step == 2:
        n_ops = len(prior["formalization"]["operators"])
        axes = list(AXES)
        if optimistic:
            cmap = {"solid": axes, "fragile": [], "illusory": []}
        else:
            rng.shuffle(axes)
            cmap = {"solid": [], "fragile": axes[:1], "illusory": axes[1:]}
        return {
            "correlation_map": cmap,
            "anti_mystical_audit": {"sdm": 10 if optimistic else 40 + 5 * n_ops + rng.randint(0, 5),
                                    "justification": f"{n_ops} opérateurs formalisés, aucune preuve de gain"},
            "fruitful_intuitions": [
                {"text": f"intuition {i + 1}", "sif": rng.randint(20, 80),
                 "justification": "testable par ablation"} for i in range(3 + n_ops % 3)],
        }
    intuitions = prior["fruitful_intuitions"]
    claims = []
    for i, it in enumerate(intuitions):
        claim = {"text": it["text"], "tag": "DEFINI" if optimistic else CLAIM_TAGS[i % len(CLAIM_TAGS)]}
        claim["refutation_test" if i == 0 else "justification"] = "permutation des étiquettes"
        claims.append(claim)
    return {
        "experimental_plan": {"baselines": list(BASELINES), "controls": list(CONTROLS)},
        "system_integration": {"points": list(INTEGRATION), "hidden_costs": ["mémoire", "latence"]},
        "smallest_decisive_experiment": {"description": "ablation gématrie contre hash aléatoire",
                                         "refutation_test": "gain nul sous permutation"},
        "claim_tagging_summary": {"claims": claims},
    }


class MockAdapter:
    kind = "honnete"
    optimistic = False
    pad = False            # bavard : gonfle la dernière étape au-delà de la limite de contexte
    plan_divisor = 1       # optimiste : budget annoncé = vrai coût // 3
    defect: str | None = None
    #: Les adaptateurs factices **annoncent** un budget (`plan()`), donc l'écart au budget annoncé
    #: est un critère qui a un sens pour eux. Un adaptateur réel ne déclare rien : voir `live.py`.
    declares_budget = True

    def __init__(self, context_limit: int) -> None:
        self.context_limit = context_limit

    def declare(self) -> dict:
        return {"context_limit": self.context_limit}

    def plan(self, info: dict) -> list[int]:
        """Estimation de tokens par étape, calculée par simulation à vide (sans remplissage)."""
        prior: dict = {}
        estimates = []
        for k in range(1, len(STEP_NAMES) + 1):
            out = _produce(k, prior, info, self.optimistic)
            prior.update(out)
            estimates.append(max(1, est_tokens(canonical_json(out)) // self.plan_divisor))
        return estimates

    def step(self, k: int, prior: dict, info: dict) -> dict:
        out = _produce(k, prior, info, self.optimistic)
        if self.pad and k == len(STEP_NAMES):
            room = max(0, self.context_limit - info["prompt_tokens"])
            out["system_integration"]["commentary"] = "x" * (room * CHARS_PER_TOKEN + 400)
        return out

    def finalize(self, merged: dict) -> str:
        data = dict(merged)
        if self.defect == "champ":
            data.pop("formalization", None)
        elif self.defect == "tag":
            data["claim_tagging_summary"] = {"claims": [
                {**c, "tag": "TAG_INCONNU"} for c in data["claim_tagging_summary"]["claims"]]}
        text = canonical_json(data)
        return "Voici l'analyse demandée : " + text if self.defect == "texte" else text


class Honest(MockAdapter):
    kind = "honnete"


class Optimistic(MockAdapter):
    kind = "optimiste"
    optimistic = True
    plan_divisor = 3


class Verbose(MockAdapter):
    kind = "bavard"
    pad = True


class BrokenField(MockAdapter):
    kind = "casse_champ"
    defect = "champ"


class BrokenTag(MockAdapter):
    kind = "casse_tag"
    defect = "tag"


class BrokenText(MockAdapter):
    kind = "casse_texte"
    defect = "texte"


ADAPTERS = {c.kind: c for c in (Honest, Optimistic, Verbose, BrokenField, BrokenTag, BrokenText,
                                OllamaAdapter, LlamaCppAdapter, ManualAdapter)}


def build(entry: dict):
    """Instancie l'adaptateur d'une entrée {name, adapter, context_limit[, params]}."""
    return ADAPTERS[entry["adapter"]](entry["context_limit"], **entry.get("params", {}))


assert all(t in TAGS for t in CLAIM_TAGS)  # cohérence avec schema.TAGS
