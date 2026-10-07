"""Validation de la sortie JSON stricte V2 : sans réseau, sans horloge."""
import copy
import json

import pytest

from bench import schema
from bench.adapters import Honest

INFO = {"seed": 1, "model": "m", "prompt_tokens": 100, "context_limit": 12000}


def valid_obj() -> dict:
    a, prior = Honest(12000), {}
    for k in (1, 2, 3):
        prior.update(a.step(k, prior, INFO))
    return json.loads(a.finalize(prior))


def check(mutate):
    obj = copy.deepcopy(valid_obj())
    mutate(obj)
    return schema.validate(json.dumps(obj))


def test_valid_output_passes():
    parsed, errors = schema.validate(json.dumps(valid_obj()))
    assert errors == [] and parsed is not None
    assert set(schema.TOP_KEYS) <= set(parsed)


def test_vocabulary_has_the_six_v2_tags():
    assert set(schema.TAGS) == {"DEFINI", "TESTABLE", "PLAUSIBLE", "SPECULATIF",
                                "PROBABLEMENT_FAUX", "NON_FALSIFIABLE"}


def test_text_outside_json_rejected():
    parsed, errors = schema.validate("Voici : " + json.dumps(valid_obj()))
    assert parsed is None and "hors JSON" in errors[0]


@pytest.mark.parametrize("key", schema.TOP_KEYS)
def test_each_missing_top_level_key_rejected(key):
    parsed, errors = check(lambda o: o.pop(key))
    assert parsed is None and errors == [f"champ manquant : {key}"]


def test_tag_outside_vocabulary_rejected():
    parsed, errors = check(lambda o: o["claim_tagging_summary"]["claims"][0].update(tag="CERTAIN"))
    assert parsed is None and any("tag hors vocabulaire" in e and "CERTAIN" in e for e in errors)


def test_no_refutation_test_rejected():
    def drop(o):
        for c in o["claim_tagging_summary"]["claims"]:
            c.pop("refutation_test", None)
            c["justification"] = "ok"
    parsed, errors = check(drop)
    assert parsed is None and "aucun test de réfutation" in errors


def test_claim_without_justification_nor_refutation_rejected():
    parsed, errors = check(lambda o: o["claim_tagging_summary"]["claims"][1].pop("justification"))
    assert parsed is None and any("sans justification" in e for e in errors)


def test_no_baseline_rejected():
    parsed, errors = check(lambda o: o["experimental_plan"].update(baselines=[]))
    assert parsed is None and "aucune baseline" in errors


def test_sdm_without_justification_rejected():
    parsed, errors = check(lambda o: o["anti_mystical_audit"].update(justification=""))
    assert parsed is None and "SDM sans justification" in errors


def test_sif_without_justification_rejected():
    parsed, errors = check(lambda o: o["fruitful_intuitions"][0].pop("justification"))
    assert parsed is None and any("SIF sans justification" in e for e in errors)


@pytest.mark.parametrize("n", [2, 8])
def test_intuition_count_must_be_3_to_7(n):
    parsed, errors = check(lambda o: o.update(
        fruitful_intuitions=[{"text": "t", "sif": 50, "justification": "j"}] * n))
    assert parsed is None and any("3 à 7" in e for e in errors)


def test_sdm_out_of_range_rejected():
    parsed, errors = check(lambda o: o["anti_mystical_audit"].update(sdm=101))
    assert parsed is None and errors
