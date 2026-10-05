import numpy as np
import pytest

from experiments.continuous_vs_vision.baselines import (
    one_hot,
    random_circle_permutation,
    rasterize,
)
from experiments.continuous_vs_vision.run_experiment import _holm, run
from experiments.continuous_vs_vision.trajectory import (
    chen_product,
    path_signature,
    trajectory_dft,
    word_to_trajectory,
)


def test_word_trajectory_uses_roots_and_linear_samples():
    path = word_to_trajectory("ab", "abcd", samples_per_segment=2)
    assert np.allclose(path, [1, (1 + 1j) / 2, 1j])
    assert np.allclose(word_to_trajectory("ac", "abcd", n=4), [1, -1])


def test_word_trajectory_rejects_ambiguous_input():
    with pytest.raises(ValueError):
        word_to_trajectory("ax", "abcd")
    with pytest.raises(ValueError):
        word_to_trajectory("a", "abcd", n=3)


def test_truncated_signature_obeys_chen_concatenation():
    path = np.asarray([0 + 0j, 1 + 0j, 1 + 2j, -1 + 1j])
    left = path_signature(path[:3])
    right = path_signature(path[2:])
    combined = chen_product(left, right)
    expected = path_signature(path)
    assert np.allclose(combined[0], expected[0])
    assert np.allclose(combined[1], expected[1])


def test_dft_and_baselines_have_expected_shapes():
    assert trajectory_dft([1, 1j]).shape == (2,)
    assert one_hot("ab", "abc").shape == (2, 3)
    first = random_circle_permutation("abcd", 7)
    assert first == random_circle_permutation("abcd", 7)
    assert sorted(first.values()) == list(range(4))
    assert rasterize("ab", "abcd", size=12).shape == (12, 12)


def test_holm_correction_is_monotone_and_bounded():
    adjusted = _holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert all(0 <= p <= 1 for p in adjusted.values())


def test_experiment_falls_back_to_declared_toy_corpus(tmp_path):
    result = run(
        max_words=20,
        permutations=5,
        raster_size=8,
        data_dir=tmp_path,
    )
    assert result["toy_corpus"] is True
    assert "jouet" in result["corpus_source"]
    assert len({item["parameter_count"] for item in result["metrics"].values()}) == 1
    assert len({
        item["inference_flops_per_example"] for item in result["metrics"].values()
    }) == 1
    assert set(result["metrics"]) == {
        "one_hot", "circle", "permuted_circle", "raster", "random_pattern"
    }
