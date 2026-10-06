import math

import numpy as np
import pytest

import circle_trace as ct


def test_angles_equidistant():
    a = ct.letter_angles(26)
    assert a[0] == 0
    assert np.allclose(np.diff(a), 2 * math.pi / 26)
    assert np.isclose(a[13], math.pi)


def test_positions_on_circle_and_equal_chords():
    pos = ct.letter_positions(22)
    assert pos.shape == (22, 2)
    assert np.allclose(np.hypot(pos[:, 0], pos[:, 1]), 1.0)
    chords = np.hypot(*(np.roll(pos, -1, axis=0) - pos).T)
    assert np.allclose(chords, chords[0])


def test_n_rays():
    rays = ct.ray_segments(28)
    assert rays.shape == (28, 2, 2)
    assert np.allclose(rays[:, 0], 0)


def test_invalid_n():
    with pytest.raises(ValueError):
        ct.letter_angles(0)


def test_trajectory_ignores_unknown_and_segments():
    letters = list("abcd")
    assert ct.trajectory_indices(list("abxd"), letters) == [0, 1, 3]
    pts = ct.trajectory_points(list("abc"), letters)
    assert pts.shape == (3, 2)
    assert ct.segments_of(pts).shape == (2, 2, 2)
    assert ct.segments_of(pts[:1]).shape == (0, 2, 2)


def test_prefix_frames():
    assert ct.prefix_units(list("abc")) == [["a"], ["a", "b"], ["a", "b", "c"]]
    frames = ct.trace_frames(list("abc"), list("abcdef"), size=32)
    assert len(frames) == 3
    assert [len(f.trajectory) for f in frames] == [1, 2, 3]


def test_trace_to_array_shapes_and_values():
    letters = list("abcdef")
    ta = ct.trace_to_array(list("abca"), letters, size=64, values=[1, 2, 3, 4, 5, 6])
    assert ta.image.shape == (64, 64) and ta.image.dtype == np.float32
    assert ta.image.max() == 1.0 and ta.image.min() == 0.0
    assert ta.gematria_image is not None and 0 < ta.gematria_image.max() <= 1.0
    assert list(ta.indices) == [0, 1, 2, 0]
    assert ct.trace_to_array([], letters, 16).image.sum() == 0


def test_gif_smoke(tmp_path):
    from PIL import Image
    letters = list("abcdefghij")
    panel = ct.Panel("t", letters, list("abc"), values=list(range(1, 11)))
    out = tmp_path / "x.gif"
    n = ct.render_gif([panel], out, fps=4, color_by="gematria", size_px=160)
    assert n == 3 and Image.open(out).n_frames == 3
