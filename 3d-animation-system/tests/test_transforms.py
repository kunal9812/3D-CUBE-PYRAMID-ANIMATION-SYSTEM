"""
tests/test_transforms.py — Unit tests for cg/transforms.py

Run with: pytest -q tests/test_transforms.py
"""

import math
import numpy as np
import pytest
from cg.transforms import (
    identity, translate, scale,
    rotate_x, rotate_y, rotate_z,
    rotate_xyz, rotate_axis,
    perspective, orthographic, look_at,
    compose,
)

# Tolerance for floating-point comparisons
ATOL = 1e-10


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_identity_is_4x4_eye():
    assert np.allclose(identity(), np.eye(4), atol=ATOL)


# ---------------------------------------------------------------------------
# Rotation: zero angle == identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [rotate_x, rotate_y, rotate_z])
def test_rotation_zero_is_identity(fn):
    assert np.allclose(fn(0.0), identity(), atol=ATOL)


# ---------------------------------------------------------------------------
# Rotation: 2π == identity (one full turn)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [rotate_x, rotate_y, rotate_z])
def test_rotation_full_turn_is_identity(fn):
    assert np.allclose(fn(2 * math.pi), identity(), atol=1e-10)


# ---------------------------------------------------------------------------
# Rotation matrices are orthonormal: R @ Rᵀ == I, det == 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [0.1, math.pi / 4, math.pi / 3, 1.23])
@pytest.mark.parametrize("fn", [rotate_x, rotate_y, rotate_z])
def test_rotation_is_orthonormal(fn, angle):
    R = fn(angle)
    R3 = R[:3, :3]                          # upper-left 3×3 rotation block
    assert np.allclose(R3 @ R3.T, np.eye(3), atol=ATOL)
    assert np.isclose(np.linalg.det(R3), 1.0, atol=ATOL)


# ---------------------------------------------------------------------------
# Rotate (1, 0, 0, 1) by 90° about Z → should give (0, 1, 0, 1)
# ---------------------------------------------------------------------------

def test_rotate_z_90_moves_x_to_y():
    p = np.array([1.0, 0.0, 0.0, 1.0])
    p_rot = rotate_z(math.pi / 2) @ p
    expected = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.allclose(p_rot, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Translate: moves points, leaves direction vectors alone (w=0)
# ---------------------------------------------------------------------------

def test_translate_moves_point():
    T = translate(3.0, -2.0, 5.0)
    p = np.array([1.0, 1.0, 1.0, 1.0])   # point (w=1)
    result = T @ p
    expected = np.array([4.0, -1.0, 6.0, 1.0])
    assert np.allclose(result, expected, atol=ATOL)


def test_translate_does_not_move_direction_vector():
    T = translate(10.0, 20.0, 30.0)
    v = np.array([1.0, 0.0, 0.0, 0.0])   # direction (w=0)
    result = T @ v
    assert np.allclose(result, v, atol=ATOL), "Translation must not affect direction vectors"


def test_translate_and_inverse_is_identity():
    a = 3.0
    T_fwd = translate(a, a, a)
    T_inv = translate(-a, -a, -a)
    assert np.allclose(T_fwd @ T_inv, identity(), atol=ATOL)
    assert np.allclose(T_inv @ T_fwd, identity(), atol=ATOL)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

def test_scale_multiplies_coordinates():
    S = scale(2.0, 3.0, 4.0)
    p = np.array([1.0, 1.0, 1.0, 1.0])
    result = S @ p
    expected = np.array([2.0, 3.0, 4.0, 1.0])
    assert np.allclose(result, expected, atol=ATOL)


def test_scale_identity_is_identity():
    assert np.allclose(scale(1.0, 1.0, 1.0), identity(), atol=ATOL)


# ---------------------------------------------------------------------------
# compose: TRS vs SRT give different results (order matters)
# ---------------------------------------------------------------------------

def test_compose_trs_differs_from_srt():
    T = translate(5.0, 0.0, 0.0)
    R = rotate_y(math.pi / 4)
    S = scale(2.0, 2.0, 2.0)
    trs = compose("TRS", T, R, S)
    srt = compose("SRT", T, R, S)
    assert not np.allclose(trs, srt, atol=1e-6), (
        "TRS and SRT should differ when T, R, S are all non-trivial"
    )


def test_compose_invalid_order_raises():
    T = R = S = identity()
    with pytest.raises(ValueError):
        compose("XYZ", T, R, S)


# ---------------------------------------------------------------------------
# rotate_xyz combines correctly
# ---------------------------------------------------------------------------

def test_rotate_xyz_equals_manual_composition():
    ax, ay, az = 0.3, 0.5, 0.7
    expected = rotate_z(az) @ rotate_y(ay) @ rotate_x(ax)
    result = rotate_xyz(ax, ay, az)
    assert np.allclose(result, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# rotate_axis (Rodrigues): axis-aligned cases match basic rotations
# ---------------------------------------------------------------------------

def test_rotate_axis_x_matches_rotate_x():
    angle = 1.1
    axis = np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotate_axis(axis, angle), rotate_x(angle), atol=1e-10)


def test_rotate_axis_z_matches_rotate_z():
    angle = 0.7
    axis = np.array([0.0, 0.0, 1.0])
    assert np.allclose(rotate_axis(axis, angle), rotate_z(angle), atol=1e-10)


def test_rotate_axis_is_orthonormal():
    axis = np.array([1.0, 1.0, 1.0])
    R = rotate_axis(axis, 0.9)
    R3 = R[:3, :3]
    assert np.allclose(R3 @ R3.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R3), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# look_at: upper 3×3 is orthonormal
# ---------------------------------------------------------------------------

def test_look_at_upper_3x3_orthonormal():
    V = look_at(
        eye=np.array([3.0, 2.0, 5.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    R3 = V[:3, :3]
    assert np.allclose(R3 @ R3.T, np.eye(3), atol=1e-10)


def test_look_at_same_eye_target_raises():
    with pytest.raises(ValueError):
        look_at(
            eye=np.array([1.0, 1.0, 1.0]),
            target=np.array([1.0, 1.0, 1.0]),
            up=np.array([0.0, 1.0, 0.0]),
        )
