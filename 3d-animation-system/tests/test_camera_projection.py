"""
tests/test_camera_projection.py — Unit tests for cg/camera.py projections

Run with: pytest -q tests/test_camera_projection.py
"""

import math
import numpy as np
import pytest
from cg.transforms import perspective, orthographic, look_at, translate, rotate_x, rotate_y


ATOL = 1e-6


# ---------------------------------------------------------------------------
# Helper: apply full perspective pipeline to a single point
# ---------------------------------------------------------------------------

def project_point(point_view: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply projection matrix then perspective divide to a view-space point."""
    clip = P @ point_view
    w = clip[3]
    ndc = clip[:3] / w
    return ndc


# ---------------------------------------------------------------------------
# Perspective: a point on the view axis projects to screen center
# ---------------------------------------------------------------------------

def test_perspective_center_point_maps_to_origin():
    """A point directly ahead of the camera (on -Z axis) should project to NDC (0, 0)."""
    P = perspective(fov_y=math.radians(60), aspect=1.0, near=0.1, far=100.0)
    # A point at (0, 0, -5) is directly on the camera's -Z axis
    point = np.array([0.0, 0.0, -5.0, 1.0])
    ndc = project_point(point, P)
    assert np.isclose(ndc[0], 0.0, atol=ATOL), "Centre point should project to NDC x=0"
    assert np.isclose(ndc[1], 0.0, atol=ATOL), "Centre point should project to NDC y=0"


# ---------------------------------------------------------------------------
# Perspective: near plane maps to NDC z = -1, far plane maps to NDC z = +1
# ---------------------------------------------------------------------------

def test_perspective_near_plane_maps_to_ndc_minus1():
    near, far = 0.5, 50.0
    P = perspective(fov_y=math.radians(60), aspect=1.0, near=near, far=far)
    point = np.array([0.0, 0.0, -near, 1.0])   # exactly on the near plane
    ndc = project_point(point, P)
    assert np.isclose(ndc[2], -1.0, atol=ATOL), f"Near plane should map to NDC z=-1, got {ndc[2]}"


def test_perspective_far_plane_maps_to_ndc_plus1():
    near, far = 0.5, 50.0
    P = perspective(fov_y=math.radians(60), aspect=1.0, near=near, far=far)
    point = np.array([0.0, 0.0, -far, 1.0])    # exactly on the far plane
    ndc = project_point(point, P)
    assert np.isclose(ndc[2], 1.0, atol=ATOL), f"Far plane should map to NDC z=+1, got {ndc[2]}"


# ---------------------------------------------------------------------------
# Perspective foreshortening: farther point → smaller projected x
# ---------------------------------------------------------------------------

def test_perspective_foreshortening():
    """
    A point at (1, 0, -5) should project farther from centre than (1, 0, -10),
    because the nearer point subtends a larger angle (perspective foreshortening).
    """
    P = perspective(fov_y=math.radians(60), aspect=1.0, near=0.1, far=100.0)
    near_pt = np.array([1.0, 0.0, -5.0, 1.0])
    far_pt  = np.array([1.0, 0.0, -10.0, 1.0])
    ndc_near = project_point(near_pt, P)
    ndc_far  = project_point(far_pt,  P)
    assert abs(ndc_near[0]) > abs(ndc_far[0]), (
        f"Nearer point should project farther from centre: "
        f"near_x={ndc_near[0]:.4f}, far_x={ndc_far[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# Orthographic: projected x/y do NOT change with depth
# ---------------------------------------------------------------------------

def test_orthographic_no_foreshortening():
    """
    In orthographic projection, moving a point along Z must not change its x/y NDC.
    """
    P = orthographic(left=-5, right=5, bottom=-5, top=5, near=0.1, far=100.0)
    p_near = np.array([2.0, 1.0, -2.0, 1.0])
    p_far  = np.array([2.0, 1.0, -50.0, 1.0])
    # Orthographic: w stays 1, so clip = P @ p, ndc = clip / 1
    ndc_near = (P @ p_near)[:3]
    ndc_far  = (P @ p_far)[:3]
    assert np.isclose(ndc_near[0], ndc_far[0], atol=ATOL), "Ortho x should be depth-independent"
    assert np.isclose(ndc_near[1], ndc_far[1], atol=ATOL), "Ortho y should be depth-independent"


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
    assert np.allclose(R3 @ R3.T, np.eye(3), atol=1e-10), "look_at upper-3x3 must be orthonormal"


# ---------------------------------------------------------------------------
# Camera view matrix from camera.py
# ---------------------------------------------------------------------------

def test_camera_view_matrix_orthonormal():
    from cg.camera import Camera
    cam = Camera()
    V = cam.view_matrix()
    R3 = V[:3, :3]
    assert np.allclose(R3 @ R3.T, np.eye(3), atol=1e-8), "Camera view matrix must be orthonormal"


def test_camera_projection_perspective():
    from cg.camera import Camera
    cam = Camera()
    cam.projection_mode = "perspective"
    P = cam.projection_matrix(aspect=16/9)
    # The (3,2) element of the perspective matrix is -1, encoding w = -z
    assert np.isclose(P[3, 2], -1.0, atol=1e-10), "Perspective matrix must have P[3,2] = -1"


def test_camera_projection_orthographic():
    from cg.camera import Camera
    cam = Camera()
    cam.projection_mode = "orthographic"
    P = cam.projection_matrix(aspect=1.0)
    # Orthographic matrix has w = 1 always → last row should be [0, 0, 0, 1]
    assert np.allclose(P[3], [0, 0, 0, 1], atol=1e-10), "Ortho matrix must have last row [0,0,0,1]"
