"""
tests/test_meshes.py — Unit tests for cg/mesh.py and cg/shapes.py

Run with: pytest -q tests/test_meshes.py
"""

import numpy as np
import pytest
from cg.mesh import Mesh
from cg.shapes import make_cube, make_pyramid


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

def test_cube_vertex_count():
    cube = make_cube()
    assert cube.num_vertices == 8, f"Cube should have 8 vertices, got {cube.num_vertices}"


def test_cube_edge_count():
    cube = make_cube()
    assert cube.num_edges == 12, f"Cube should have 12 edges, got {cube.num_edges}"


def test_cube_face_count():
    cube = make_cube()
    assert cube.num_faces == 6, f"Cube should have 6 faces, got {cube.num_faces}"


# ---------------------------------------------------------------------------
# Pyramid geometry
# ---------------------------------------------------------------------------

def test_pyramid_vertex_count():
    pyr = make_pyramid()
    assert pyr.num_vertices == 5, f"Pyramid should have 5 vertices, got {pyr.num_vertices}"


def test_pyramid_edge_count():
    pyr = make_pyramid()
    assert pyr.num_edges == 8, f"Pyramid should have 8 edges, got {pyr.num_edges}"


def test_pyramid_face_count():
    pyr = make_pyramid()
    assert pyr.num_faces == 5, f"Pyramid should have 5 faces, got {pyr.num_faces}"


# ---------------------------------------------------------------------------
# Homogeneous coordinate: all w == 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_fn", [make_cube, make_pyramid])
def test_all_w_equals_one(shape_fn):
    mesh = shape_fn()
    w_values = mesh.vertices[:, 3]
    assert np.all(w_values == 1.0), (
        f"All vertices must have w=1, found: {w_values}"
    )


@pytest.mark.parametrize("shape_fn", [make_cube, make_pyramid])
def test_vertices_shape(shape_fn):
    mesh = shape_fn()
    assert mesh.vertices.shape[1] == 4, "Vertex array must have 4 columns (x, y, z, w)"
    assert mesh.vertices.dtype == np.float64, "Vertices must be float64"


# ---------------------------------------------------------------------------
# Face normals point AWAY from the mesh centroid
# (verifies CCW winding is consistent)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_fn", [make_cube, make_pyramid])
def test_face_normals_point_outward(shape_fn):
    mesh = shape_fn()
    centroid = mesh.centroid()

    for i in range(mesh.num_faces):
        normal = mesh.face_normal(i)                  # computed from CCW winding

        # A representative point on the face (average of its vertices)
        face = mesh.faces[i]
        face_center = mesh.vertices[face, :3].mean(axis=0)

        # Vector from centroid to face center — should point in the same
        # general direction as the outward normal (positive dot product)
        outward = face_center - centroid
        dot = np.dot(normal, outward)

        assert dot > 0, (
            f"Face {i} normal points INWARD (dot={dot:.4f}). "
            f"Check winding order for face {face}."
        )


# ---------------------------------------------------------------------------
# Vertex index sanity: every face index is within bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_fn", [make_cube, make_pyramid])
def test_face_indices_in_bounds(shape_fn):
    mesh = shape_fn()
    n = mesh.num_vertices
    for face in mesh.faces:
        for idx in face:
            assert 0 <= idx < n, f"Face index {idx} out of range [0, {n})"


@pytest.mark.parametrize("shape_fn", [make_cube, make_pyramid])
def test_edge_indices_in_bounds(shape_fn):
    mesh = shape_fn()
    n = mesh.num_vertices
    for i, j in mesh.edges:
        assert 0 <= i < n and 0 <= j < n, f"Edge ({i},{j}) out of range"


# ---------------------------------------------------------------------------
# face_colors: count matches face count when provided
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_fn", [make_cube, make_pyramid])
def test_face_colors_match_face_count(shape_fn):
    mesh = shape_fn()
    if mesh.face_colors is not None:
        assert len(mesh.face_colors) == mesh.num_faces, (
            "face_colors list length must match number of faces"
        )
