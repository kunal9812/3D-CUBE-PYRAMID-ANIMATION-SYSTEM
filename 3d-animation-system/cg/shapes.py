"""
cg/shapes.py — Primitive 3D Shape Factories
=============================================
CG Topic: 3D Geometry — vertices, edges, faces, winding order

Each factory function returns a Mesh object with vertices in homogeneous
coordinates (w = 1), edges for wireframe drawing, and faces with consistent
counter-clockwise winding (viewed from outside) for back-face culling.

Coordinate system:
  - Objects are centred at the origin.
  - Cube side length = 2 (coordinates ±1 on each axis).
  - Pyramid base sits on y = -1, apex at y = +1.
"""

import numpy as np
from cg.mesh import Mesh


# ---------------------------------------------------------------------------
# Colour palette (one colour per face, for flat shading)
# ---------------------------------------------------------------------------

_CUBE_COLORS = [
    (220,  60,  60),   # +X face — red
    ( 60, 160, 220),   # -X face — blue
    ( 60, 200,  60),   # +Y face — green
    (220, 180,  60),   # -Y face — yellow
    (200,  60, 200),   # +Z face — magenta
    ( 60, 200, 200),   # -Z face — cyan
]

_PYRAMID_COLORS = [
    (200, 160,  60),   # base quad    — gold
    (220,  60,  60),   # front tri    — red
    ( 60, 160, 220),   # right tri    — blue
    ( 60, 200,  60),   # back tri     — green
    (200,  60, 200),   # left tri     — magenta
]


# ---------------------------------------------------------------------------
# Cube
# ---------------------------------------------------------------------------

def make_cube(
    center: tuple = (0.0, 0.0, 0.0),
    size: tuple = (2.0, 2.0, 2.0),
    custom_vertices: np.ndarray | None = None,
) -> Mesh:
    """
    Create a cube mesh.

    Parameters
    ----------
    center : (cx, cy, cz) — world-space centre of the cube. Default (0,0,0).
    size   : (width, height, depth) — full side lengths along X, Y, Z.
             Default (2, 2, 2) reproduces the original ±1 unit cube.
    custom_vertices : optional (8, 4) or (8, 3) numpy array.
             When supplied, ``center`` and ``size`` are ignored and these
             exact vertex positions are used instead.  A 4th column of
             ones (homogeneous w=1) is added automatically if only 3
             columns are given.

    Vertex numbering (CCW winding from outside, right-hand system):

           6 -------- 7
          /|          /|
         / |         / |
        4 -------- 5   |
        |  2 ------|- 3       Y
        | /        | /        |
        |/         |/         +--- X
        0 -------- 1         /
                             Z

    Default vertices (center=origin, size=2×2×2):
        0: (-1, -1, +1)   front-bottom-left
        1: (+1, -1, +1)   front-bottom-right
        2: (-1, -1, -1)   back-bottom-left
        3: (+1, -1, -1)   back-bottom-right
        4: (-1, +1, +1)   front-top-left
        5: (+1, +1, +1)   front-top-right
        6: (-1, +1, -1)   back-top-left
        7: (+1, +1, -1)   back-top-right

    Faces (6 quads, CCW winding from outside):
        +X: 1, 3, 7, 5
        -X: 0, 4, 6, 2
        +Y: 4, 5, 7, 6
        -Y: 0, 2, 3, 1
        +Z: 0, 1, 5, 4
        -Z: 2, 6, 7, 3
    """
    if custom_vertices is not None:
        # Use exact user-supplied coordinates
        cv = np.asarray(custom_vertices, dtype=np.float64)
        if cv.shape == (8, 3):
            # Add homogeneous column
            ones = np.ones((8, 1), dtype=np.float64)
            cv = np.hstack([cv, ones])
        if cv.shape != (8, 4):
            raise ValueError(
                f"custom_vertices must be shape (8, 3) or (8, 4), got {cv.shape}"
            )
        verts = cv
    else:
        # Generate vertices from center + size
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        hw = float(size[0]) / 2.0   # half-width  (X)
        hh = float(size[1]) / 2.0   # half-height (Y)
        hd = float(size[2]) / 2.0   # half-depth  (Z)
        verts = np.array([
            [cx - hw, cy - hh, cz + hd, 1],   # 0 front-bottom-left
            [cx + hw, cy - hh, cz + hd, 1],   # 1 front-bottom-right
            [cx - hw, cy - hh, cz - hd, 1],   # 2 back-bottom-left
            [cx + hw, cy - hh, cz - hd, 1],   # 3 back-bottom-right
            [cx - hw, cy + hh, cz + hd, 1],   # 4 front-top-left
            [cx + hw, cy + hh, cz + hd, 1],   # 5 front-top-right
            [cx - hw, cy + hh, cz - hd, 1],   # 6 back-top-left
            [cx + hw, cy + hh, cz - hd, 1],   # 7 back-top-right
        ], dtype=np.float64)

    # 12 edges (each unique pair of adjacent vertices)
    edges = [
        # bottom ring
        (0, 1), (1, 3), (3, 2), (2, 0),
        # top ring
        (4, 5), (5, 7), (7, 6), (6, 4),
        # vertical pillars
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # 6 quad faces — CCW winding when viewed from outside
    faces = [
        (1, 3, 7, 5),   # +X  (right face)
        (0, 4, 6, 2),   # -X  (left face)
        (4, 5, 7, 6),   # +Y  (top face)
        (0, 2, 3, 1),   # -Y  (bottom face)
        (0, 1, 5, 4),   # +Z  (front face)
        (2, 6, 7, 3),   # -Z  (back face)
    ]

    return Mesh(
        vertices=verts,
        edges=edges,
        faces=faces,
        face_colors=list(_CUBE_COLORS),
    )


# ---------------------------------------------------------------------------
# Pyramid
# ---------------------------------------------------------------------------

def make_pyramid(
    center: tuple = (0.0, 0.0, 0.0),
    base_size: tuple = (2.0, 2.0),
    height: float = 2.0,
    custom_vertices: np.ndarray | None = None,
) -> Mesh:
    """
    Create a square-base pyramid mesh.

    Parameters
    ----------
    center      : (cx, cy, cz) — world-space centre of the base square.
                  Default (0,0,0) → base sits on y = -1, apex at y = +1.
    base_size   : (base_width, base_depth) — full side lengths of the base
                  along X and Z.  Default (2, 2) reproduces the original ±1 base.
    height      : full height of the pyramid (base centre to apex).
                  Default 2.0 → apex at y = centre_y + height.
    custom_vertices : optional (5, 4) or (5, 3) numpy array.
                  When supplied, all other parameters are ignored and these
                  exact vertex positions are used.  Row order:
                    0 = front-left base, 1 = front-right base,
                    2 = back-right base, 3 = back-left base, 4 = apex.

    Default vertices (center=origin, base_size=2×2, height=2):
        0: (-1, -1, +1)   front-left  base corner
        1: (+1, -1, +1)   front-right base corner
        2: (+1, -1, -1)   back-right  base corner
        3: (-1, -1, -1)   back-left   base corner
        4: ( 0, +1,  0)   apex

    Faces (1 quad base + 4 triangles, CCW winding from outside):
        base  : 0, 3, 2, 1    (CCW when viewed from below — looking up)
        front : 0, 1, 4
        right : 1, 2, 4
        back  : 2, 3, 4
        left  : 3, 0, 4
    """
    if custom_vertices is not None:
        cv = np.asarray(custom_vertices, dtype=np.float64)
        if cv.shape == (5, 3):
            ones = np.ones((5, 1), dtype=np.float64)
            cv = np.hstack([cv, ones])
        if cv.shape != (5, 4):
            raise ValueError(
                f"custom_vertices must be shape (5, 3) or (5, 4), got {cv.shape}"
            )
        verts = cv
    else:
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        hw = float(base_size[0]) / 2.0   # half base-width  (X)
        hd = float(base_size[1]) / 2.0   # half base-depth  (Z)
        base_y = cy - float(height) / 2.0
        apex_y = cy + float(height) / 2.0
        verts = np.array([
            [cx - hw, base_y, cz + hd, 1],   # 0 front-left  base
            [cx + hw, base_y, cz + hd, 1],   # 1 front-right base
            [cx + hw, base_y, cz - hd, 1],   # 2 back-right  base
            [cx - hw, base_y, cz - hd, 1],   # 3 back-left   base
            [cx,      apex_y, cz,       1],   # 4 apex
        ], dtype=np.float64)

    # 8 edges
    edges = [
        # base square
        (0, 1), (1, 2), (2, 3), (3, 0),
        # slant edges to apex
        (0, 4), (1, 4), (2, 4), (3, 4),
    ]

    # 5 faces — CCW winding from outside
    faces = [
        (0, 3, 2, 1),   # base quad (CCW viewed from below: -Y direction)
        (0, 1, 4),      # front triangle
        (1, 2, 4),      # right triangle
        (2, 3, 4),      # back triangle
        (3, 0, 4),      # left triangle
    ]

    return Mesh(
        vertices=verts,
        edges=edges,
        faces=faces,
        face_colors=list(_PYRAMID_COLORS),
    )
