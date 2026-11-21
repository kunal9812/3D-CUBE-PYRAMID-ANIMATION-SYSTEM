"""
cg/mesh.py — Mesh Data Structure
=================================
CG Topic: 3D Geometry Representation

A Mesh stores a 3D object as:
  - vertices  : (N, 4) array of homogeneous coordinates [x, y, z, 1]
  - edges     : list of (i, j) pairs — used for wireframe rendering
  - faces     : list of vertex-index tuples — used for solid rendering
  - face_colors: optional list of (R, G, B) base colours, one per face

Face winding convention:
  Vertices in each face tuple are listed in COUNTER-CLOCKWISE order when the
  face is viewed from OUTSIDE the object. This consistent winding is what lets
  back-face culling decide which faces point toward the camera.

Why homogeneous coordinates?
  Every vertex carries a 4th component w = 1. This lets us apply any 4×4
  transform matrix (including translations) with a single matrix multiply.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


# A face is a tuple of at least 3 vertex indices.
# Quads use 4 indices, triangles use 3.
Face = Tuple[int, ...]
Edge = Tuple[int, int]
Color = Tuple[int, int, int]  # RGB, each 0–255


@dataclass
class Mesh:
    """
    Holds the geometry of a 3D object in object (local) space.

    Attributes
    ----------
    vertices   : np.ndarray, shape (N, 4), dtype float64
                 Each row is a homogeneous point [x, y, z, 1].
    edges      : list of (i, j) — indices into vertices, used for wireframe drawing.
    faces      : list of vertex-index tuples (CCW winding from outside).
    face_colors: optional list of (R, G, B) base colours, one per face.
                 If None, a default colour is used by the renderer.
    """

    vertices: np.ndarray           # shape (N, 4)
    edges: List[Edge]
    faces: List[Face]
    face_colors: Optional[List[Color]] = field(default=None)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def face_normal(self, face_idx: int) -> np.ndarray:
        """
        Compute the outward-facing normal for a face using the cross product.

        For face vertices A, B, C (in CCW order from outside):
            edge1 = B - A
            edge2 = C - A
            normal = normalize(edge1 × edge2)

        The cross product of two edges in CCW winding points OUTWARD (away from
        the interior), by the right-hand rule.

        Returns a (3,) unit vector.
        """
        face = self.faces[face_idx]
        A = self.vertices[face[0], :3]
        B = self.vertices[face[1], :3]
        C = self.vertices[face[2], :3]
        edge1 = B - A
        edge2 = C - A
        normal = np.cross(edge1, edge2)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            return np.array([0.0, 0.0, 1.0])  # degenerate face fallback
        return normal / length

    def centroid(self) -> np.ndarray:
        """Return the average position of all vertices as a (3,) array."""
        return self.vertices[:, :3].mean(axis=0)
