"""
cg/scene_object.py — SceneObject: Mesh + Transform State
==========================================================
CG Topic: Model Matrix & TRS Transform Composition

A SceneObject wraps a Mesh with position, rotation, and scale, and
computes the 4×4 model matrix that transforms the mesh from object
(local) space into world space.

Model matrix (TRS order):
    M = T(position) @ R(rotation) @ S(scale)

Applied right-to-left:
  1. S: Scale the mesh around its local origin.
  2. R: Rotate the scaled mesh around its local origin.
  3. T: Translate it to its world-space position.

Changing the order (to SRT) demonstrates why matrix multiplication
order matters — the result is very different.
"""

import numpy as np
from cg.mesh import Mesh
from cg import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALE_MIN: float = 0.1   # prevent degenerate / inside-out geometry
SCALE_MAX: float = 5.0


class SceneObject:
    """
    A renderable object in the scene.

    Attributes
    ----------
    mesh             : the geometry (Mesh dataclass)
    position         : (3,) array — world-space translation
    rotation         : (3,) array — Euler angles in RADIANS (X, Y, Z)
    scale            : (3,) array — scale factors (clamped to [0.1, 5.0])
    transform_order  : "TRS" (standard) or "SRT" (demonstration mode)
    reflection_plane : "none" | "XY" | "YZ" | "XZ" — active reflection plane
    """

    def __init__(self, mesh: Mesh) -> None:
        self.mesh: Mesh = mesh
        self.position: np.ndarray = np.zeros(3, dtype=np.float64)
        self.rotation: np.ndarray = np.zeros(3, dtype=np.float64)  # radians
        self.scale: np.ndarray = np.ones(3, dtype=np.float64)
        self.transform_order: str = "TRS"
        self.reflection_plane: str = "none"   # no reflection by default

    # ------------------------------------------------------------------
    # Model matrix
    # ------------------------------------------------------------------

    def model_matrix(self) -> np.ndarray:
        """
        Compute and return the 4×4 model matrix.

        Combines translation, rotation and scale into a single matrix that
        moves the object from local (object) space to world space.

        When a reflection plane is active the reflection matrix is PRE-multiplied
        (applied first, in object space) so the object is mirrored around the
        chosen coordinate plane before any other transform is applied:

            M = compose(T, R, S) @ Reflect(plane)   [reflection_plane != "none"]
            M = compose(T, R, S)                     [reflection_plane == "none"]

        Pre-multiplying keeps the reflection axis fixed to the object's LOCAL
        origin, which is the most intuitive result for a student demo.
        """
        T = transforms.translate(*self.position)
        R = transforms.rotate_xyz(*self.rotation)
        S = transforms.scale(*self.scale)
        base = transforms.compose(self.transform_order, T, R, S)
        if self.reflection_plane != "none":
            RF = transforms.reflect(self.reflection_plane)
            return base @ RF
        return base

    # ------------------------------------------------------------------
    # Transform helpers (called by input_handler)
    # ------------------------------------------------------------------

    def translate_by(self, dx: float, dy: float, dz: float) -> None:
        """Move the object by a delta in world-space X, Y, Z."""
        self.position += np.array([dx, dy, dz], dtype=np.float64)

    def rotate_by(self, dax: float, day: float, daz: float) -> None:
        """Add delta Euler angles (in radians) to the object's rotation."""
        self.rotation += np.array([dax, day, daz], dtype=np.float64)

    def scale_by(self, ds: float) -> None:
        """
        Apply a uniform scale delta (adds ds to each scale component).
        Clamped to [SCALE_MIN, SCALE_MAX] to keep geometry sane.
        """
        self.scale = np.clip(self.scale + ds, SCALE_MIN, SCALE_MAX)

    # ------------------------------------------------------------------
    # Exact-value setters — used by the user-input panels
    # ------------------------------------------------------------------

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set the object's world-space position to an exact (x, y, z)."""
        self.position = np.array([x, y, z], dtype=np.float64)

    def set_rotation_deg(self, rx: float, ry: float, rz: float) -> None:
        """Set the object's rotation from Euler angles given in DEGREES."""
        self.rotation = np.radians(np.array([rx, ry, rz], dtype=np.float64))

    def set_scale(self, sx: float, sy: float, sz: float) -> None:
        """Set a non-uniform scale, clamped to [SCALE_MIN, SCALE_MAX] per axis."""
        self.scale = np.clip(
            np.array([sx, sy, sz], dtype=np.float64), SCALE_MIN, SCALE_MAX
        )

    def apply_rotation_deg(self, rx: float, ry: float, rz: float) -> None:
        """Add Euler-angle offsets given in DEGREES to the current rotation."""
        self.rotation += np.radians(np.array([rx, ry, rz], dtype=np.float64))

    def reset(self) -> None:
        """Restore object to identity transform (centred, no rotation, unit scale, no reflection)."""
        self.position = np.zeros(3, dtype=np.float64)
        self.rotation = np.zeros(3, dtype=np.float64)
        self.scale = np.ones(3, dtype=np.float64)
        self.reflection_plane = "none"

    def toggle_transform_order(self) -> None:
        """Switch between TRS and SRT to demonstrate matrix-order dependence."""
        self.transform_order = "SRT" if self.transform_order == "TRS" else "TRS"

    def cycle_reflection(self) -> None:
        """
        Cycle the active reflection plane through: none → XY → YZ → XZ → none …

        Each call moves to the next plane in the cycle.  When the plane is
        "none" the reflection matrix is the identity — no visible change.

        CG concept: a reflection matrix has determinant = -1, which FLIPS face
        winding (CCW → CW). The renderer detects this via the ``reflection_plane``
        attribute and reverses the back-face culling sign accordingly so the
        geometry looks correct in solid mode.
        """
        planes = transforms.REFLECTION_PLANES   # ["none", "XY", "YZ", "XZ"]
        current_idx = planes.index(self.reflection_plane)
        self.reflection_plane = planes[(current_idx + 1) % len(planes)]

    # ------------------------------------------------------------------
    # Convenience properties (used by HUD for display)
    # ------------------------------------------------------------------

    @property
    def rotation_deg(self) -> np.ndarray:
        """Rotation as a (3,) array in DEGREES for display."""
        return np.degrees(self.rotation)
