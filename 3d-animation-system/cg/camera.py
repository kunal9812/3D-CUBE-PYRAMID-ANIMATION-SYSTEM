"""
cg/camera.py — Orbit Camera
============================
CG Topic: Viewing Transform & Projection

The camera orbits around the world origin (always looking at the origin).
State is stored as spherical coordinates: yaw (horizontal angle), pitch
(vertical angle), and distance from origin.

View-matrix construction:
    V = Rx(pitch) @ Ry(yaw) @ T(0, 0, -distance)

Applied right-to-left:
  1. T(0, 0, -distance) — place the camera `distance` units along +Z in
     view space (so the origin is `distance` units in front of the camera).
  2. Ry(yaw)  — rotate around Y to set horizontal angle.
  3. Rx(pitch)— rotate around X to set vertical angle (tilt up/down).

This is an orbit camera (also called a "turntable" camera). The camera
always looks at the world origin.

Projection:
  - Perspective: uses transforms.perspective().  w = -z creates foreshortening.
  - Orthographic: uses transforms.orthographic(). No divide by w, so depth has no
    visual effect on object size.
"""

import math
import numpy as np
from cg import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_YAW: float = math.radians(30.0)          # initial horizontal angle (rad)
DEFAULT_PITCH: float = math.radians(20.0)         # initial vertical angle (rad)
DEFAULT_DISTANCE: float = 6.0                     # distance from origin
DEFAULT_FOV_DEG: float = 60.0                     # vertical field-of-view (degrees)
DEFAULT_NEAR: float = 0.1
DEFAULT_FAR: float = 100.0

PITCH_LIMIT: float = math.radians(89.0)           # prevent gimbal flip at ±90°

# Camera presets — (yaw_deg, pitch_deg, distance)
PRESETS = {
    "front":      (0.0,   0.0,   6.0),
    "top":        (0.0,   90.0,  6.0),
    "side":       (90.0,  0.0,   6.0),
    "isometric":  (45.0,  35.264, 8.0),  # classic isometric angle (arctan(1/√2) ≈ 35.264°)
}


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class Camera:
    """
    Orbit camera that always looks at the world origin.

    Attributes
    ----------
    yaw      : horizontal rotation angle in radians
    pitch    : vertical tilt angle in radians (clamped to ±89°)
    distance : how far the camera is from the origin
    fov      : vertical field-of-view in DEGREES (converted internally)
    near/far : clipping plane distances
    projection_mode : "perspective" or "orthographic"
    """

    def __init__(self) -> None:
        self.yaw: float = DEFAULT_YAW
        self.pitch: float = DEFAULT_PITCH
        self.distance: float = DEFAULT_DISTANCE
        self.fov: float = DEFAULT_FOV_DEG          # stored in degrees; UI shows degrees
        self.near: float = DEFAULT_NEAR
        self.far: float = DEFAULT_FAR
        self.projection_mode: str = "perspective"

    # ------------------------------------------------------------------
    # View matrix
    # ------------------------------------------------------------------

    def view_matrix(self) -> np.ndarray:
        """
        Build the 4×4 view matrix for the current orbit state.

        Uses transforms.look_at() with the camera eye position derived from
        the spherical coordinates (yaw, pitch, distance).

        Eye position in Cartesian (right-handed, camera looks at origin):
            eye.x =  distance * sin(yaw) * cos(pitch)
            eye.y =  distance * sin(pitch)
            eye.z =  distance * cos(yaw) * cos(pitch)

        Sign conventions:
          - Positive yaw rotates the camera to the right (world rotates left).
          - Positive pitch tilts the camera upward (world tilts down).
        """
        cp = math.cos(self.pitch)
        eye = np.array([
            self.distance * math.sin(self.yaw)  * cp,
            self.distance * math.sin(self.pitch),
            self.distance * math.cos(self.yaw)  * cp,
        ])
        return transforms.look_at(eye, target=np.zeros(3), up=np.array([0.0, 1.0, 0.0]))

    # ------------------------------------------------------------------
    # Projection matrix
    # ------------------------------------------------------------------

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """
        Return the projection matrix for the given viewport aspect ratio.

        aspect = viewport_width / viewport_height

        For orthographic mode, the frustum half-height is computed so the object
        appears a similar size to the perspective view at the current distance.
        The half-width scales by aspect.
        """
        if self.projection_mode == "perspective":
            return transforms.perspective(
                fov_y=math.radians(self.fov),
                aspect=aspect,
                near=self.near,
                far=self.far,
            )
        else:  # "orthographic"
            # Scale the ortho box to approximate perspective at current distance.
            # h = distance * tan(fov/2) gives the half-height that would be
            # visible at the focus distance in perspective mode.
            h = self.distance * math.tan(math.radians(self.fov) / 2.0)
            w = h * aspect
            return transforms.orthographic(
                left=-w, right=w,
                bottom=-h, top=h,
                near=self.near, far=self.far,
            )

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------

    def orbit(self, dx: float, dy: float) -> None:
        """
        Orbit the camera by mouse drag deltas.

        dx > 0 → drag right → camera yaw increases (world rotates left).
        dy > 0 → drag down  → camera pitch decreases (world tilts up).

        Pitch is clamped to ±89° to prevent the camera flipping over the poles.
        """
        self.yaw += dx
        self.pitch -= dy
        self.pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, self.pitch))

    def zoom(self, delta: float) -> None:
        """
        Zoom by changing distance. delta < 0 zooms in (moves camera closer).
        Distance is clamped to [0.5, 50] to prevent clipping or excessive zoom.
        """
        self.distance -= delta
        self.distance = max(0.5, min(50.0, self.distance))

    def set_preset(self, name: str) -> None:
        """
        Jump to a named camera preset.

        Available presets: "front", "top", "side", "isometric"
        """
        if name not in PRESETS:
            raise ValueError(f"Unknown camera preset: {name!r}")
        yaw_deg, pitch_deg, dist = PRESETS[name]
        self.yaw = math.radians(yaw_deg)
        self.pitch = math.radians(pitch_deg)
        self.distance = dist

    def reset(self) -> None:
        """Restore the camera to its default orbit position."""
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.distance = DEFAULT_DISTANCE
        self.fov = DEFAULT_FOV_DEG
        self.projection_mode = "perspective"

    def toggle_projection(self) -> None:
        """Switch between perspective and orthographic projection."""
        if self.projection_mode == "perspective":
            self.projection_mode = "orthographic"
        else:
            self.projection_mode = "perspective"

    def adjust_fov(self, delta_deg: float) -> None:
        """Increase or decrease the field of view. Clamped to [10°, 120°]."""
        self.fov = max(10.0, min(120.0, self.fov + delta_deg))

    # ------------------------------------------------------------------
    # Properties for HUD display
    # ------------------------------------------------------------------

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self.pitch)
