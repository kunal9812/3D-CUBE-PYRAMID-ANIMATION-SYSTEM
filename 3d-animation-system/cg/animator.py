"""
cg/animator.py — Time-Based Animations
========================================
CG Topic: Demonstrating All Three Transformations Over Time

Three independent animations, each toggle-able:
  - AutoRotate  : constant angular velocity (demonstrates ROTATION)
  - Bounce      : sinusoidal translation along Y (demonstrates TRANSLATION)
  - ScalePulse  : sinusoidal uniform scale between 0.7 and 1.3 (demonstrates SCALING)

Key design: dt (seconds since last frame) is used instead of frame counts,
so animation speed is hardware-independent.

Each animation adds an OFFSET to the object's base transform — it does not
overwrite the user's manual transforms. This lets the user continue to
move/rotate the object while animations run.
"""

import math
import numpy as np
from cg.scene_object import SceneObject

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTO_ROTATE_SPEED: float = math.radians(45.0)  # 45°/s about Y axis
BOUNCE_AMPLITUDE: float = 0.8                  # ±0.8 units along Y
BOUNCE_FREQUENCY: float = 1.0                  # 1 cycle per second
SCALE_AMPLITUDE: float = 0.3                   # scale oscillates by ±0.3 (0.7 to 1.3)
SCALE_FREQUENCY: float = 0.8                   # 0.8 cycles per second


# ---------------------------------------------------------------------------
# Individual animation classes
# ---------------------------------------------------------------------------

class AutoRotate:
    """
    Rotates the object continuously about the Y axis.

    Demonstrates: ROTATION transformation.
    Uses dt so 1 second always equals AUTO_ROTATE_SPEED radians regardless of FPS.
    """

    def __init__(self) -> None:
        self.enabled: bool = True
        self.speed: float = AUTO_ROTATE_SPEED  # radians per second

    def update(self, obj: SceneObject, dt: float) -> None:
        """Add angular delta to the object's Y rotation (modifies base rotation)."""
        if self.enabled:
            obj.rotation[1] += self.speed * dt   # rotate_y: Y component of Euler angles


class Bounce:
    """
    Moves the object up and down in a sine wave along the Y axis.

    Demonstrates: TRANSLATION transformation.
    The offset is computed each frame and applied relative to the base position.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._time: float = 0.0              # accumulated time for sin phase

    def update(self, obj: SceneObject, dt: float) -> None:
        """
        Compute sinusoidal Y offset and add it as a translation offset.

        Because we want to ADD to the base position (not overwrite it), we
        track the previous offset and subtract it before adding the new one.
        """
        if not self.enabled:
            # If we were just disabled, ensure we leave position clean
            return
        prev_offset = BOUNCE_AMPLITUDE * math.sin(
            2 * math.pi * BOUNCE_FREQUENCY * (self._time - dt)
        ) if self._time > 0 else 0.0
        self._time += dt
        new_offset = BOUNCE_AMPLITUDE * math.sin(
            2 * math.pi * BOUNCE_FREQUENCY * self._time
        )
        # Delta offset: remove previous contribution, add new one
        obj.position[1] += new_offset - prev_offset

    def reset_time(self) -> None:
        self._time = 0.0


class ScalePulse:
    """
    Oscillates the object's uniform scale between (1 - amplitude) and (1 + amplitude).

    Demonstrates: SCALING transformation.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._time: float = 0.0

    def update(self, obj: SceneObject, dt: float) -> None:
        """Compute sinusoidal scale factor and set it (overrides manual scale)."""
        if not self.enabled:
            return
        prev_s = 1.0 + SCALE_AMPLITUDE * math.sin(
            2 * math.pi * SCALE_FREQUENCY * (self._time - dt)
        ) if self._time > 0 else 1.0
        self._time += dt
        new_s = 1.0 + SCALE_AMPLITUDE * math.sin(
            2 * math.pi * SCALE_FREQUENCY * self._time
        )
        # Add delta: remove previous animated contribution, add new
        delta = new_s - prev_s
        obj.scale = np.clip(obj.scale + delta, 0.1, 5.0)

    def reset_time(self) -> None:
        self._time = 0.0


# ---------------------------------------------------------------------------
# Animator — manages all animations
# ---------------------------------------------------------------------------

class Animator:
    """
    Manages all animations for a SceneObject.

    Usage:
        animator = Animator()
        # in game loop:
        animator.update(obj, dt)
    """

    def __init__(self) -> None:
        self.auto_rotate = AutoRotate()
        self.bounce = Bounce()
        self.scale_pulse = ScalePulse()
        self.paused: bool = False

    def update(self, obj: SceneObject, dt: float) -> None:
        """Apply all enabled animations to `obj` using the elapsed time `dt`."""
        if self.paused:
            return
        self.auto_rotate.update(obj, dt)
        self.bounce.update(obj, dt)
        self.scale_pulse.update(obj, dt)

    def pause_resume(self) -> None:
        """Toggle global pause/resume of all animations."""
        self.paused = not self.paused

    def toggle_rotate(self) -> None:
        self.auto_rotate.enabled = not self.auto_rotate.enabled

    def toggle_bounce(self) -> None:
        self.bounce.enabled = not self.bounce.enabled
        if self.bounce.enabled:
            self.bounce.reset_time()

    def toggle_scale_pulse(self) -> None:
        self.scale_pulse.enabled = not self.scale_pulse.enabled
        if self.scale_pulse.enabled:
            self.scale_pulse.reset_time()
