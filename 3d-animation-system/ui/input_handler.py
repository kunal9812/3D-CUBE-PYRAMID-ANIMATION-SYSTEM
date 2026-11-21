"""
ui/input_handler.py — Keyboard and Mouse Input
================================================
Maps keyboard and mouse events to actions on the SceneObject and Camera.

All movement amounts are scaled by `dt` (seconds) so speed is
frame-rate independent. Key bindings are defined as constants at the top
so the student can easily find and change them.

User-Input Panel keys (open modal overlays):
  I — open "New Shape" panel (Insert custom cube or pyramid)
  T — open "Transform" panel (enter exact translation / rotation / scale)
"""

import math
import os
import datetime
import pygame
from cg.scene_object import SceneObject
from cg.camera import Camera
from cg.animator import Animator

# ---------------------------------------------------------------------------
# Speed constants (per second)
# ---------------------------------------------------------------------------

TRANSLATE_SPEED: float = 2.0          # units/second for arrow keys
ROTATE_SPEED: float = math.radians(90.0)   # 90°/second for rotation keys
SCALE_SPEED: float = 0.5              # scale units/second
FOV_SPEED: float = 20.0              # degrees/second for FOV keys
MOUSE_SENSITIVITY: float = 0.005     # radians per pixel drag
SCROLL_SENSITIVITY: float = 0.3      # distance units per scroll notch

# Screenshot save directory
SCREENSHOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs", "screenshots"
)


class InputHandler:
    """
    Translates Pygame events and held keys into object/camera actions.

    Call:
      process_events(events, obj, camera, animator) — for one-shot key events
      process_held_keys(dt, obj, camera) — for keys that act while held down

    Panel integration:
      When a modal input panel is active, pass ``active_panel`` to
      ``process_events``; all keyboard events are forwarded to the panel
      and normal game controls are suppressed.
    """

    def __init__(self) -> None:
        self._mouse_dragging: bool = False
        self._last_mouse: tuple = (0, 0)

    # ------------------------------------------------------------------
    # Event-based input (fires once per key press)
    # ------------------------------------------------------------------

    def process_events(
        self,
        events: list,
        obj: SceneObject,
        camera: Camera,
        animator: Animator,
        shapes: list,          # list of Mesh objects to cycle through
        current_shape_idx: int,
        wireframe: bool,
        show_grid: bool,
        show_help: bool,
        surface: pygame.Surface,
        active_panel=None,     # InputPanel instance or None
    ) -> tuple:
        """
        Process a list of Pygame events. Returns updated state values:
        (quit, current_shape_idx, wireframe, show_grid, show_help,
         open_new_shape_panel, open_transform_panel)

        When ``active_panel`` is provided all keyboard events are routed
        to the panel; normal keybindings are disabled while it is open.
        """
        quit_requested = False
        open_new_shape  = False
        open_transform  = False

        for event in events:
            if event.type == pygame.QUIT:
                quit_requested = True

            elif event.type == pygame.KEYDOWN:

                # ---- If a modal input panel is open, forward events to it --
                if active_panel is not None:
                    active_panel.handle_event(event)
                    continue   # skip all normal keybindings

                k = event.key

                # ---- quit ------------------------------------------------
                if k == pygame.K_ESCAPE:
                    quit_requested = True

                # ---- shape switch (Tab) -----------------------------------
                elif k == pygame.K_TAB:
                    current_shape_idx = (current_shape_idx + 1) % len(shapes)
                    obj.mesh = shapes[current_shape_idx]

                # ---- camera presets (1-4) --------------------------------
                elif k == pygame.K_1:
                    camera.set_preset("front")
                elif k == pygame.K_2:
                    camera.set_preset("top")
                elif k == pygame.K_3:
                    camera.set_preset("side")
                elif k == pygame.K_4:
                    camera.set_preset("isometric")

                # ---- projection toggle (P) --------------------------------
                elif k == pygame.K_p:
                    camera.toggle_projection()

                # ---- transform order toggle (O) --------------------------
                elif k == pygame.K_o:
                    obj.toggle_transform_order()

                # ---- reflection plane cycle (X) --------------------------
                # Cycles: none → XY → YZ → XZ → none …
                # CG concept: a reflection matrix has det = -1, flipping
                # face winding. The renderer corrects back-face culling sign.
                elif k == pygame.K_x:
                    obj.cycle_reflection()

                # ---- animation controls ----------------------------------
                elif k == pygame.K_SPACE:
                    animator.pause_resume()
                elif k == pygame.K_m:
                    animator.toggle_rotate()
                elif k == pygame.K_b:
                    animator.toggle_bounce()
                elif k == pygame.K_n:
                    animator.toggle_scale_pulse()

                # ---- display toggles ------------------------------------
                elif k == pygame.K_f:
                    wireframe = not wireframe
                elif k == pygame.K_g:
                    show_grid = not show_grid
                elif k == pygame.K_h:
                    show_help = not show_help

                # ---- reset (R) ------------------------------------------
                elif k == pygame.K_r:
                    obj.reset()
                    camera.reset()

                # ---- screenshot (F12) ------------------------------------
                elif k == pygame.K_F12:
                    self._save_screenshot(surface)

                # ---- USER INPUT PANELS ----------------------------------
                # I → open "New Shape" panel to create a custom cube/pyramid
                elif k == pygame.K_i:
                    open_new_shape = True

                # T → open "Transform" panel for exact value entry
                elif k == pygame.K_t:
                    open_transform = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if active_panel is not None:
                    continue   # ignore mouse clicks while panel is open
                if event.button == 1:   # left button — start orbit drag
                    self._mouse_dragging = True
                    self._last_mouse = event.pos
                elif event.button == 4:  # scroll up — zoom in
                    camera.zoom(-SCROLL_SENSITIVITY)
                elif event.button == 5:  # scroll down — zoom out
                    camera.zoom(SCROLL_SENSITIVITY)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._mouse_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self._mouse_dragging and active_panel is None:
                    dx = event.pos[0] - self._last_mouse[0]
                    dy = event.pos[1] - self._last_mouse[1]
                    camera.orbit(dx * MOUSE_SENSITIVITY, dy * MOUSE_SENSITIVITY)
                    self._last_mouse = event.pos

        return (
            quit_requested,
            current_shape_idx,
            wireframe,
            show_grid,
            show_help,
            open_new_shape,
            open_transform,
        )

    # ------------------------------------------------------------------
    # Held-key input (fires every frame while key is held)
    # ------------------------------------------------------------------

    def process_held_keys(
        self,
        dt: float,
        obj: SceneObject,
        camera: Camera,
        panel_active: bool = False,
    ) -> None:
        """
        Read currently held keys and apply continuous actions.
        dt is seconds since last frame — all speeds multiplied by dt.
        Pass panel_active=True to suppress movement while a panel is open.
        """
        if panel_active:
            return

        keys = pygame.key.get_pressed()
        t = TRANSLATE_SPEED * dt
        r = ROTATE_SPEED * dt
        s = SCALE_SPEED * dt

        # ---- Translation (arrows + W/S) ----------------------------------
        if keys[pygame.K_LEFT]:
            obj.translate_by(-t, 0, 0)
        if keys[pygame.K_RIGHT]:
            obj.translate_by( t, 0, 0)
        if keys[pygame.K_UP]:
            obj.translate_by(0,  t, 0)
        if keys[pygame.K_DOWN]:
            obj.translate_by(0, -t, 0)
        if keys[pygame.K_w]:
            obj.translate_by(0, 0, -t)
        if keys[pygame.K_s]:
            obj.translate_by(0, 0,  t)

        # ---- Rotation (Q/E = X, A/D = Y, Z/C = Z) ----------------------
        if keys[pygame.K_q]:
            obj.rotate_by(-r, 0, 0)
        if keys[pygame.K_e]:
            obj.rotate_by( r, 0, 0)
        if keys[pygame.K_a]:
            obj.rotate_by(0, -r, 0)
        if keys[pygame.K_d]:
            obj.rotate_by(0,  r, 0)
        if keys[pygame.K_z]:
            obj.rotate_by(0, 0, -r)
        if keys[pygame.K_c]:
            obj.rotate_by(0, 0,  r)

        # ---- Scale (+ / -) -----------------------------------------------
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            obj.scale_by( s)
        if keys[pygame.K_MINUS]:
            obj.scale_by(-s)

        # ---- FOV ([ / ]) ------------------------------------------------
        if keys[pygame.K_LEFTBRACKET]:
            camera.adjust_fov(-FOV_SPEED * dt)
        if keys[pygame.K_RIGHTBRACKET]:
            camera.adjust_fov( FOV_SPEED * dt)

    # ------------------------------------------------------------------
    # Screenshot helper
    # ------------------------------------------------------------------

    def _save_screenshot(self, surface: pygame.Surface) -> None:
        """Save the current frame as a PNG to docs/screenshots/."""
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
        pygame.image.save(surface, path)
        print(f"Screenshot saved: {path}")
