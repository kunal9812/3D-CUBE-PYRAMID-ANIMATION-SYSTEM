"""
ui/hud.py — Heads-Up Display
==============================
Draws on-screen text panels showing:
  - FPS, shape name, projection mode, transform order, FOV
  - Camera yaw / pitch / distance
  - Object position / rotation (degrees) / scale
  - Live 4×4 model matrix (key viva feature!)
  - Optional help overlay with control table

User-input panel hint bar:
  When a modal input panel (NewShapePanel / TransformPanel) is active, a
  semi-transparent status bar is drawn at the bottom of the screen to remind
  the user that the panel is open.

Design:
  All text is rendered with pygame.font (no external fonts needed).
  Panels have a semi-transparent dark background for readability.
"""

import pygame
from cg.scene_object import SceneObject
from cg.camera import Camera
from cg.animator import Animator

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

PANEL_BG_COLOR = (10, 10, 20, 180)     # RGBA — semi-transparent
TEXT_COLOR = (200, 220, 255)
LABEL_COLOR = (140, 180, 255)
MATRIX_COLOR = (180, 255, 180)
TITLE_COLOR = (255, 220, 100)
WARN_COLOR = (255, 160, 60)
INPUT_HINT_COLOR = (255, 220, 100)
INPUT_HINT_BG = (10, 30, 10, 200)

FONT_SIZE_SMALL = 14
FONT_SIZE_NORMAL = 15
FONT_SIZE_HEADER = 17

MARGIN = 8                              # padding inside panel
LINE_H = 18                             # line height


# ---------------------------------------------------------------------------
# HUD class
# ---------------------------------------------------------------------------

class HUD:
    """
    Renders the on-screen information overlay onto a Pygame surface.

    All drawing is done on a separate transparent surface then blitted,
    so the 3D scene underneath is not obscured by black boxes.
    """

    def __init__(self) -> None:
        pygame.font.init()
        self._font_sm = pygame.font.SysFont("monospace", FONT_SIZE_SMALL)
        self._font   = pygame.font.SysFont("monospace", FONT_SIZE_NORMAL)
        self._font_hd = pygame.font.SysFont("monospace", FONT_SIZE_HEADER, bold=True)

    # ------------------------------------------------------------------
    # Main draw call
    # ------------------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        obj: SceneObject,
        camera: Camera,
        animator: Animator,
        fps: float,
        shape_name: str,
        wireframe: bool,
        show_help: bool,
        panel_active: bool = False,
        panel_kind: str = "",
    ) -> None:
        """Draw all HUD panels onto `surface`.

        Parameters
        ----------
        panel_active : True while a modal input panel is open.
        panel_kind   : "new_shape" or "transform" — used for the hint bar.
        """
        w, h = surface.get_size()

        # ---- left panel: status info ------------------------------------
        self._draw_status_panel(surface, obj, camera, animator, fps, shape_name, wireframe)

        # ---- right panel: live 4×4 model matrix -------------------------
        self._draw_matrix_panel(surface, obj, w)

        # ---- help overlay -----------------------------------------------
        if show_help and not panel_active:
            self._draw_help_overlay(surface, w, h)

        # ---- panel-active status bar at the bottom ----------------------
        if panel_active:
            self._draw_panel_hint_bar(surface, w, h, panel_kind)

    # ------------------------------------------------------------------
    # Status panel (top-left)
    # ------------------------------------------------------------------

    def _draw_status_panel(
        self,
        surface: pygame.Surface,
        obj: SceneObject,
        camera: Camera,
        animator: Animator,
        fps: float,
        shape_name: str,
        wireframe: bool,
    ) -> None:
        lines = []

        # FPS and shape
        lines.append(("TITLE", f"3D Animation System"))
        lines.append(("LABEL", f"Shape:"))
        lines.append(("TEXT",  f"  {shape_name}"))
        lines.append(("LABEL", f"FPS:"))
        lines.append(("TEXT",  f"  {fps:.1f}"))
        lines.append(("SEP",   ""))

        # Camera
        lines.append(("LABEL", "Camera"))
        lines.append(("TEXT",  f"  Projection: {camera.projection_mode.upper()}"))
        lines.append(("TEXT",  f"  FOV:        {camera.fov:.1f}°"))
        lines.append(("TEXT",  f"  Yaw:        {camera.yaw_deg:.1f}°"))
        lines.append(("TEXT",  f"  Pitch:      {camera.pitch_deg:.1f}°"))
        lines.append(("TEXT",  f"  Distance:   {camera.distance:.2f}"))
        lines.append(("SEP",   ""))

        # Object transform
        pos = obj.position
        rot = obj.rotation_deg
        sc  = obj.scale
        lines.append(("LABEL", "Object Transform"))
        lines.append(("TEXT",  f"  Order:  {obj.transform_order}"))
        lines.append(("TEXT",  f"  Pos:    ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"))
        lines.append(("TEXT",  f"  Rot°:   ({rot[0]:+.1f}, {rot[1]:+.1f}, {rot[2]:+.1f})"))
        lines.append(("TEXT",  f"  Scale:  ({sc[0]:.2f}, {sc[1]:.2f}, {sc[2]:.2f})"))
        # Reflection state — highlight in warning colour when active
        refl = obj.reflection_plane
        refl_tag = "WARN" if refl != "none" else "TEXT"
        refl_label = f"  Reflect: {refl.upper() if refl != 'none' else 'off'} [X]"
        lines.append((refl_tag, refl_label))
        lines.append(("SEP",   ""))

        # Animations
        lines.append(("LABEL", "Animations"))
        pa = "PAUSED" if animator.paused else "RUNNING"
        lines.append(("WARN" if animator.paused else "TEXT", f"  Status: {pa}"))
        lines.append(("TEXT",  f"  [M] Auto-rotate: {'ON' if animator.auto_rotate.enabled else 'off'}"))
        lines.append(("TEXT",  f"  [B] Bounce:      {'ON' if animator.bounce.enabled else 'off'}"))
        lines.append(("TEXT",  f"  [N] ScalePulse:  {'ON' if animator.scale_pulse.enabled else 'off'}"))
        lines.append(("SEP",   ""))

        # Render mode
        lines.append(("TEXT",  f"[F] Mode: {'Wireframe' if wireframe else 'Solid'}"))
        lines.append(("TEXT",  "[H] Toggle help"))
        lines.append(("SEP",   ""))

        # User-input panel hints
        lines.append(("LABEL", "User Input Panels"))
        lines.append(("TEXT",  "  [I] New shape (custom coords)"))
        lines.append(("TEXT",  "  [T] Transform (exact values)"))

        self._blit_panel(surface, lines, x=MARGIN, y=MARGIN, max_width=280)

    # ------------------------------------------------------------------
    # Matrix panel (top-right)
    # ------------------------------------------------------------------

    def _draw_matrix_panel(
        self,
        surface: pygame.Surface,
        obj: SceneObject,
        screen_width: int,
    ) -> None:
        """
        Render the live 4×4 model matrix as a formatted grid.

        This is the key viva feature — the student can point to it and explain
        how position/rotation/scale map to the matrix values.
        """
        M = obj.model_matrix()

        header = [("TITLE", "Model Matrix (4×4)")]

        matrix_lines = []
        for row in range(4):
            row_str = "  | " + "  ".join(f"{M[row, col]:+7.3f}" for col in range(4)) + " |"
            matrix_lines.append(("MATRIX", row_str))

        all_lines = header + matrix_lines
        panel_w = 330
        x = screen_width - panel_w - MARGIN
        self._blit_panel(surface, all_lines, x=x, y=MARGIN, max_width=panel_w)

    # ------------------------------------------------------------------
    # Panel-active hint bar (bottom of screen)
    # ------------------------------------------------------------------

    def _draw_panel_hint_bar(
        self,
        surface: pygame.Surface,
        screen_w: int,
        screen_h: int,
        panel_kind: str,
    ) -> None:
        """Draw a narrow banner at the bottom while a modal panel is open."""
        bar_h = 28
        bar = pygame.Surface((screen_w, bar_h), pygame.SRCALPHA)
        bar.fill(INPUT_HINT_BG)
        surface.blit(bar, (0, screen_h - bar_h))

        if panel_kind == "new_shape":
            msg = "NEW SHAPE PANEL OPEN — Tab: next field | Left/Right: switch shape | Enter: create | Esc: cancel"
        elif panel_kind == "transform":
            msg = "TRANSFORM PANEL OPEN — Tab: next field | Enter: apply | Esc: cancel"
        else:
            msg = "INPUT PANEL OPEN — Tab: next field | Enter: submit | Esc: cancel"

        hint_surf = self._font_sm.render(msg, True, INPUT_HINT_COLOR)
        surface.blit(hint_surf, (MARGIN, screen_h - bar_h + 6))

    # ------------------------------------------------------------------
    # Help overlay (full-screen semi-transparent)
    # ------------------------------------------------------------------

    def _draw_help_overlay(
        self,
        surface: pygame.Surface,
        screen_w: int,
        screen_h: int,
    ) -> None:
        """Draw the controls reference table in the centre of the screen."""
        controls = [
            ("TITLE", "Controls Reference"),
            ("SEP", ""),
            ("LABEL", "Object Transforms"),
            ("TEXT", "  Arrows / W,S       Translate X/Y/Z"),
            ("TEXT", "  Q/E               Rotate about X"),
            ("TEXT", "  A/D               Rotate about Y"),
            ("TEXT", "  Z/C               Rotate about Z"),
            ("TEXT", "  + / -             Uniform scale"),
            ("SEP", ""),
            ("LABEL", "Camera"),
            ("TEXT", "  Mouse drag (LMB)  Orbit"),
            ("TEXT", "  Mouse wheel       Zoom"),
            ("TEXT", "  [ / ]             Decrease / Increase FOV"),
            ("TEXT", "  1 2 3 4           Presets: Front/Top/Side/Iso"),
            ("TEXT", "  P                 Toggle Perspective / Ortho"),
            ("SEP", ""),
            ("LABEL", "Animations"),
            ("TEXT", "  Space             Pause / Resume"),
            ("TEXT", "  M                 Auto-rotate"),
            ("TEXT", "  B                 Bounce"),
            ("TEXT", "  N                 Scale pulse"),
            ("SEP", ""),
            ("LABEL", "User Input Panels"),
            ("TEXT", "  I                 New shape (custom coordinates)"),
            ("TEXT", "  T                 Transform (type exact values)"),
            ("SEP", ""),
            ("LABEL", "Display"),
            ("TEXT", "  Tab               Switch Cube / Pyramid"),
            ("TEXT", "  F                 Wireframe / Solid"),
            ("TEXT", "  G                 Grid + Axes"),
            ("TEXT", "  O                 Transform order TRS / SRT"),
            ("TEXT", "  X                 Reflection plane (none/XY/YZ/XZ)"),
            ("TEXT", "  R                 Reset object & camera"),
            ("TEXT", "  F12               Save screenshot"),
            ("TEXT", "  H                 Toggle this help"),
            ("TEXT", "  Esc               Quit"),
        ]

        # Measure panel height
        panel_h = MARGIN * 2 + len(controls) * LINE_H
        panel_w = 420
        px = (screen_w - panel_w) // 2
        py = (screen_h - panel_h) // 2

        self._blit_panel(surface, controls, x=px, y=py, max_width=panel_w)

    # ------------------------------------------------------------------
    # Generic panel renderer
    # ------------------------------------------------------------------

    def _blit_panel(
        self,
        surface: pygame.Surface,
        lines: list,
        x: int,
        y: int,
        max_width: int,
    ) -> None:
        """
        Draw a rounded-rectangle panel with coloured text lines.

        `lines` is a list of (tag, text_str) pairs where tag is one of:
          "TITLE", "LABEL", "TEXT", "MATRIX", "SEP", "WARN"
        """
        # Build a temp surface for alpha blending
        panel_h = MARGIN * 2 + sum(1 if tag != "SEP" else 0 for tag, _ in lines) * LINE_H
        panel_h += sum(LINE_H // 2 for tag, _ in lines if tag == "SEP")

        panel = pygame.Surface((max_width, panel_h + MARGIN), pygame.SRCALPHA)
        panel.fill(PANEL_BG_COLOR)
        surface.blit(panel, (x, y))

        cy = y + MARGIN
        for tag, text in lines:
            if tag == "SEP":
                cy += LINE_H // 2
                continue
            color = {
                "TITLE":  TITLE_COLOR,
                "LABEL":  LABEL_COLOR,
                "TEXT":   TEXT_COLOR,
                "MATRIX": MATRIX_COLOR,
                "WARN":   WARN_COLOR,
            }.get(tag, TEXT_COLOR)
            font = self._font_hd if tag == "TITLE" else (
                self._font_sm if tag == "MATRIX" else self._font
            )
            rendered = font.render(text, True, color)
            surface.blit(rendered, (x + MARGIN, cy))
            cy += LINE_H
