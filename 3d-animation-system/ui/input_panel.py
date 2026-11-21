"""
ui/input_panel.py — Interactive Input Panel Overlay
=====================================================
CG Topic: User-Controlled Transformations & Custom Shape Creation

Provides a Pygame-based text-entry overlay that sits on top of the 3D viewport.
Two modes are available:

  Mode A — "New Shape" (press I in the main app):
    Enter center coordinates, size, and colour to spawn a custom cube or pyramid.

  Mode B — "Transform" (press T in the main app):
    Type exact translation, rotation (degrees), or scale values and apply them
    to the currently-selected shape in one shot.

Keyboard handling inside the panel:
  - Digit keys, '.' and '-' are accepted in numeric fields.
  - Backspace deletes the last character.
  - Tab / Shift+Tab move between fields.
  - Enter / Return on the last field (or anytime for single-field forms) submits.
  - Escape cancels without applying anything.
  - Left/Right arrow keys switch shape type (New-Shape mode only).

The panel is completely self-contained: call handle_event() for each Pygame
event, then call draw() every frame to paint the overlay.  After submission
the `submitted` flag is True and `result` holds the parsed data dict.
"""

import pygame

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PANEL_BG        = (10, 10, 25, 215)         # RGBA semi-transparent
FIELD_BG        = (30, 30, 50)
FIELD_ACTIVE_BG = (50, 55, 90)
FIELD_BORDER    = (80, 90, 130)
FIELD_ACTIVE_BORDER = (130, 170, 255)
TEXT_COLOR      = (210, 225, 255)
LABEL_COLOR     = (140, 180, 255)
TITLE_COLOR     = (255, 220, 100)
HINT_COLOR      = (160, 180, 160)
SUBMIT_COLOR    = (80, 200, 120)
SECTION_COLOR   = (200, 160, 80)

FONT_SIZE       = 15
TITLE_SIZE      = 17
FIELD_H         = 26
FIELD_W         = 180
LABEL_W         = 160
MARGIN          = 14
ROW_H           = 32
SECTION_GAP     = 10


# ---------------------------------------------------------------------------
# Helper: safe float parse
# ---------------------------------------------------------------------------

def _parse_float(text: str, default: float = 0.0) -> float:
    """Return float from text, or default if text is empty / invalid."""
    try:
        return float(text) if text not in ("", "-", ".") else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Field descriptor
# ---------------------------------------------------------------------------

class _Field:
    """A single labelled numeric text-entry field."""

    def __init__(self, label: str, default: str = "0") -> None:
        self.label: str = label
        self.text: str = default

    def handle_char(self, char: str) -> None:
        """Accept a printable character into the field."""
        if char == "-":
            # Allow minus only at start
            if self.text == "" or self.text == "0":
                self.text = "-"
        elif char == ".":
            if "." not in self.text:
                self.text += "."
        elif char.isdigit():
            if self.text == "0":
                self.text = char
            elif self.text == "-0":
                self.text = "-" + char
            else:
                self.text += char

    def backspace(self) -> None:
        self.text = self.text[:-1]

    @property
    def value(self) -> float:
        return _parse_float(self.text, 0.0)


# ---------------------------------------------------------------------------
# Base panel
# ---------------------------------------------------------------------------

class _BasePanel:
    """Shared rendering and navigation logic."""

    def __init__(self, title: str) -> None:
        pygame.font.init()
        self._font       = pygame.font.SysFont("monospace", FONT_SIZE)
        self._font_title = pygame.font.SysFont("monospace", TITLE_SIZE, bold=True)
        self._font_hint  = pygame.font.SysFont("monospace", FONT_SIZE - 1)

        self.title: str = title
        self._fields: list[_Field] = []   # filled by subclass
        self._active: int = 0             # index of focused field

        self.submitted: bool = False      # True once Enter accepted
        self.cancelled: bool = False      # True once Esc pressed
        self.result: dict = {}            # parsed values after submission

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        """Route a Pygame event into the panel. Call for every event."""
        if event.type != pygame.KEYDOWN:
            return

        k = event.key

        if k == pygame.K_ESCAPE:
            self.cancelled = True

        elif k == pygame.K_RETURN or k == pygame.K_KP_ENTER:
            if self._active == len(self._fields) - 1:
                self._submit()
            else:
                self._active = min(self._active + 1, len(self._fields) - 1)

        elif k == pygame.K_TAB:
            mods = getattr(event, "mod", 0) | pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                self._active = max(self._active - 1, 0)
            else:
                self._active = min(self._active + 1, len(self._fields) - 1)

        elif k == pygame.K_BACKSPACE:
            self._fields[self._active].backspace()

        else:
            # Accept printable characters for numeric input
            char = event.unicode
            if char in "0123456789.-":
                self._fields[self._active].handle_char(char)

        # Let subclasses intercept additional keys (e.g. shape toggle)
        self._handle_extra_key(k)

    def _handle_extra_key(self, key: int) -> None:
        """Override in subclass to handle additional key events."""
        pass

    def _submit(self) -> None:
        """Override in subclass to validate and populate self.result."""
        self.submitted = True

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        """Paint the panel overlay onto `surface`."""
        sw, sh = surface.get_size()
        panel_w, panel_h = self._panel_size()
        px = (sw - panel_w) // 2
        py = (sh - panel_h) // 2

        # Semi-transparent background
        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill(PANEL_BG)
        surface.blit(bg, (px, py))

        # Border
        pygame.draw.rect(surface, FIELD_ACTIVE_BORDER, (px, py, panel_w, panel_h), 2, border_radius=6)

        # Title
        title_surf = self._font_title.render(self.title, True, TITLE_COLOR)
        surface.blit(title_surf, (px + MARGIN, py + MARGIN))

        # Content (delegated to subclass)
        content_y = py + MARGIN + TITLE_SIZE + MARGIN
        self._draw_content(surface, px, content_y)

        # Footer hint
        hint = "Tab/Shift+Tab: navigate   Enter: next/submit   Esc: cancel"
        hint_surf = self._font_hint.render(hint, True, HINT_COLOR)
        surface.blit(hint_surf, (px + MARGIN, py + panel_h - FONT_SIZE - MARGIN))

    def _panel_size(self) -> tuple[int, int]:
        """Override to return (width, height)."""
        return (460, 200)

    def _draw_content(self, surface: pygame.Surface, px: int, start_y: int) -> None:
        """Override to draw fields and labels inside the panel."""
        pass

    # ------------------------------------------------------------------
    # Shared field renderer
    # ------------------------------------------------------------------

    def _draw_field_row(
        self,
        surface: pygame.Surface,
        field: _Field,
        idx: int,
        x: int,
        y: int,
    ) -> None:
        """Draw one label + input-box row."""
        active = idx == self._active

        # Label
        label_surf = self._font.render(field.label, True, LABEL_COLOR)
        surface.blit(label_surf, (x, y + 5))

        # Input box
        box_x = x + LABEL_W
        bg_col = FIELD_ACTIVE_BG if active else FIELD_BG
        border_col = FIELD_ACTIVE_BORDER if active else FIELD_BORDER
        pygame.draw.rect(surface, bg_col, (box_x, y, FIELD_W, FIELD_H), border_radius=4)
        pygame.draw.rect(surface, border_col, (box_x, y, FIELD_W, FIELD_H), 1, border_radius=4)

        # Text inside box
        display = field.text + ("|" if active else "")
        text_surf = self._font.render(display, True, TEXT_COLOR)
        surface.blit(text_surf, (box_x + 6, y + 5))


# ---------------------------------------------------------------------------
# Mode A — New Shape panel
# ---------------------------------------------------------------------------

class NewShapePanel(_BasePanel):
    """
    Panel that collects parameters to create a new cube or pyramid.

    Result dict keys (after submission):
        "shape"  : "cube" or "pyramid"
        "cx", "cy", "cz"       : center coordinates
        For cube:
            "width", "height", "depth"
        For pyramid:
            "base_w", "base_d", "pyramid_height"
        "r", "g", "b"          : face colour, 0–255
    """

    _SHAPES = ["cube", "pyramid"]

    def __init__(self) -> None:
        super().__init__("New Shape — Enter Parameters")
        self._shape_idx: int = 0   # 0=cube, 1=pyramid
        self._rebuild_fields()

    # ------------------------------------------------------------------
    # Fields vary by shape type
    # ------------------------------------------------------------------

    def _rebuild_fields(self) -> None:
        """(Re)create field list for the currently selected shape."""
        # Common: center
        common = [
            _Field("Center X:", "0"),
            _Field("Center Y:", "0"),
            _Field("Center Z:", "0"),
        ]
        if self._shape_idx == 0:   # cube
            shape_fields = [
                _Field("Width  (X):", "2"),
                _Field("Height (Y):", "2"),
                _Field("Depth  (Z):", "2"),
            ]
        else:                      # pyramid
            shape_fields = [
                _Field("Base Width:", "2"),
                _Field("Base Depth:", "2"),
                _Field("Height:",     "2"),
            ]
        color_fields = [
            _Field("Color R (0-255):", "100"),
            _Field("Color G (0-255):", "160"),
            _Field("Color B (0-255):", "255"),
        ]
        old_active = self._active
        self._fields = common + shape_fields + color_fields
        self._active = min(old_active, len(self._fields) - 1)

    def _handle_extra_key(self, key: int) -> None:
        """Left/Right arrow toggle shape type."""
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            self._shape_idx = 1 - self._shape_idx
            self._rebuild_fields()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def _submit(self) -> None:
        cx, cy, cz = (self._fields[i].value for i in range(3))
        a, b, c    = (self._fields[i].value for i in range(3, 6))
        r          = max(0, min(255, int(self._fields[6].value)))
        g          = max(0, min(255, int(self._fields[7].value)))
        bl         = max(0, min(255, int(self._fields[8].value)))

        shape_name = self._SHAPES[self._shape_idx]
        self.result = {
            "shape": shape_name,
            "cx": cx, "cy": cy, "cz": cz,
            "r": r, "g": g, "b": bl,
        }
        if shape_name == "cube":
            self.result["width"]  = abs(a) or 1.0
            self.result["height"] = abs(b) or 1.0
            self.result["depth"]  = abs(c) or 1.0
        else:
            self.result["base_w"]          = abs(a) or 1.0
            self.result["base_d"]          = abs(b) or 1.0
            self.result["pyramid_height"]  = abs(c) or 1.0

        self.submitted = True

    # ------------------------------------------------------------------
    # Size / drawing
    # ------------------------------------------------------------------

    def _panel_size(self) -> tuple[int, int]:
        num_fields = len(self._fields)
        h = MARGIN + TITLE_SIZE + MARGIN            # title
        h += ROW_H                                  # shape selector row
        h += SECTION_GAP
        h += ROW_H * num_fields                    # fields
        h += SECTION_GAP + FONT_SIZE + MARGIN * 2  # hint
        return (LABEL_W + FIELD_W + MARGIN * 2 + 20, h)

    def _draw_content(self, surface: pygame.Surface, px: int, start_y: int) -> None:
        cx = px + MARGIN
        y  = start_y

        # Shape selector
        shape_name = self._SHAPES[self._shape_idx].capitalize()
        selector = self._font.render(
            f"Shape: ◄ {shape_name} ► (Left/Right to switch)", True, SECTION_COLOR
        )
        surface.blit(selector, (cx, y + 5))
        y += ROW_H + SECTION_GAP

        # Draw section headers + fields
        section_headers = {
            0: "--- Center Position ---",
            3: "--- Size ---",
            6: "--- Colour (0–255) ---",
        }

        for i, field in enumerate(self._fields):
            if i in section_headers:
                hdr_surf = self._font_hint.render(section_headers[i], True, SECTION_COLOR)
                surface.blit(hdr_surf, (cx, y))
                y += SECTION_GAP + 2
            self._draw_field_row(surface, field, i, cx, y)
            y += ROW_H

        # Submit hint
        hint = self._font.render("Enter on last field to CREATE shape", True, SUBMIT_COLOR)
        surface.blit(hint, (cx, y + 4))


# ---------------------------------------------------------------------------
# Mode B — Transform panel
# ---------------------------------------------------------------------------

class TransformPanel(_BasePanel):
    """
    Panel for entering exact translation, rotation, and scale values.

    All three groups are shown at once; the user fills them in and
    presses Enter on the last field to apply everything simultaneously.

    Rotation is entered in DEGREES (converted to radians before returning).

    Result dict keys (after submission):
        "tx", "ty", "tz"  — translation offsets (added to current position)
        "rx", "ry", "rz"  — rotation offsets in DEGREES (caller converts)
        "sx", "sy", "sz"  — NEW absolute scale values (replace current scale)
    """

    # Field order: 0-2 translate, 3-5 rotate, 6-8 scale
    def __init__(self, obj=None) -> None:
        super().__init__("Transform — Enter Values")

        # Pre-fill scale fields with the object's current scale if provided
        sx = sy = sz = "1"
        if obj is not None:
            sx = f"{obj.scale[0]:.3f}"
            sy = f"{obj.scale[1]:.3f}"
            sz = f"{obj.scale[2]:.3f}"

        self._fields = [
            # Translation
            _Field("Translate X:", "0"),
            _Field("Translate Y:", "0"),
            _Field("Translate Z:", "0"),
            # Rotation (degrees)
            _Field("Rotate X (°):", "0"),
            _Field("Rotate Y (°):", "0"),
            _Field("Rotate Z (°):", "0"),
            # Scale (absolute)
            _Field("Scale X:", sx),
            _Field("Scale Y:", sy),
            _Field("Scale Z:", sz),
        ]

    def _submit(self) -> None:
        self.result = {
            "tx": self._fields[0].value,
            "ty": self._fields[1].value,
            "tz": self._fields[2].value,
            "rx": self._fields[3].value,
            "ry": self._fields[4].value,
            "rz": self._fields[5].value,
            "sx": self._fields[6].value or 1.0,
            "sy": self._fields[7].value or 1.0,
            "sz": self._fields[8].value or 1.0,
        }
        self.submitted = True

    def _panel_size(self) -> tuple[int, int]:
        h = MARGIN + TITLE_SIZE + MARGIN
        h += (ROW_H + SECTION_GAP) * 3   # 3 sections with gap before each
        h += ROW_H * 9                    # 9 fields
        h += SECTION_GAP + FONT_SIZE + MARGIN * 2
        return (LABEL_W + FIELD_W + MARGIN * 2 + 20, h)

    def _draw_content(self, surface: pygame.Surface, px: int, start_y: int) -> None:
        cx = px + MARGIN
        y  = start_y

        section_headers = {
            0: "--- Translation (units to ADD to position) ---",
            3: "--- Rotation (degrees to ADD to current rotation) ---",
            6: "--- Scale (new absolute value, e.g. 1.0 = normal) ---",
        }

        for i, field in enumerate(self._fields):
            if i in section_headers:
                hdr_surf = self._font_hint.render(section_headers[i], True, SECTION_COLOR)
                surface.blit(hdr_surf, (cx, y))
                y += SECTION_GAP + 2
            self._draw_field_row(surface, field, i, cx, y)
            y += ROW_H

        hint = self._font.render("Enter on last field to APPLY transform", True, SUBMIT_COLOR)
        surface.blit(hint, (cx, y + 4))
