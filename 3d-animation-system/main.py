

import sys
import pygame

from cg.shapes import make_cube, make_pyramid
from cg.scene_object import SceneObject
from cg.camera import Camera
from cg.renderer import Renderer, BACKGROUND_COLOR
from cg.animator import Animator
from ui.input_handler import InputHandler
from ui.hud import HUD
from ui.input_panel import NewShapePanel, TransformPanel


WINDOW_TITLE = "3D Animation System — CG Mini Project"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 720
TARGET_FPS = 60

# Shape names for HUD display (updated when user creates a custom shape)
SHAPE_NAMES = ["Cube", "Pyramid"]


# ---------------------------------------------------------------------------
# Panel result processors
# ---------------------------------------------------------------------------

def _apply_new_shape(result: dict, obj: SceneObject) -> None:
    """
    Given a validated result dict from NewShapePanel, build a new Mesh from the
    user-entered coordinates and swap it onto the scene object.

    The object's transform (position/rotation/scale) is reset so the user can
    see the shape cleanly at first.

    CG concept demonstrated: constructing a Mesh from arbitrary vertex
    coordinates and observing how the geometry changes in the viewport.
    """
    shape = result["shape"]
    cx, cy, cz = result["cx"], result["cy"], result["cz"]

    if shape == "cube":
        w = max(result["width"],  0.01)
        h = max(result["height"], 0.01)
        d = max(result["depth"],  0.01)
        new_mesh = make_cube(center=(cx, cy, cz), size=(w, h, d))
    else:
        bw = max(result["base_w"],         0.01)
        bd = max(result["base_d"],         0.01)
        ph = max(result["pyramid_height"], 0.01)
        new_mesh = make_pyramid(center=(cx, cy, cz), base_size=(bw, bd), height=ph)

    obj.mesh = new_mesh
    obj.reset()   # zero out position/rotation/scale so the custom shape is centred


def _apply_transform(result: dict, obj: SceneObject) -> None:
    """
    Apply the user-entered translation, rotation, and scale values to `obj`.

    Translation and rotation are ADDITIVE (offsets on top of current values).
    Scale values are set as ABSOLUTE replacements (not offsets).

    CG concept demonstrated: exact numeric control over the TRS model matrix
    — the user types values and immediately sees the matrix update in the HUD.
    """
    # Translation — add offset to current position (reuses the existing translate_by method)
    obj.translate_by(result["tx"], result["ty"], result["tz"])

    # Rotation — add degree offsets (converted to radians inside apply_rotation_deg)
    obj.apply_rotation_deg(result["rx"], result["ry"], result["rz"])

    # Scale — set absolute values (not an offset)
    obj.set_scale(result["sx"], result["sy"], result["sz"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point: initialise Pygame, build objects, run the event loop."""

    # ---- Pygame setup ----------------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.RESIZABLE,     # allow the user to resize the window
    )
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()

    # ---- Build scene objects ---------------------------------------------
    shapes = [make_cube(), make_pyramid()]       # Meshes to cycle through
    current_shape_idx = 0

    obj = SceneObject(shapes[current_shape_idx])  # wraps a mesh with transforms
    camera = Camera()                             # orbit camera
    renderer = Renderer()                         # stateless renderer
    animator = Animator()                         # time-based animations
    input_handler = InputHandler()                # keyboard / mouse mapper
    hud = HUD()                                   # on-screen text overlay

    # ---- Display-mode state (not stored in any class — UI-level state) ---
    wireframe: bool = False    # start in solid mode
    show_grid: bool = True
    show_help: bool = True     # show help on first launch

    # ---- Modal input panel state -----------------------------------------
    # active_panel is None when no panel is open, or a NewShapePanel /
    # TransformPanel instance while the user is entering values.
    # Only one panel can be open at a time.
    active_panel = None          # NewShapePanel | TransformPanel | None
    panel_kind: str = ""         # "new_shape" or "transform" (for HUD hint bar)

    # ---- Main loop -------------------------------------------------------
    running = True
    while running:
        # -- delta time (seconds since last frame) -------------------------
        dt = clock.tick(TARGET_FPS) / 1000.0   # ms → seconds
        dt = min(dt, 0.05)                       # cap dt to avoid spiral of death

        # -- events --------------------------------------------------------
        events = pygame.event.get()

        (
            quit_req,
            current_shape_idx,
            wireframe,
            show_grid,
            show_help,
            open_new_shape,
            open_transform,
        ) = input_handler.process_events(
            events=events,
            obj=obj,
            camera=camera,
            animator=animator,
            shapes=shapes,
            current_shape_idx=current_shape_idx,
            wireframe=wireframe,
            show_grid=show_grid,
            show_help=show_help,
            surface=screen,
            active_panel=active_panel,   # panel receives keyboard while open
        )

        if quit_req:
            running = False

        # Handle window resize: recompute aspect ratio automatically
        for event in events:
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )

        # -- open panels on I / T key press (only when no panel is open) ---
        if open_new_shape and active_panel is None:
            active_panel = NewShapePanel()
            panel_kind = "new_shape"

        if open_transform and active_panel is None:
            # Pre-fill scale fields with the object's current scale
            active_panel = TransformPanel(obj=obj)
            panel_kind = "transform"

        # -- process panel submission / cancellation -----------------------
        if active_panel is not None:
            if active_panel.submitted:
                if panel_kind == "new_shape":
                    _apply_new_shape(active_panel.result, obj)
                    # Reset Tab-shape index to avoid stale mesh from shapes list
                    current_shape_idx = 0
                elif panel_kind == "transform":
                    _apply_transform(active_panel.result, obj)
                active_panel = None
                panel_kind = ""
            elif active_panel.cancelled:
                active_panel = None
                panel_kind = ""

        # -- held keys (continuous movement) — suppressed while panel open --
        input_handler.process_held_keys(
            dt, obj, camera, panel_active=(active_panel is not None)
        )

        # -- update animations ---------------------------------------------
        animator.update(obj, dt)

        # -- render --------------------------------------------------------
        screen.fill(BACKGROUND_COLOR)

        renderer.draw(
            surface=screen,
            obj=obj,
            camera=camera,
            wireframe=wireframe,
            show_grid=show_grid,
        )

        # -- draw modal input panel (on top of 3D scene, below HUD) -------
        if active_panel is not None:
            active_panel.draw(screen)

        # -- HUD -----------------------------------------------------------
        fps = clock.get_fps()
        hud.draw(
            surface=screen,
            obj=obj,
            camera=camera,
            animator=animator,
            fps=fps,
            shape_name=SHAPE_NAMES[current_shape_idx],
            wireframe=wireframe,
            show_help=show_help,
            panel_active=(active_panel is not None),
            panel_kind=panel_kind,
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
