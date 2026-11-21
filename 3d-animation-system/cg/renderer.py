"""
cg/renderer.py — 3D Rendering Pipeline
========================================
CG Topic: The Full Viewing Pipeline + Back-Face Culling + Painter's Algorithm

This module implements the complete pipeline that takes a 3D object and
draws it on a 2D Pygame surface. Each stage is a clearly labelled function.

Pipeline stages:
  1. Model transform    : object space → world space      (apply model matrix)
  2. View transform     : world space → camera/view space (apply view matrix)
  3. Projection         : view space  → clip space        (apply projection matrix)
  4. Near-plane guard   : skip geometry behind the camera (w ≤ 0)
  5. Perspective divide : clip space  → NDC               (divide by w)
  6. Viewport mapping   : NDC → screen pixels             (y-flip because screen y ↓)
  7. Draw               : wireframe (edges) or solid (faces + culling + depth sort)

Back-face culling (solid mode):
  A face is invisible if it points away from the camera.
  In view space (camera at origin for perspective):
    face is VISIBLE if dot(normal_view, vertex_view) < 0
  For orthographic, camera is at infinity along -Z:
    face is VISIBLE if normal_view.z > 0

Painter's algorithm:
  Sort visible faces by their average Z depth in view space, farthest first,
  then draw them back-to-front. Simple but breaks for intersecting faces.

Flat shading (Lambert):
  intensity = max(0, dot(normal_world, light_dir)) + ambient
  colour = base_colour * intensity
"""

import numpy as np
import pygame
from typing import Tuple, Optional

from cg.mesh import Mesh
from cg.scene_object import SceneObject
from cg.camera import Camera

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colours
WIREFRAME_COLOR: Tuple[int, int, int] = (180, 220, 255)
EDGE_OUTLINE_COLOR: Tuple[int, int, int] = (30, 30, 30)
AXIS_X_COLOR: Tuple[int, int, int] = (220, 50, 50)    # red  — X
AXIS_Y_COLOR: Tuple[int, int, int] = (50, 200, 50)    # green — Y
AXIS_Z_COLOR: Tuple[int, int, int] = (50, 100, 220)   # blue  — Z
GRID_COLOR: Tuple[int, int, int] = (60, 60, 70)
BACKGROUND_COLOR: Tuple[int, int, int] = (20, 20, 30)

# Light direction (world space, unit vector pointing toward the light)
LIGHT_DIR: np.ndarray = np.array([0.5, 1.0, 0.7], dtype=np.float64)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)

AMBIENT: float = 0.25        # minimum brightness even on dark faces
NEAR_EPSILON: float = 0.001  # skip geometry with w ≤ this (behind / at camera)

AXIS_LENGTH: float = 1.5     # length of the gizmo axis lines
GRID_SIZE: int = 5            # grid extends ±GRID_SIZE units on X and Z
GRID_STEP: float = 1.0        # spacing between grid lines


# ---------------------------------------------------------------------------
# Renderer class
# ---------------------------------------------------------------------------

class Renderer:
    """
    Stateless renderer — all state lives in SceneObject and Camera.
    Call draw() once per frame.
    """

    def __init__(self) -> None:
        pass  # no instance state needed

    # ==================================================================
    # Public entry point
    # ==================================================================

    def draw(
        self,
        surface: pygame.Surface,
        obj: SceneObject,
        camera: Camera,
        wireframe: bool = True,
        show_grid: bool = True,
    ) -> None:
        """
        Render one SceneObject to `surface`.

        Parameters
        ----------
        surface   : the Pygame surface to draw on (already cleared)
        obj       : scene object with mesh + transforms
        camera    : camera with view + projection settings
        wireframe : True → draw edges only; False → solid with shading
        show_grid : whether to draw the ground grid and axis gizmo
        """
        width, height = surface.get_size()
        aspect = width / height

        # ---- matrices ----------------------------------------------------
        M = obj.model_matrix()                       # model (object → world)
        V = camera.view_matrix()                     # view  (world  → camera)
        P = camera.projection_matrix(aspect)         # projection
        VP = V                                       # combined view for short refs
        MVP = P @ V @ M                              # full MVP: object → clip space

        # ---- transform all vertices of the mesh -------------------------
        # Stage 1 & 2: object → world → view space
        verts_obj = obj.mesh.vertices                # (N, 4)
        # World space (for lighting normals)
        world_verts = (M @ verts_obj.T).T            # (N, 4) world
        # View space (for depth sorting and back-face culling)
        view_verts = (V @ world_verts.T).T           # (N, 4) view
        # Clip space (for projection)
        clip_verts = (P @ view_verts.T).T            # (N, 4) clip

        # ---- helper: project clip → screen pixels -----------------------
        def clip_to_screen(clip: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
            """
            Stages 4-6: near-plane guard, perspective divide, viewport mapping.

            Returns (screen_xy, valid_mask) where screen_xy is (N,2) int32
            and valid_mask is a boolean (N,) array — True where w > epsilon.
            """
            w = clip[:, 3]
            valid = w > NEAR_EPSILON       # guard: skip points behind/at camera
            # Perspective divide: NDC = clip_xyz / w
            # Use a safe w (replace invalid w with 1 to avoid div-by-zero)
            safe_w = np.where(valid, w, 1.0)
            ndc_x = clip[:, 0] / safe_w
            ndc_y = clip[:, 1] / safe_w
            # Viewport mapping: NDC [-1,1] → screen pixels
            # Note: screen Y is FLIPPED (screen y grows downward)
            sx = ((ndc_x + 1.0) * 0.5 * width ).astype(np.int32)
            sy = ((1.0 - ndc_y) * 0.5 * height).astype(np.int32)
            screen_xy = np.stack([sx, sy], axis=1)
            return screen_xy, valid

        screen_pts, valid_mask = clip_to_screen(clip_verts)

        # ---- draw grid and axes -----------------------------------------
        if show_grid:
            self._draw_grid(surface, V, P, width, height)
            self._draw_axes(surface, M, V, P, width, height)

        # ---- draw the mesh -----------------------------------------------
        if wireframe:
            self._draw_wireframe(surface, obj.mesh, screen_pts, valid_mask)
        else:
            self._draw_solid(
                surface, obj.mesh,
                screen_pts, valid_mask,
                world_verts, view_verts,
                M, camera,
                reflected=(obj.reflection_plane != "none"),
            )

    # ==================================================================
    # Stage 7a: Wireframe drawing
    # ==================================================================

    def _draw_wireframe(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        screen_pts: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        """Draw each edge as a 2D line between its projected endpoints."""
        for i, j in mesh.edges:
            # Only draw the edge if BOTH endpoints are in front of the camera
            if valid_mask[i] and valid_mask[j]:
                p1 = (int(screen_pts[i, 0]), int(screen_pts[i, 1]))
                p2 = (int(screen_pts[j, 0]), int(screen_pts[j, 1]))
                pygame.draw.line(surface, WIREFRAME_COLOR, p1, p2, 1)

    # ==================================================================
    # Stage 7b: Solid drawing (culling + depth sort + shading)
    # ==================================================================

    def _draw_solid(
        self,
        surface: pygame.Surface,
        mesh: Mesh,
        screen_pts: np.ndarray,
        valid_mask: np.ndarray,
        world_verts: np.ndarray,
        view_verts: np.ndarray,
        M: np.ndarray,
        camera: Camera,
        reflected: bool = False,
    ) -> None:
        """
        Draw filled faces using:
          1. Back-face culling   — skip faces pointing away from camera
          2. Painter's algorithm — draw farthest faces first
          3. Flat shading        — Lambert diffuse + ambient

        reflected : True when the model matrix contains a reflection.
          A reflection flips face winding (CCW → CW), so the culling sign is
          reversed to avoid accidentally hiding ALL faces instead of back-faces.
        """
        visible_faces = []

        # Culling sign: normally +1; flip to -1 when geometry is reflected
        # (det(M) < 0 ⟺ winding is reversed).
        cull_sign = -1.0 if reflected else 1.0

        for fi, face in enumerate(mesh.faces):
            # --- back-face culling ----------------------------------------
            # Compute the face normal in VIEW space (from the first 3 vertices)
            A_v = view_verts[face[0], :3]
            B_v = view_verts[face[1], :3]
            C_v = view_verts[face[2], :3]
            edge1 = B_v - A_v
            edge2 = C_v - A_v
            normal_view = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal_view)
            if norm_len < 1e-10:
                continue                           # degenerate face, skip
            normal_view /= norm_len

            if camera.projection_mode == "perspective":
                # Camera is at origin in view space.
                # Face visible if normal points TOWARD camera (dot < 0).
                # Multiply by cull_sign to handle reflected winding.
                view_dot = cull_sign * np.dot(normal_view, A_v)
                if view_dot >= 0:
                    continue                       # back face — skip
            else:
                # Orthographic: camera is effectively at infinity along +Z.
                # Face visible if normal's Z component is positive (faces us).
                if cull_sign * normal_view[2] <= 0:
                    continue

            # --- check all vertices are in front of camera ---------------
            if not all(valid_mask[idx] for idx in face):
                continue

            # --- depth key: average view-space Z for painter's sort ------
            avg_z = np.mean([view_verts[idx, 2] for idx in face])

            # --- flat shading (Lambert) -----------------------------------
            # Compute normal in WORLD space for lighting
            A_w = world_verts[face[0], :3]
            B_w = world_verts[face[1], :3]
            C_w = world_verts[face[2], :3]
            e1_w = B_w - A_w
            e2_w = C_w - A_w
            normal_world = np.cross(e1_w, e2_w)
            nw_len = np.linalg.norm(normal_world)
            if nw_len > 1e-10:
                normal_world /= nw_len

            diffuse = max(0.0, np.dot(normal_world, LIGHT_DIR))
            intensity = AMBIENT + (1.0 - AMBIENT) * diffuse

            # Base colour from mesh, or a default grey
            if mesh.face_colors is not None:
                base_color = mesh.face_colors[fi]
            else:
                base_color = (180, 180, 180)

            shaded_color = tuple(
                min(255, int(c * intensity)) for c in base_color
            )

            visible_faces.append((avg_z, fi, face, shaded_color))

        # --- painter's algorithm: sort farthest first (most negative Z first) ---
        visible_faces.sort(key=lambda x: x[0])  # ascending Z = farthest first (Z is negative in view space)

        # --- rasterise each face ------------------------------------------
        for _z, _fi, face, color in visible_faces:
            # Collect 2D screen points for this polygon
            poly = [(int(screen_pts[idx, 0]), int(screen_pts[idx, 1])) for idx in face]
            if len(poly) >= 3:
                pygame.draw.polygon(surface, color, poly)
                # Thin edge outline for visual clarity
                pygame.draw.polygon(surface, EDGE_OUTLINE_COLOR, poly, 1)

    # ==================================================================
    # Axes gizmo
    # ==================================================================

    def _draw_axes(
        self,
        surface: pygame.Surface,
        M: np.ndarray,
        V: np.ndarray,
        P: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """
        Draw the world-space X (red), Y (green), Z (blue) axes.
        They are drawn through the full pipeline so they follow the scene.
        The axes originate at the world origin (0,0,0).
        """
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        x_tip  = np.array([AXIS_LENGTH, 0.0, 0.0, 1.0])
        y_tip  = np.array([0.0, AXIS_LENGTH, 0.0, 1.0])
        z_tip  = np.array([0.0, 0.0, AXIS_LENGTH, 1.0])

        axis_points = np.array([origin, x_tip, y_tip, z_tip])  # (4, 4)

        # Apply view + projection (axes live in world space, skip model matrix)
        clip = (P @ V @ axis_points.T).T  # (4, 4) clip space

        def to_screen(clip_pt: np.ndarray):
            w = clip_pt[3]
            if w <= NEAR_EPSILON:
                return None
            sx = int((clip_pt[0] / w + 1.0) * 0.5 * width)
            sy = int((1.0 - clip_pt[1] / w) * 0.5 * height)
            return (sx, sy)

        o  = to_screen(clip[0])
        px = to_screen(clip[1])
        py = to_screen(clip[2])
        pz = to_screen(clip[3])

        if o and px:
            pygame.draw.line(surface, AXIS_X_COLOR, o, px, 2)
        if o and py:
            pygame.draw.line(surface, AXIS_Y_COLOR, o, py, 2)
        if o and pz:
            pygame.draw.line(surface, AXIS_Z_COLOR, o, pz, 2)

    # ==================================================================
    # Ground grid
    # ==================================================================

    def _draw_grid(
        self,
        surface: pygame.Surface,
        V: np.ndarray,
        P: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """
        Draw a flat XZ grid on y = -1 (just below the object) as a reference.
        Lines are drawn through the full view+projection pipeline.
        """
        def project(pt: np.ndarray):
            """Project a world-space point to a screen pixel, or None if behind."""
            clip_pt = P @ V @ pt
            w = clip_pt[3]
            if w <= NEAR_EPSILON:
                return None
            sx = int((clip_pt[0] / w + 1.0) * 0.5 * width)
            sy = int((1.0 - clip_pt[1] / w) * 0.5 * height)
            return (sx, sy)

        y = -1.0  # grid sits just below the centred object
        lo = -GRID_SIZE
        hi = GRID_SIZE

        # Lines along X direction (varying X, fixed Z)
        z = lo
        while z <= hi + 0.001:
            p1 = project(np.array([lo, y, z, 1.0]))
            p2 = project(np.array([hi, y, z, 1.0]))
            if p1 and p2:
                pygame.draw.line(surface, GRID_COLOR, p1, p2, 1)
            z += GRID_STEP

        # Lines along Z direction (varying Z, fixed X)
        x = lo
        while x <= hi + 0.001:
            p1 = project(np.array([x, y, lo, 1.0]))
            p2 = project(np.array([x, y, hi, 1.0]))
            if p1 and p2:
                pygame.draw.line(surface, GRID_COLOR, p1, p2, 1)
            x += GRID_STEP
