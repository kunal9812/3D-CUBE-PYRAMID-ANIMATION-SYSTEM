"""
tests/test_user_input_features.py
===================================
Automated tests for all user-input features added in the requested_changes:

  1. Parameterised make_cube()   — center, size, custom_vertices
  2. Parameterised make_pyramid() — center, base_size, height, custom_vertices
  3. SceneObject new transform helpers — apply_translation, apply_rotation_deg,
     set_position, set_rotation_deg, set_scale
  4. NewShapePanel — field defaults, char input, backspace, Tab navigation,
     shape toggling, submission, cancellation
  5. TransformPanel — field defaults pre-filled from obj, submission with
     expected result dict, cancellation
  6. main.py helper functions — _apply_new_shape, _apply_transform
  7. InputHandler — new 7-tuple return value, panel routing (I / T keys),
     event suppression while panel open
  8. HUD — draw() accepts panel_active / panel_kind without raising
  9. Backward-compatibility — no-arg calls to make_cube / make_pyramid still
     produce the original geometry
"""

import math
import numpy as np
import pytest
import pygame

# ---------------------------------------------------------------------------
# Bring in all modules under test (no display needed for most tests)
# ---------------------------------------------------------------------------
from cg.shapes import make_cube, make_pyramid
from cg.scene_object import SceneObject
from ui.input_panel import NewShapePanel, TransformPanel, _parse_float


# ===========================================================================
# 1.  make_cube() — parameterised
# ===========================================================================

class TestMakeCubeParameterised:

    def test_no_args_gives_original_geometry(self):
        """Backward compat: 8 verts at ±1, centered at origin."""
        m = make_cube()
        assert m.num_vertices == 8
        assert m.num_edges == 12
        assert m.num_faces == 6
        xyz = m.vertices[:, :3]
        assert np.allclose(xyz.mean(axis=0), [0, 0, 0], atol=1e-9)
        assert np.allclose(np.abs(xyz), 1.0)

    def test_custom_center_shifts_centroid(self):
        m = make_cube(center=(3, -1, 2))
        centroid = m.vertices[:, :3].mean(axis=0)
        assert np.allclose(centroid, [3, -1, 2], atol=1e-9)

    def test_custom_size_affects_extents(self):
        m = make_cube(center=(0, 0, 0), size=(4, 2, 6))
        xyz = m.vertices[:, :3]
        assert np.isclose(xyz[:, 0].max() - xyz[:, 0].min(), 4.0)
        assert np.isclose(xyz[:, 1].max() - xyz[:, 1].min(), 2.0)
        assert np.isclose(xyz[:, 2].max() - xyz[:, 2].min(), 6.0)

    def test_all_w_equal_one(self):
        m = make_cube(center=(5, 5, 5), size=(3, 3, 3))
        assert np.all(m.vertices[:, 3] == 1.0)

    def test_custom_vertices_3col_accepted(self):
        verts3 = np.array([
            [-2, -1,  2], [ 2, -1,  2], [-2, -1, -2], [ 2, -1, -2],
            [-2,  1,  2], [ 2,  1,  2], [-2,  1, -2], [ 2,  1, -2],
        ], dtype=float)
        m = make_cube(custom_vertices=verts3)
        assert m.vertices.shape == (8, 4)
        assert np.all(m.vertices[:, 3] == 1.0)
        assert np.allclose(m.vertices[:, :3], verts3)

    def test_custom_vertices_4col_accepted(self):
        verts4 = np.ones((8, 4))
        verts4[:, :3] = np.random.default_rng(42).uniform(-3, 3, (8, 3))
        m = make_cube(custom_vertices=verts4)
        assert m.vertices.shape == (8, 4)

    def test_custom_vertices_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="custom_vertices"):
            make_cube(custom_vertices=np.zeros((7, 3)))

    def test_face_count_unchanged_with_custom_verts(self):
        m = make_cube(custom_vertices=np.random.rand(8, 3))
        assert m.num_faces == 6
        assert m.num_edges == 12

    def test_default_size_matches_no_arg_call(self):
        """make_cube() and make_cube(center=(0,0,0), size=(2,2,2)) must be identical."""
        m_default = make_cube()
        m_explicit = make_cube(center=(0, 0, 0), size=(2, 2, 2))
        assert np.allclose(m_default.vertices, m_explicit.vertices)


# ===========================================================================
# 2.  make_pyramid() — parameterised
# ===========================================================================

class TestMakePyramidParameterised:

    def test_no_args_gives_original_geometry(self):
        m = make_pyramid()
        assert m.num_vertices == 5
        assert m.num_edges == 8
        assert m.num_faces == 5

    def test_custom_center_shifts_base_vertices(self):
        m = make_pyramid(center=(1, 2, 3), base_size=(2, 2), height=2)
        base_y = m.vertices[:4, 1]           # first 4 verts are base
        apex_y = m.vertices[4, 1]            # last vert is apex
        assert np.allclose(base_y, 2 - 1, atol=1e-9)   # center_y - height/2
        assert np.isclose(apex_y, 2 + 1)                # center_y + height/2

    def test_height_controls_apex_elevation(self):
        m = make_pyramid(center=(0, 0, 0), base_size=(2, 2), height=6)
        apex_y = m.vertices[4, 1]
        assert np.isclose(apex_y, 3.0)       # height/2 above centre

    def test_base_size_controls_extents(self):
        m = make_pyramid(center=(0, 0, 0), base_size=(4, 6), height=2)
        base_x_range = m.vertices[:4, 0].max() - m.vertices[:4, 0].min()
        base_z_range = m.vertices[:4, 2].max() - m.vertices[:4, 2].min()
        assert np.isclose(base_x_range, 4.0)
        assert np.isclose(base_z_range, 6.0)

    def test_all_w_equal_one(self):
        m = make_pyramid(center=(1, 1, 1), base_size=(3, 3), height=3)
        assert np.all(m.vertices[:, 3] == 1.0)

    def test_custom_vertices_3col_accepted(self):
        verts3 = np.array([
            [-1, -1,  1], [ 1, -1,  1], [ 1, -1, -1], [-1, -1, -1], [0, 2, 0]
        ], dtype=float)
        m = make_pyramid(custom_vertices=verts3)
        assert m.vertices.shape == (5, 4)
        assert np.allclose(m.vertices[:, :3], verts3)

    def test_custom_vertices_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="custom_vertices"):
            make_pyramid(custom_vertices=np.zeros((6, 3)))

    def test_default_matches_explicit_call(self):
        m1 = make_pyramid()
        m2 = make_pyramid(center=(0, 0, 0), base_size=(2, 2), height=2)
        assert np.allclose(m1.vertices, m2.vertices)


# ===========================================================================
# 3.  SceneObject new transform helpers
# ===========================================================================

class TestSceneObjectHelpers:

    @pytest.fixture()
    def obj(self):
        return SceneObject(make_cube())

    # apply_translation
    def test_apply_translation_adds_offset(self, obj):
        obj.position = np.array([1.0, 2.0, 3.0])
        obj.translate_by(0.5, -1.0, 2.0)
        assert np.allclose(obj.position, [1.5, 1.0, 5.0])

    def test_apply_translation_zero_no_change(self, obj):
        obj.position = np.array([3.0, -1.0, 0.5])
        obj.translate_by(0, 0, 0)
        assert np.allclose(obj.position, [3.0, -1.0, 0.5])

    def test_apply_translation_negative(self, obj):
        obj.translate_by(-5, -5, -5)
        assert np.allclose(obj.position, [-5, -5, -5])

    # apply_rotation_deg
    def test_apply_rotation_deg_converts_to_radians(self, obj):
        obj.rotation = np.zeros(3)
        obj.apply_rotation_deg(90, 0, 0)
        assert np.isclose(obj.rotation[0], math.radians(90))
        assert np.isclose(obj.rotation[1], 0)
        assert np.isclose(obj.rotation[2], 0)

    def test_apply_rotation_deg_is_additive(self, obj):
        obj.rotation = np.array([math.radians(30), 0.0, 0.0])
        obj.apply_rotation_deg(60, 0, 0)
        assert np.isclose(obj.rotation[0], math.radians(90))

    def test_apply_rotation_deg_negative(self, obj):
        obj.apply_rotation_deg(-45, 0, 0)
        assert np.isclose(obj.rotation[0], math.radians(-45))

    # set_position
    def test_set_position_overwrites(self, obj):
        obj.position = np.array([99.0, 99.0, 99.0])
        obj.set_position(1, 2, 3)
        assert np.allclose(obj.position, [1, 2, 3])

    def test_set_position_independent_of_previous(self, obj):
        obj.translate_by(100, 100, 100)
        obj.set_position(0, 0, 0)
        assert np.allclose(obj.position, [0, 0, 0])

    # set_rotation_deg
    def test_set_rotation_deg_converts(self, obj):
        obj.set_rotation_deg(90, 45, 30)
        assert np.isclose(obj.rotation[0], math.radians(90))
        assert np.isclose(obj.rotation[1], math.radians(45))
        assert np.isclose(obj.rotation[2], math.radians(30))

    def test_set_rotation_deg_overwrites(self, obj):
        obj.rotation = np.array([1.0, 2.0, 3.0])
        obj.set_rotation_deg(0, 0, 0)
        assert np.allclose(obj.rotation, [0, 0, 0])

    # set_scale
    def test_set_scale_sets_exact_values(self, obj):
        obj.set_scale(2.0, 3.0, 0.5)
        assert np.allclose(obj.scale, [2.0, 3.0, 0.5])

    def test_set_scale_clamps_to_min(self, obj):
        obj.set_scale(0.0, 0.0, 0.0)
        assert np.all(obj.scale >= 0.1)

    def test_set_scale_clamps_to_max(self, obj):
        obj.set_scale(999, 999, 999)
        assert np.all(obj.scale <= 5.0)

    def test_set_scale_negative_clamped_to_min(self, obj):
        obj.set_scale(-1, -1, -1)
        assert np.all(obj.scale >= 0.1)

    # model_matrix still correct after helpers
    def test_model_matrix_reflects_set_position(self, obj):
        obj.set_position(5, 0, 0)
        M = obj.model_matrix()
        # Translation column of TRS matrix
        assert np.isclose(M[0, 3], 5.0)
        assert np.isclose(M[1, 3], 0.0)
        assert np.isclose(M[2, 3], 0.0)

    def test_model_matrix_reflects_set_scale(self, obj):
        obj.set_rotation_deg(0, 0, 0)
        obj.set_position(0, 0, 0)
        obj.set_scale(3, 2, 1)
        M = obj.model_matrix()
        assert np.isclose(M[0, 0], 3.0)
        assert np.isclose(M[1, 1], 2.0)
        assert np.isclose(M[2, 2], 1.0)


# ===========================================================================
# 4.  NewShapePanel — UI logic (no display window needed)
# ===========================================================================

class TestNewShapePanel:

    @pytest.fixture(autouse=True)
    def init_pygame(self):
        pygame.display.init()
        pygame.font.init()
        yield
        pygame.quit()

    @pytest.fixture()
    def panel(self):
        return NewShapePanel()

    # Initial state
    def test_initial_shape_is_cube(self, panel):
        assert panel._SHAPES[panel._shape_idx] == "cube"

    def test_initial_not_submitted(self, panel):
        assert not panel.submitted

    def test_initial_not_cancelled(self, panel):
        assert not panel.cancelled

    def test_initial_field_count_cube(self, panel):
        # 3 center + 3 size + 3 color = 9
        assert len(panel._fields) == 9

    def test_initial_field_count_pyramid(self, panel):
        panel._shape_idx = 1
        panel._rebuild_fields()
        assert len(panel._fields) == 9

    # Shape toggle with Left/Right arrow
    def _make_key_event(self, key):
        """Synthesise a minimal KEYDOWN event."""
        event = pygame.event.Event(pygame.KEYDOWN, {
            "key": key,
            "mod": 0,
            "unicode": "",
            "scancode": 0,
        })
        return event

    def test_right_arrow_switches_to_pyramid(self, panel):
        panel.handle_event(self._make_key_event(pygame.K_RIGHT))
        assert panel._SHAPES[panel._shape_idx] == "pyramid"

    def test_left_arrow_toggles_back_to_cube(self, panel):
        panel.handle_event(self._make_key_event(pygame.K_RIGHT))  # → pyramid
        panel.handle_event(self._make_key_event(pygame.K_LEFT))   # → cube
        assert panel._SHAPES[panel._shape_idx] == "cube"

    # Character input
    def test_digit_appended_to_active_field(self, panel):
        panel._active = 0
        panel._fields[0].text = ""
        panel.handle_event(self._make_key_event_char("5"))
        assert panel._fields[0].text == "5"

    def test_multiple_digits_accumulate(self, panel):
        panel._active = 0
        panel._fields[0].text = ""
        for ch in "123":
            panel.handle_event(self._make_key_event_char(ch))
        assert panel._fields[0].text == "123"

    def test_backspace_removes_last_char(self, panel):
        panel._fields[0].text = "42"
        panel._active = 0
        panel.handle_event(self._make_key_event(pygame.K_BACKSPACE))
        assert panel._fields[0].text == "4"

    def test_minus_at_start_accepted(self, panel):
        panel._fields[0].text = "0"
        panel._active = 0
        panel.handle_event(self._make_key_event_char("-"))
        assert panel._fields[0].text == "-"

    def test_decimal_point_accepted(self, panel):
        panel._fields[0].text = "1"
        panel._active = 0
        panel.handle_event(self._make_key_event_char("."))
        assert "." in panel._fields[0].text

    def test_duplicate_decimal_rejected(self, panel):
        panel._fields[0].text = "1.5"
        panel._active = 0
        panel.handle_event(self._make_key_event_char("."))
        assert panel._fields[0].text.count(".") == 1

    # Tab navigation
    def test_tab_advances_active_field(self, panel):
        panel._active = 0
        panel.handle_event(self._make_key_event(pygame.K_TAB))
        assert panel._active == 1

    def test_tab_does_not_overflow(self, panel):
        panel._active = len(panel._fields) - 1
        panel.handle_event(self._make_key_event(pygame.K_TAB))
        assert panel._active == len(panel._fields) - 1

    def test_shift_tab_goes_back(self, panel):
        panel._active = 3
        event = pygame.event.Event(pygame.KEYDOWN, {
            "key": pygame.K_TAB,
            "mod": pygame.KMOD_SHIFT,
            "unicode": "",
            "scancode": 0,
        })
        panel.handle_event(event)
        assert panel._active == 2

    def test_shift_tab_does_not_underflow(self, panel):
        panel._active = 0
        event = pygame.event.Event(pygame.KEYDOWN, {
            "key": pygame.K_TAB,
            "mod": pygame.KMOD_SHIFT,
            "unicode": "",
            "scancode": 0,
        })
        panel.handle_event(event)
        assert panel._active == 0

    # Escape → cancelled
    def test_escape_sets_cancelled(self, panel):
        panel.handle_event(self._make_key_event(pygame.K_ESCAPE))
        assert panel.cancelled
        assert not panel.submitted

    # Enter on last field → submitted
    def test_enter_on_last_field_submits(self, panel):
        panel._active = len(panel._fields) - 1
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        assert panel.submitted
        assert not panel.cancelled

    def test_enter_on_non_last_field_advances(self, panel):
        panel._active = 0
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        assert panel._active == 1
        assert not panel.submitted

    # Result dict for cube
    def test_cube_result_has_expected_keys(self, panel):
        panel._active = len(panel._fields) - 1
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        r = panel.result
        for key in ("shape", "cx", "cy", "cz", "width", "height", "depth", "r", "g", "b"):
            assert key in r, f"Missing key: {key}"

    def test_cube_result_shape_is_cube(self, panel):
        panel._active = len(panel._fields) - 1
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        assert panel.result["shape"] == "cube"

    # Result dict for pyramid
    def test_pyramid_result_has_expected_keys(self, panel):
        panel.handle_event(self._make_key_event(pygame.K_RIGHT))  # switch to pyramid
        panel._active = len(panel._fields) - 1
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        r = panel.result
        for key in ("shape", "cx", "cy", "cz", "base_w", "base_d", "pyramid_height", "r", "g", "b"):
            assert key in r, f"Missing key: {key}"

    # draw() does not raise
    def test_draw_does_not_raise(self, panel):
        surf = pygame.display.set_mode((400, 300))
        panel.draw(surf)   # should not raise

    # --- helpers ---
    def _make_key_event_char(self, char):
        char_to_key = {
            "0": pygame.K_0, "1": pygame.K_1, "2": pygame.K_2,
            "3": pygame.K_3, "4": pygame.K_4, "5": pygame.K_5,
            "6": pygame.K_6, "7": pygame.K_7, "8": pygame.K_8,
            "9": pygame.K_9, ".": pygame.K_PERIOD, "-": pygame.K_MINUS,
        }
        return pygame.event.Event(pygame.KEYDOWN, {
            "key": char_to_key.get(char, 0),
            "mod": 0,
            "unicode": char,
            "scancode": 0,
        })


# ===========================================================================
# 5.  TransformPanel
# ===========================================================================

class TestTransformPanel:

    @pytest.fixture(autouse=True)
    def init_pygame(self):
        pygame.display.init()
        pygame.font.init()
        yield
        pygame.quit()

    @pytest.fixture()
    def obj(self):
        o = SceneObject(make_cube())
        o.set_scale(2.0, 3.0, 0.5)
        return o

    @pytest.fixture()
    def panel(self, obj):
        return TransformPanel(obj=obj)

    def _make_key_event(self, key, mod=0, char=""):
        return pygame.event.Event(pygame.KEYDOWN, {
            "key": key, "mod": mod, "unicode": char, "scancode": 0,
        })

    # Pre-fill from object
    def test_scale_fields_prefilled_from_object(self, panel, obj):
        sx_field = panel._fields[6]
        sy_field = panel._fields[7]
        sz_field = panel._fields[8]
        assert np.isclose(float(sx_field.text), obj.scale[0], atol=1e-3)
        assert np.isclose(float(sy_field.text), obj.scale[1], atol=1e-3)
        assert np.isclose(float(sz_field.text), obj.scale[2], atol=1e-3)

    def test_total_field_count(self, panel):
        assert len(panel._fields) == 9

    # Escape
    def test_escape_cancels(self, panel):
        panel.handle_event(self._make_key_event(pygame.K_ESCAPE))
        assert panel.cancelled
        assert not panel.submitted

    # Enter on last field submits
    def test_enter_on_last_submits(self, panel):
        panel._active = 8
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        assert panel.submitted

    # Result dict keys
    def test_result_has_all_keys(self, panel):
        panel._active = 8
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        r = panel.result
        for key in ("tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"):
            assert key in r, f"Missing key: {key}"

    def test_default_translation_is_zero(self, panel):
        panel._active = 8
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        assert panel.result["tx"] == 0.0
        assert panel.result["ty"] == 0.0
        assert panel.result["tz"] == 0.0

    def test_default_rotation_is_zero(self, panel):
        panel._active = 8
        panel.handle_event(self._make_key_event(pygame.K_RETURN))
        assert panel.result["rx"] == 0.0
        assert panel.result["ry"] == 0.0
        assert panel.result["rz"] == 0.0

    # draw() does not raise
    def test_draw_does_not_raise(self, panel):
        surf = pygame.display.set_mode((400, 300))
        panel.draw(surf)


# ===========================================================================
# 6.  main.py helper functions (_apply_new_shape, _apply_transform)
# ===========================================================================

# Import directly from main

class TestMainHelpers:

    @pytest.fixture()
    def obj(self):
        return SceneObject(make_cube())

    def _import_helpers(self):
        """Import _apply_new_shape and _apply_transform from main without
        running main() (Pygame display not needed)."""
        import main as m
        return m._apply_new_shape, m._apply_transform

    # _apply_new_shape — cube
    def test_apply_new_shape_cube_swaps_mesh(self, obj):
        _apply_new_shape, _ = self._import_helpers()
        result = {
            "shape": "cube",
            "cx": 0, "cy": 0, "cz": 0,
            "width": 4, "height": 2, "depth": 3,
            "r": 100, "g": 150, "b": 200,
        }
        _apply_new_shape(result, obj)
        assert obj.mesh.num_vertices == 8
        # Extents reflect user-entered size
        xyz = obj.mesh.vertices[:, :3]
        assert np.isclose(xyz[:, 0].max() - xyz[:, 0].min(), 4.0)

    def test_apply_new_shape_cube_resets_transform(self, obj):
        _apply_new_shape, _ = self._import_helpers()
        obj.set_position(99, 99, 99)
        obj.set_scale(5, 5, 5)
        result = {
            "shape": "cube",
            "cx": 0, "cy": 0, "cz": 0,
            "width": 2, "height": 2, "depth": 2,
            "r": 0, "g": 0, "b": 0,
        }
        _apply_new_shape(result, obj)
        assert np.allclose(obj.position, [0, 0, 0])
        assert np.allclose(obj.scale, [1, 1, 1])

    # _apply_new_shape — pyramid
    def test_apply_new_shape_pyramid_swaps_mesh(self, obj):
        _apply_new_shape, _ = self._import_helpers()
        result = {
            "shape": "pyramid",
            "cx": 0, "cy": 0, "cz": 0,
            "base_w": 3, "base_d": 3, "pyramid_height": 4,
            "r": 200, "g": 100, "b": 50,
        }
        _apply_new_shape(result, obj)
        assert obj.mesh.num_vertices == 5

    def test_apply_new_shape_uses_center(self, obj):
        _apply_new_shape, _ = self._import_helpers()
        result = {
            "shape": "cube",
            "cx": 5, "cy": -2, "cz": 1,
            "width": 2, "height": 2, "depth": 2,
            "r": 0, "g": 0, "b": 0,
        }
        _apply_new_shape(result, obj)
        centroid = obj.mesh.vertices[:, :3].mean(axis=0)
        assert np.allclose(centroid, [5, -2, 1], atol=1e-9)

    # _apply_transform
    def test_apply_transform_translation_additive(self, obj):
        _, _apply_transform = self._import_helpers()
        obj.set_position(1, 2, 3)
        result = {"tx": 1, "ty": -2, "tz": 5, "rx": 0, "ry": 0, "rz": 0,
                  "sx": 1, "sy": 1, "sz": 1}
        _apply_transform(result, obj)
        assert np.allclose(obj.position, [2, 0, 8])

    def test_apply_transform_rotation_additive_in_radians(self, obj):
        _, _apply_transform = self._import_helpers()
        obj.rotation = np.zeros(3)
        result = {"tx": 0, "ty": 0, "tz": 0, "rx": 90, "ry": 0, "rz": 0,
                  "sx": 1, "sy": 1, "sz": 1}
        _apply_transform(result, obj)
        assert np.isclose(obj.rotation[0], math.radians(90))

    def test_apply_transform_scale_absolute(self, obj):
        _, _apply_transform = self._import_helpers()
        result = {"tx": 0, "ty": 0, "tz": 0, "rx": 0, "ry": 0, "rz": 0,
                  "sx": 2, "sy": 3, "sz": 0.5}
        _apply_transform(result, obj)
        assert np.allclose(obj.scale, [2, 3, 0.5])

    def test_apply_transform_zero_scale_clamped(self, obj):
        _, _apply_transform = self._import_helpers()
        result = {"tx": 0, "ty": 0, "tz": 0, "rx": 0, "ry": 0, "rz": 0,
                  "sx": 0, "sy": 0, "sz": 0}
        _apply_transform(result, obj)
        assert np.all(obj.scale >= 0.1)   # clamped by set_scale


# ===========================================================================
# 7.  InputHandler — new 7-tuple return and panel routing
# ===========================================================================

class TestInputHandlerNewFeatures:

    @pytest.fixture(autouse=True)
    def init_pygame(self):
        pygame.display.init()
        pygame.font.init()
        yield
        pygame.quit()

    @pytest.fixture()
    def setup(self):
        from ui.input_handler import InputHandler
        from cg.camera import Camera
        from cg.animator import Animator
        handler = InputHandler()
        obj = SceneObject(make_cube())
        cam = Camera()
        anim = Animator()
        surface = pygame.display.set_mode((400, 300))
        shapes = [make_cube(), make_pyramid()]
        return handler, obj, cam, anim, surface, shapes

    def _call(self, handler, obj, cam, anim, surface, shapes, events, panel=None):
        return handler.process_events(
            events=events,
            obj=obj,
            camera=cam,
            animator=anim,
            shapes=shapes,
            current_shape_idx=0,
            wireframe=False,
            show_grid=True,
            show_help=False,
            surface=surface,
            active_panel=panel,
        )

    def _key_event(self, key, char=""):
        return pygame.event.Event(pygame.KEYDOWN, {
            "key": key, "mod": 0, "unicode": char, "scancode": 0,
        })

    # Return tuple has 7 elements
    def test_returns_7_tuple(self, setup):
        handler, obj, cam, anim, surface, shapes = setup
        result = self._call(handler, obj, cam, anim, surface, shapes, [])
        assert len(result) == 7

    # I key → open_new_shape = True
    def test_i_key_sets_open_new_shape(self, setup):
        handler, obj, cam, anim, surface, shapes = setup
        events = [self._key_event(pygame.K_i)]
        result = self._call(handler, obj, cam, anim, surface, shapes, events)
        _, _, _, _, _, open_new_shape, open_transform = result
        assert open_new_shape is True
        assert open_transform is False

    # T key → open_transform = True
    def test_t_key_sets_open_transform(self, setup):
        handler, obj, cam, anim, surface, shapes = setup
        events = [self._key_event(pygame.K_t)]
        result = self._call(handler, obj, cam, anim, surface, shapes, events)
        _, _, _, _, _, open_new_shape, open_transform = result
        assert open_transform is True
        assert open_new_shape is False

    # Neither I nor T → both False
    def test_no_panel_key_both_false(self, setup):
        handler, obj, cam, anim, surface, shapes = setup
        events = [self._key_event(pygame.K_g)]
        result = self._call(handler, obj, cam, anim, surface, shapes, events)
        _, _, _, _, _, open_new_shape, open_transform = result
        assert open_new_shape is False
        assert open_transform is False

    # When panel is active, I/T should still return False (panel blocks)
    def test_panel_open_suppresses_i_key(self, setup):
        handler, obj, cam, anim, surface, shapes = setup
        dummy_panel = NewShapePanel()
        events = [self._key_event(pygame.K_i)]
        result = self._call(handler, obj, cam, anim, surface, shapes, events, panel=dummy_panel)
        _, _, _, _, _, open_new_shape, _ = result
        assert open_new_shape is False   # panel was open, event was consumed by it

    # Esc with no panel open → quit
    def test_esc_no_panel_quits(self, setup):
        handler, obj, cam, anim, surface, shapes = setup
        events = [self._key_event(pygame.K_ESCAPE)]
        result = self._call(handler, obj, cam, anim, surface, shapes, events)
        quit_req = result[0]
        assert quit_req is True

    # process_held_keys suppressed when panel_active=True
    def test_held_keys_suppressed_with_panel_active(self, setup):
        handler, obj, cam, anim, surface, shapes = setup
        # Manually press LEFT key
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {
            "key": pygame.K_LEFT, "mod": 0, "unicode": "", "scancode": 0,
        }))
        pos_before = obj.position.copy()
        handler.process_held_keys(0.1, obj, cam, panel_active=True)
        assert np.allclose(obj.position, pos_before)   # should NOT have moved


# ===========================================================================
# 8.  HUD — draw() with new parameters
# ===========================================================================

class TestHUDNewParameters:

    @pytest.fixture(autouse=True)
    def init_pygame(self):
        pygame.display.init()
        pygame.font.init()
        yield
        pygame.quit()

    def test_draw_with_panel_active_true(self):
        from ui.hud import HUD
        from cg.camera import Camera
        from cg.animator import Animator
        hud = HUD()
        surface = pygame.display.set_mode((800, 600))
        obj = SceneObject(make_cube())
        cam = Camera()
        anim = Animator()
        # Should not raise with either panel_kind
        hud.draw(surface, obj, cam, anim, 60.0, "Cube", False, False,
                 panel_active=True, panel_kind="new_shape")
        hud.draw(surface, obj, cam, anim, 60.0, "Cube", False, False,
                 panel_active=True, panel_kind="transform")

    def test_draw_without_panel_active(self):
        from ui.hud import HUD
        from cg.camera import Camera
        from cg.animator import Animator
        hud = HUD()
        surface = pygame.display.set_mode((800, 600))
        obj = SceneObject(make_cube())
        cam = Camera()
        anim = Animator()
        hud.draw(surface, obj, cam, anim, 60.0, "Pyramid", True, True,
                 panel_active=False, panel_kind="")

    def test_draw_old_signature_still_works(self):
        """panel_active / panel_kind default to False/'', old callers unaffected."""
        from ui.hud import HUD
        from cg.camera import Camera
        from cg.animator import Animator
        hud = HUD()
        surface = pygame.display.set_mode((800, 600))
        obj = SceneObject(make_cube())
        cam = Camera()
        anim = Animator()
        # Calling without panel kwargs must NOT raise
        hud.draw(surface, obj, cam, anim, 30.0, "Cube", False, False)


# ===========================================================================
# 9.  Backward compatibility — no-arg shape calls
# ===========================================================================

class TestBackwardCompatibility:

    def test_make_cube_no_args_vertex_coords(self):
        """Original cube: all coords exactly ±1 (side = 2, centered at origin)."""
        m = make_cube()
        xyz = m.vertices[:, :3]
        assert set(xyz[:, 0].tolist()) == {-1.0, 1.0}
        assert set(xyz[:, 1].tolist()) == {-1.0, 1.0}
        assert set(xyz[:, 2].tolist()) == {-1.0, 1.0}

    def test_make_pyramid_no_args_apex_at_plus_one(self):
        m = make_pyramid()
        apex = m.vertices[4, :3]
        assert np.allclose(apex, [0, 1, 0])

    def test_make_pyramid_no_args_base_at_minus_one(self):
        m = make_pyramid()
        base_y = m.vertices[:4, 1]
        assert np.all(base_y == -1.0)

    def test_scene_object_reset_still_works(self):
        obj = SceneObject(make_cube())
        obj.translate_by(5, 5, 5)
        obj.apply_rotation_deg(90, 45, 30)
        obj.set_scale(3, 3, 3)
        obj.reset()
        assert np.allclose(obj.position, [0, 0, 0])
        assert np.allclose(obj.rotation, [0, 0, 0])
        assert np.allclose(obj.scale, [1, 1, 1])

    def test_existing_translate_by_still_works(self):
        obj = SceneObject(make_cube())
        obj.translate_by(1, 2, 3)
        assert np.allclose(obj.position, [1, 2, 3])

    def test_existing_rotate_by_still_works(self):
        obj = SceneObject(make_cube())
        obj.rotate_by(0.1, 0.2, 0.3)
        assert np.allclose(obj.rotation, [0.1, 0.2, 0.3])

    def test_existing_scale_by_still_works(self):
        obj = SceneObject(make_cube())
        obj.scale_by(0.5)
        assert np.allclose(obj.scale, [1.5, 1.5, 1.5])


# ===========================================================================
# 10.  _parse_float helper
# ===========================================================================

class TestParseFloat:

    def test_integer_string(self):
        assert _parse_float("42") == 42.0

    def test_decimal_string(self):
        assert _parse_float("3.14") == pytest.approx(3.14)

    def test_negative_string(self):
        assert _parse_float("-5") == -5.0

    def test_negative_decimal(self):
        assert _parse_float("-1.5") == pytest.approx(-1.5)

    def test_empty_string_returns_default(self):
        assert _parse_float("", default=99.0) == 99.0

    def test_minus_only_returns_default(self):
        assert _parse_float("-", default=0.0) == 0.0

    def test_dot_only_returns_default(self):
        assert _parse_float(".", default=0.0) == 0.0

    def test_invalid_string_returns_default(self):
        assert _parse_float("abc", default=7.0) == 7.0

    def test_zero(self):
        assert _parse_float("0") == 0.0
