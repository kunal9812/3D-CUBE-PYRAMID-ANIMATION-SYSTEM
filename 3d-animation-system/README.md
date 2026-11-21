# 3D Cube/Pyramid Animation System

An interactive desktop application that demonstrates core **Computer Graphics** concepts through live 3D object manipulation — built entirely from scratch using Python, NumPy, and Pygame.

> **All 3D mathematics (transformation matrices, camera, projection, culling, depth sorting, shading) is implemented by hand.** Pygame is used only to draw 2D lines and polygons.

---

## Features

- **3D Transformations**: translate, rotate (X/Y/Z), scale — applied via homogeneous 4×4 matrices
- **Rotation Matrices**: separate Rx, Ry, Rz, combined with Rodrigues' formula bonus
- **Homogeneous Coordinates**: every vertex is `[x, y, z, 1]ᵀ`, every transform is a 4×4 matrix
- **Perspective Projection**: FOV-based frustum, perspective divide, and orthographic comparison
- **Orbit Camera**: yaw/pitch/distance control; front, top, side, isometric presets
- **Solid Rendering**: back-face culling, painter's algorithm depth sort, Lambert flat shading
- **Time-based Animations**: auto-rotate (rotation), bounce (translation), scale pulse (scaling)
- **Live HUD**: FPS, object state, and live 4×4 model matrix display
- **Screenshots**: F12 saves to `docs/screenshots/`

---

## Install & Run

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py

# 4. Run tests
pytest -q
```

**Requirements:** Python 3.10+, numpy, pygame, pytest (see `requirements.txt`)

---

## Controls

| Key / Input | Action |
|---|---|
| **Arrow keys** | Translate object X / Y |
| **W / S** | Translate object along Z |
| **Q / E** | Rotate about X axis |
| **A / D** | Rotate about Y axis |
| **Z / C** | Rotate about Z axis |
| **+ / -** | Uniform scale |
| **[ / ]** | Decrease / Increase FOV |
| **Mouse drag (LMB)** | Orbit camera |
| **Mouse wheel** | Zoom (camera distance) |
| **1 2 3 4** | Camera presets: Front / Top / Side / Isometric |
| **P** | Toggle Perspective / Orthographic |
| **O** | Toggle transform order TRS / SRT |
| **Space** | Pause / Resume animations |
| **M** | Toggle auto-rotate |
| **B** | Toggle bounce |
| **N** | Toggle scale pulse |
| **I** | Open "New Shape" panel — enter custom coordinates |
| **T** | Open "Transform" panel — enter exact transform values |
| **Tab** | Switch Cube / Pyramid |
| **F** | Toggle Wireframe / Solid |
| **G** | Toggle grid and axes |
| **R** | Reset object and camera |
| **F12** | Save screenshot |
| **H** | Toggle help overlay |
| **Esc** | Quit |

---

## Project Structure

```
3d-animation-system/
├── main.py                  # entry point: window, event loop
├── requirements.txt
├── README.md
├── CLAUDE.md                # project conventions
├── cg/
│   ├── transforms.py        # ALL 4×4 matrix functions
│   ├── mesh.py              # Mesh dataclass
│   ├── shapes.py            # make_cube(), make_pyramid()
│   ├── scene_object.py      # SceneObject: mesh + model matrix
│   ├── camera.py            # Orbit camera, view + projection matrices
│   ├── renderer.py          # Full rendering pipeline
│   └── animator.py          # Time-based animations
├── ui/
│   ├── input_handler.py     # Keyboard / mouse → actions
│   ├── input_panel.py       # Modal panels for entering coords & transforms
│   └── hud.py               # On-screen text, live matrix panel
├── tests/
│   ├── test_transforms.py
│   ├── test_camera_projection.py
│   ├── test_meshes.py
│   └── test_user_input_features.py
└── docs/
    ├── report.md
    └── screenshots/
```

---

## CG Concepts → Code Map

| CG Concept | File | Key Functions |
|---|---|---|
| Homogeneous coordinates | `cg/transforms.py` | All functions — every matrix is 4×4 |
| Translation matrix | `cg/transforms.py` | `translate(tx, ty, tz)` |
| Scale matrix | `cg/transforms.py` | `scale(sx, sy, sz)` |
| Rotation matrices (Rx, Ry, Rz) | `cg/transforms.py` | `rotate_x/y/z(angle)` |
| Euler composition & order | `cg/transforms.py` | `rotate_xyz()`, `compose()` |
| Rodrigues' rotation formula | `cg/transforms.py` | `rotate_axis(axis, angle)` |
| Perspective projection | `cg/transforms.py` | `perspective(fov, aspect, near, far)` |
| Orthographic projection | `cg/transforms.py` | `orthographic(...)` |
| View / camera matrix | `cg/transforms.py`, `cg/camera.py` | `look_at()`, `Camera.view_matrix()` |
| Model matrix (TRS) | `cg/scene_object.py` | `SceneObject.model_matrix()` |
| Full rendering pipeline | `cg/renderer.py` | `Renderer.draw()` |
| Back-face culling | `cg/renderer.py` | `_draw_solid()` |
| Painter's algorithm | `cg/renderer.py` | `_draw_solid()` depth sort |
| Lambert flat shading | `cg/renderer.py` | `_draw_solid()` lighting |
| Time-based animation | `cg/animator.py` | `AutoRotate`, `Bounce`, `ScalePulse` |
