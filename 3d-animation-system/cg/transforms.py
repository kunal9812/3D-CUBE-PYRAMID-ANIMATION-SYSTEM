"""
cg/transforms.py — 3D Transformation Matrices
==============================================
CG Topic: Homogeneous Coordinates & 3D Transformation Matrices

Convention (important — every other module follows this):
  - Column vectors: a 3D point is represented as a 4×1 column vector [x, y, z, 1]ᵀ.
  - A point is transformed by LEFT-multiplying with the matrix: p' = M @ p.
  - Vertex arrays are shape (N, 4). To transform all N vertices at once:
        world_verts = (M @ verts.T).T
  - Right-handed coordinate system. The camera looks down the -Z axis in view space.
  - Angles are in RADIANS everywhere in this module (the UI converts to/from degrees).
  - Composition order: T @ R @ S  (Scale first, then Rotate, then Translate).
    In NumPy that means S is applied first because p' = (T @ R @ S) @ p.

Why 4×4 matrices?
  Using 4×4 matrices with a 4th coordinate w (the "homogeneous" coordinate) lets us
  express TRANSLATION as a matrix multiply — something you cannot do with 3×3 matrices.
  For a point, w = 1. For a direction vector, w = 0 (so translations don't move vectors).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def identity() -> np.ndarray:
    """
    Return the 4×4 identity matrix.

        | 1 0 0 0 |
        | 0 1 0 0 |
        | 0 0 1 0 |
        | 0 0 0 1 |

    Geometrically: no change — the point stays where it is.
    """
    return np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Return a 4x4 translation matrix that moves a point by (tx, ty, tz).

        | 1  0  0  tx |
        | 0  1  0  ty |
        | 0  0  1  tz |
        | 0  0  0   1 |

    Why this works: multiplying [x, y, z, 1]ᵀ gives [x+tx, y+ty, z+tz, 1]ᵀ.
    Direction vectors with w=0 are NOT affected — as expected, since translating
    a direction makes no physical sense.
    """
    M = np.eye(4, dtype=np.float64)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    """
    Return a 4x4 non-uniform scale matrix.

        | sx  0   0   0 |
        | 0   sy  0   0 |
        | 0   0   sz  0 |
        | 0   0   0   1 |

    Geometrically: stretches or shrinks the object along each axis independently.
    Uniform scale: sx == sy == sz.
    Negative values mirror the object across that axis (be careful — this
    reverses face winding and breaks back-face culling).
    """
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    M[3, 3] = 1.0
    return M


# ---------------------------------------------------------------------------
# Rotation matrices (right-handed, standard sign convention)
# ---------------------------------------------------------------------------

def rotate_x(angle: float) -> np.ndarray:
    """
    Rotation about the X axis by `angle` radians (right-hand rule: thumb along +X,
    fingers curl from +Y toward +Z).

        | 1    0       0    0 |
        | 0   cos(a) -sin(a) 0 |
        | 0   sin(a)  cos(a) 0 |
        | 0    0       0    1 |

    Geometrically: tilts the object forward/backward (pitch).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1,  0,  0,  0],
        [0,  c, -s,  0],
        [0,  s,  c,  0],
        [0,  0,  0,  1],
    ], dtype=np.float64)


def rotate_y(angle: float) -> np.ndarray:
    """
    Rotation about the Y axis by `angle` radians (right-hand rule: thumb along +Y,
    fingers curl from +Z toward +X).

        |  cos(a)  0  sin(a)  0 |
        |    0     1    0     0 |
        | -sin(a)  0  cos(a)  0 |
        |    0     0    0     1 |

    Note the transposed sin signs vs Rx — this follows from the right-hand rule
    for Y, where a positive rotation takes +X toward -Z.

    Geometrically: spins the object left/right (yaw).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [ c,  0,  s,  0],
        [ 0,  1,  0,  0],
        [-s,  0,  c,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)


def rotate_z(angle: float) -> np.ndarray:
    """
    Rotation about the Z axis by `angle` radians (right-hand rule: thumb along +Z,
    fingers curl from +X toward +Y).

        | cos(a) -sin(a)  0  0 |
        | sin(a)  cos(a)  0  0 |
        |   0       0     1  0 |
        |   0       0     0  1 |

    Geometrically: rolls the object clockwise or counter-clockwise (roll).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s,  0,  0],
        [s,  c,  0,  0],
        [0,  0,  1,  0],
        [0,  0,  0,  1],
    ], dtype=np.float64)


def rotate_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    """
    Combined Euler-angle rotation: Rz(az) @ Ry(ay) @ Rx(ax).

    Application order (right-to-left in NumPy):
      1. Rx(ax) — rotate about X first
      2. Ry(ay) — then rotate about Y
      3. Rz(az) — then rotate about Z

    This is the "intrinsic XYZ" or "extrinsic ZYX" convention.

    WARNING — Gimbal lock: when pitch (ax) approaches ±90°, the X and Z axes
    become colinear and the system loses one degree of freedom. The report.md
    discusses this limitation.
    """
    return rotate_z(az) @ rotate_y(ay) @ rotate_x(ax)


def rotate_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues' rotation formula — rotate `angle` radians about an arbitrary unit `axis`.

    For a unit vector n = (nx, ny, nz) and angle θ:

        R = cos(θ)·I + (1-cos(θ))·(n⊗n) + sin(θ)·[n]×

    where [n]× is the skew-symmetric cross-product matrix:
        |  0   -nz   ny |
        |  nz   0   -nx |
        | -ny   nx   0  |

    Geometrically: a single rotation that avoids Gimbal lock.
    Useful as a stretch goal / bonus topic in the viva.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  # ensure unit length
    nx, ny, nz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c

    # Construct the 3×3 Rodrigues matrix
    R3 = np.array([
        [t*nx*nx + c,    t*nx*ny - s*nz, t*nx*nz + s*ny],
        [t*nx*ny + s*nz, t*ny*ny + c,    t*ny*nz - s*nx],
        [t*nx*nz - s*ny, t*ny*nz + s*nx, t*nz*nz + c   ],
    ], dtype=np.float64)

    # Embed in a 4×4 homogeneous matrix
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R3
    return M


# ---------------------------------------------------------------------------
# Projection matrices
# ---------------------------------------------------------------------------

def perspective(
    fov_y: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """
    OpenGL-style perspective projection matrix.

    Parameters
    ----------
    fov_y  : vertical field-of-view in RADIANS.
    aspect : width / height of the viewport.
    near   : distance to near clipping plane (positive, e.g. 0.1).
    far    : distance to far clipping plane (positive, e.g. 100).

    The matrix maps a view-space frustum to a clip-space cube, and sets w = -z_view
    so that after the perspective divide (NDC = clip / w), objects farther away
    appear smaller — this is what creates the perspective effect.

        | f/a   0      0           0        |
        |  0    f      0           0        |
        |  0    0   (f+n)/(n-f)  2fn/(n-f) |
        |  0    0     -1           0        |

    where f = 1 / tan(fov_y / 2)  (the "focal length" of the lens).

    Derivation sketch (similar triangles):
      A point (x, y, z) in view space (z < 0 in front of camera).
      The projected x on the near plane is x_p = (n * x) / (-z).
      Normalised: x_ndc = x_p / (n * tan(fov/2)).
      This simplifies to  x_ndc = x * f / (-z) = (f * x) / (-z).
      Setting clip_x = f * x  and  w = -z  gives  x_ndc = clip_x / w. ✓
    """
    f = 1.0 / np.tan(fov_y / 2.0)
    return np.array([
        [f / aspect, 0,  0,                         0              ],
        [0,          f,  0,                         0              ],
        [0,          0,  (far + near) / (near - far), 2*far*near / (near - far)],
        [0,          0, -1,                         0              ],
    ], dtype=np.float64)


def orthographic(
    left: float, right: float,
    bottom: float, top: float,
    near: float, far: float,
) -> np.ndarray:
    """
    Orthographic projection matrix — maps an axis-aligned box to NDC cube [-1,1]³.

        | 2/(r-l)    0        0      -(r+l)/(r-l) |
        |    0    2/(t-b)     0      -(t+b)/(t-b) |
        |    0       0    -2/(f-n)   -(f+n)/(f-n) |
        |    0       0        0           1        |

    Unlike perspective, there is NO division by w (w stays 1 throughout).
    Therefore parallel lines stay parallel — distant objects appear the same
    size as nearby ones. Good for engineering/CAD views (top, side, front).
    """
    rml = right - left
    tmb = top - bottom
    fmn = far - near
    return np.array([
        [2/rml, 0,     0,      -(right+left)/rml],
        [0,     2/tmb, 0,      -(top+bottom)/tmb],
        [0,     0,     -2/fmn, -(far+near)/fmn  ],
        [0,     0,     0,       1               ],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# View matrix — look_at
# ---------------------------------------------------------------------------

def look_at(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """
    Build a view matrix that positions a camera at `eye`, looking at `target`,
    with the given `up` direction hint.

    Steps:
      1. forward = normalize(target - eye)   — direction camera looks (+Z in world, -Z in view)
      2. right   = normalize(forward × up)   — camera's right direction
      3. true_up = right × forward           — recomputed up (perpendicular to both)

    The view matrix is the inverse of the camera's world-space transform.
    For a rotation matrix R and translation t, the inverse is [Rᵀ | -Rᵀt].

    This gives:
        | rx  ry  rz  -dot(right, eye)    |
        | ux  uy  uz  -dot(true_up, eye)  |
        | -fx -fy -fz  dot(forward, eye)  |
        |  0   0   0   1                  |

    Convention: camera looks down its LOCAL -Z axis (hence the negated forward row).
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-10:
        raise ValueError("eye and target are at the same position")
    forward = forward / norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-10:
        raise ValueError("forward and up vectors are parallel")
    right = right / right_norm

    true_up = np.cross(right, forward)  # already unit length

    M = np.array([
        [ right[0],    right[1],    right[2],   -np.dot(right,   eye)],
        [ true_up[0],  true_up[1],  true_up[2], -np.dot(true_up, eye)],
        [-forward[0], -forward[1], -forward[2],  np.dot(forward,  eye)],
        [ 0,           0,           0,           1                    ],
    ], dtype=np.float64)
    return M


# ---------------------------------------------------------------------------
# Reflection matrices
# ---------------------------------------------------------------------------

def reflect(plane: str) -> np.ndarray:
    """
    Return a 4×4 reflection matrix that mirrors geometry across a coordinate plane.

    Parameters
    ----------
    plane : one of "XY", "YZ", or "XZ" (case-insensitive).

    Mathematical background
    -----------------------
    A reflection across a plane simply negates the axis that is PERPENDICULAR
    to that plane:

      Reflect across XY  → negate Z  →  diag(+1, +1, -1, +1)
      Reflect across YZ  → negate X  →  diag(-1, +1, +1, +1)
      Reflect across XZ  → negate Y  →  diag(+1, -1, +1, +1)

    These are diagonal matrices, so they're trivial to construct but have an
    important geometric property: they FLIP face winding (CCW → CW), which
    means back-face culling must compensate.  The renderer handles this by
    negating the culling dot-product test when a reflection is active.

    Composing reflections:
      Two reflections across the same plane cancel out (R² = I).
      Two reflections across *different* planes is equivalent to a 180° rotation.

    Returns a (4, 4) float64 ndarray.
    """
    M = np.eye(4, dtype=np.float64)
    p = plane.upper()
    if p == "XY":
        M[2, 2] = -1.0   # negate Z
    elif p == "YZ":
        M[0, 0] = -1.0   # negate X
    elif p == "XZ":
        M[1, 1] = -1.0   # negate Y
    else:
        raise ValueError(f"Unknown reflection plane: {plane!r}. Use 'XY', 'YZ', or 'XZ'.")
    return M


REFLECTION_PLANES = ["none", "XY", "YZ", "XZ"]   # cycle order for the toggle key


# ---------------------------------------------------------------------------
# Model-matrix composition
# ---------------------------------------------------------------------------

def compose(
    order: str,
    T: np.ndarray,
    R: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    Compose a model matrix from Translation (T), Rotation (R), Scale (S) matrices.

    Parameters
    ----------
    order : "TRS" (default, physically sensible) or "SRT" (for demonstration).
    T, R, S : 4×4 matrices from translate(), rotate_*(), scale().

    Returns the combined 4×4 model matrix.

    Why order matters:
      "TRS" means: first scale the mesh, then rotate it, then translate it to its
      world position. This is the standard used in almost every 3D engine.

      "SRT" means: first translate, then rotate around the world origin, then scale
      the whole thing — which gives a very different result when T ≠ identity.

    NumPy note: because we use column vectors (p' = M @ p), the rightmost matrix
    is applied first. So  T @ R @ S  applies S, then R, then T. ✓
    """
    if order == "TRS":
        return T @ R @ S
    elif order == "SRT":
        return S @ R @ T
    else:
        raise ValueError(f"Unknown compose order: {order!r}. Use 'TRS' or 'SRT'.")
