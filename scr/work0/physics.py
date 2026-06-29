# physics.py
import taichi as ti
import math
from config import *

@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o