import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# 窗口配置
WINDOW_RES = (700, 700)
WINDOW_TITLE = "3D Transformation (Taichi)"

# 相机与投影参数
EYE_POS = ti.Vector([0.0, 0.0, 5.0])
FOV = 45.0
ASPECT_RATIO = 1.0
Z_NEAR = 0.1
Z_FAR = 50.0

# 初始三角形顶点（模型空间坐标）
INIT_VERTICES = [
    [2.0, 0.0, -2.0],
    [0.0, 2.0, -2.0],
    [-2.0, 0.0, -2.0]
]