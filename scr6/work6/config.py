import taichi as ti

# 初始化后端
ti.init(arch=ti.cpu)

# 渲染分辨率
RES = 256

# 图像缓冲区
target_pixels = ti.field(dtype=ti.f32, shape=(RES, RES))
display_pixels = ti.field(dtype=ti.f32, shape=(RES * 2, RES))

# 可微优化变量（开启梯度）
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)

# 场景几何参数
SPHERE_CENTER = ti.Vector([0.5, 0.5, 0.5])
SPHERE_RADIUS = 0.3
TARGET_LIGHT = [0.8, 0.8, 0.2]

# 优化超参数默认值
BETA1 = 0.9
BETA2 = 0.999
LR = 0.02
EPS = 1e-8