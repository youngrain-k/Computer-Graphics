import taichi as ti

# 初始化渲染后端
ti.init(arch=ti.gpu)

# 窗口分辨率
RES_X = 800
RES_Y = 600

# 像素画布
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

# Phong光照全局参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())