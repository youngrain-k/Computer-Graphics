import taichi as ti

# 初始化GPU后端
ti.init(arch=ti.gpu)

# 窗口分辨率
RES_X = 800
RES_Y = 600

# 像素渲染画布
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

# 光源位置交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())

# 光线最大弹射次数
max_bounces = ti.field(ti.i32, shape=())

# 材质类型常量
MAT_DIFFUSE = 0
MAT_MIRROR = 1