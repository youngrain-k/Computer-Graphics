import taichi as ti

# 初始化后端
ti.init(arch=ti.gpu)

# 物理仿真全局参数
N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_d = 5.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# 质点数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

# 隐式欧拉迭代缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧相关数据场
max_springs = N * N * 4
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())