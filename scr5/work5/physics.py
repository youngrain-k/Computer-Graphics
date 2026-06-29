import taichi as ti
from config import *

# 向量归一化
@ti.func
def normalize(v):
    return v / v.norm(1e-5)

# 光线反射计算
@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# 球体光线求交
@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

# 地面平面求交
@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

# 场景整体相交检测
@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 左侧漫反射红球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.8, 0.1, 0.1])
        hit_mat = MAT_DIFFUSE

    # 右侧镜面银球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 棋盘格地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

# 光线追踪渲染内核
@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        u = (i - RES_X / 2.0) / RES_Y * 2.0
        v = (j - RES_Y / 2.0) / RES_Y * 2.0

        ro = ti.Vector([0.0, 1.0, 5.0])
        rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        for bounce in range(max_bounces[None]):
            t, N, obj_color, mat_id = scene_intersect(ro, rd)
            if t > 1e9:
                final_color += throughput * bg_color
                break

            p = ro + rd * t
            if mat_id == MAT_MIRROR:
                ro = p + N * 1e-4
                rd = normalize(reflect(rd, N))
                throughput *= 0.8 * obj_color
            elif mat_id == MAT_DIFFUSE:
                L = normalize(light_pos - p)
                shadow_ray_orig = p + N * 1e-4
                shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                dist_to_light = (light_pos - p).norm()
                in_shadow = 0.0
                if shadow_t < dist_to_light:
                    in_shadow = 1.0

                ambient = 0.2 * obj_color
                direct_light = ambient
                if in_shadow == 0.0:
                    diff = ti.max(0.0, N.dot(L))
                    diffuse = 0.8 * diff * obj_color
                    direct_light += diffuse

                final_color += throughput * direct_light
                break
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)