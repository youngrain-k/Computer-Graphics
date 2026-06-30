import taichi as ti
from config import *

@ti.kernel
def generate_target():
    """生成目标参考真值图像"""
    for i, j in target_pixels:
        x = (i + 0.5) / RES
        y = (j + 0.5) / RES
        dx = x - SPHERE_CENTER[0]
        dy = y - SPHERE_CENTER[1]
        dist_sq = dx**2 + dy**2

        if dist_sq < SPHERE_RADIUS**2:
            dz = ti.sqrt(SPHERE_RADIUS**2 - dist_sq)
            z = SPHERE_CENTER[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - SPHERE_CENTER).normalized()
            
            target_light_vec = ti.Vector(TARGET_LIGHT)
            l_dir = (target_light_vec - p).normalized()

            dot_val = n.dot(l_dir)
            target_pixels[i, j] = ti.max(0.0, ti.min(1.0, dot_val))
        else:
            target_pixels[i, j] = 0.0

@ti.kernel
def render_and_compute_loss():
    """正向渲染 + 计算MSE损失，支持自动微分"""
    for i, j in target_pixels:
        x = (i + 0.5) / RES
        y = (j + 0.5) / RES
        dx = x - SPHERE_CENTER[0]
        dy = y - SPHERE_CENTER[1]
        dist_sq = dx**2 + dy**2

        intensity = 0.0
        if dist_sq < SPHERE_RADIUS**2:
            dz = ti.sqrt(SPHERE_RADIUS**2 - dist_sq)
            z = SPHERE_CENTER[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - SPHERE_CENTER).normalized()
            l_dir = (light_pos[None] - p).normalized()

            dot_val = n.dot(l_dir)
            intensity = ti.max(0.1 * dot_val, dot_val)
        
        diff = intensity - target_pixels[i, j]
        loss[None] += (1.0 / (RES * RES)) * (diff ** 2)

        display_pixels[i, j] = target_pixels[i, j]
        display_pixels[i + RES, j] = ti.max(0.0, ti.min(1.0, intensity))