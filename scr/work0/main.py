import taichi as ti
from config import *
from physics import get_model_matrix, get_view_matrix, get_projection_matrix

# 全局场定义
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

@ti.kernel
def compute_transform(angle: ti.f32):
    """计算顶点的 MVP 变换与视口映射"""
    model = get_model_matrix(angle)
    view = get_view_matrix(EYE_POS)
    proj = get_projection_matrix(FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)
    mvp = proj @ view @ model

    for i in range(3):
        v = vertices[i]
        # 补全齐次坐标
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换：映射到 GUI 的 [0, 1] 空间
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化三角形顶点
    for i in range(3):
        vertices[i] = INIT_VERTICES[i]

    # 创建 GUI 窗口
    gui = ti.GUI(WINDOW_TITLE, res=WINDOW_RES)
    angle = 0.0  # 旋转角度

    while gui.running:
        # 处理所有事件（修复按键无响应问题）
        for e in gui.get_events():
            if e.key == 'a':
                angle += 10.0
            elif e.key == 'd':
                angle -= 10.0
            elif e.key == ti.GUI.ESCAPE:
                gui.running = False

        # 更新变换
        compute_transform(angle)

        # 获取屏幕坐标并绘制三角形
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]

        gui.line(a, b, radius=2, color=0xFF0000)   # 红边
        gui.line(b, c, radius=2, color=0x00FF00)   # 绿边
        gui.line(c, a, radius=2, color=0x0000FF)   # 蓝边

        gui.show()

if __name__ == '__main__':
    main()