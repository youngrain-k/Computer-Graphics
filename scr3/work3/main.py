import numpy as np
from config import *
from physics import de_casteljau, clear_pixels, draw_curve_kernel

def main():
    window = ti.ui.Window("Bezier Curve", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []

    while window.running:
        # 鼠标事件
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c':
                control_points.clear()

        clear_pixels()
        cnt = len(control_points)

        # 生成贝塞尔曲线
        if cnt >= 2:
            curve_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_idx in range(NUM_SEGMENTS + 1):
                t = t_idx / NUM_SEGMENTS
                curve_np[t_idx] = de_casteljau(control_points, t)
            curve_points_field.from_numpy(curve_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        canvas.set_image(pixels)

        # 绘制控制点与控制线
        if cnt > 0:
            np_p = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_p[:cnt] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_p)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))

            if cnt >= 2:
                idx_list = []
                for i in range(cnt - 1):
                    idx_list.extend([i, i+1])
                np_idx = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_idx[:len(idx_list)] = np.array(idx_list)
                gui_indices.from_numpy(np_idx)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5,0.5,0.5))

        window.show()

if __name__ == '__main__':
    main()