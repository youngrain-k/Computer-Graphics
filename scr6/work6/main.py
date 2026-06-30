import math
import taichi as ti
from config import *
from core import generate_target, render_and_compute_loss

def main():
    # 初始化真值图
    generate_target()

    # 初始光源位置
    light_pos[None] = [0.2, 0.2, 0.8]

    # Adam动量变量
    m = [0.0, 0.0, 0.0]
    v = [0.0, 0.0, 0.0]

    # 初始化GUI窗口
    gui = ti.GUI("Differentiable Rendering (Left: Target, Right: Current)", res=(RES * 2, RES))

    print(f"Target Light Position: {TARGET_LIGHT}")
    init_pos = light_pos[None]
    print(f"Initial Light Position: [{init_pos[0]:.3f}, {init_pos[1]:.3f}, {init_pos[2]:.3f}]")
    print("-" * 40)

    # 优化迭代循环
    for iters in range(1, 301):
        loss[None] = 0.0

        # 自动微分反向传播
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()

        grad = light_pos.grad[None]

        # Adam参数更新
        for c in range(3):
            m[c] = BETA1 * m[c] + (1 - BETA1) * grad[c]
            v[c] = BETA2 * v[c] + (1 - BETA2) * grad[c] * grad[c]

            m_hat = m[c] / (1 - BETA1 ** iters)
            v_hat = v[c] / (1 - BETA2 ** iters)

            light_pos[None][c] -= LR * m_hat / (math.sqrt(v_hat) + EPS)

        # 打印日志
        if iters % 10 == 0:
            now_pos = light_pos[None]
            print(f"Iter {iters:03d} | Loss: {loss[None]:.6f} | "
                  f"Light Pos: [{now_pos[0]:.3f}, {now_pos[1]:.3f}, {now_pos[2]:.3f}]")

        # 画面刷新
        gui.set_image(display_pixels)
        gui.show()

if __name__ == "__main__":
    main()