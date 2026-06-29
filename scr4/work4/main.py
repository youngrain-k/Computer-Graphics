from config import *
from physics import render

def main():
    window = ti.ui.Window("Phong Shading Demo", (RES_X, RES_Y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # 初始化光照参数
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0

    while window.running:
        render()
        canvas.set_image(pixels)

        # 滑动条调节材质
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.22):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)

        window.show()

if __name__ == '__main__':
    main()