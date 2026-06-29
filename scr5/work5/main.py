import sys
from config import *
from physics import render

def main():
    window = ti.ui.Window("Ray Tracing Demo", (RES_X, RES_Y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3

    try:
        while window.running:
            render()
            canvas.set_image(pixels)

            with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.22):
                light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
                light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
                light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
                max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)

            window.show()
    except KeyboardInterrupt:
        print("用户中断，安全退出")
    except Exception as e:
        print("异常：", e)
    finally:
        window.close()
        sys.exit(0)

if __name__ == '__main__':
    main()