import numpy as np
from glumpy import app
from livecode import Layer, make_window, start_shell

size = 620, 660
# size = 1366, 720

size = np.array(size)

def get_shaders(s):
    return ('shader/lib.glsl', 'shader/'+s+'.glsl')

screen = Layer(size, get_shaders('display'))
feedback = Layer(size, get_shaders('feedback2'), n=3)
filtered = Layer(size, get_shaders('filter'), n=2)

def image():
    filtered(color=feedback)
    feedback(filtered=filtered)
    screen(color=feedback)

window = make_window(image, size, title='scratch')
@window.event
def on_resize(w,h):
    screen.resize((w,h))

shell = start_shell(locals())

app.run()

# class Window(app.Canvas):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._timer = app.Timer('auto', connect=self.update, start=True)
#         self.show()
#     def on_draw(self, event):
#         screen.resize(np.array(self.size)*self.pixel_scale)
#         draw()
#         self.title=str(self.fps).encode('ascii')
#
# if __name__ == '__main__':
#     app.set_interactive()
#
# window = Window('pylivecode', size, keys='interactive')
# window.measure_fps(callback=lambda x: None)
# app.run()
# # try:
# #     app.run()
# # except KeyboardInterrupt:
# #     pass
