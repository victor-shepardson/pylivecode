import numpy as np
from vispy import app
from vispy.io import imsave
from livecode import Layer
import IPython

size = 1080, 1080
frame = 0
# size = 1366, 720

size = np.array(size)

def get_shaders(s):
    return ('shader/dreamer/common.glsl', 'shader/dreamer/'+s)

post = Layer(size, get_shaders('post.glsl'), n=1)
colors = Layer(size, get_shaders('colors.glsl'), n=2)
displacements = Layer(size,  get_shaders('displacements.glsl'), n=2)

screen = Layer(size, 'shader/display.glsl')
readback = Layer(size, 'shader/readback.glsl', n=1, autoread=True, short=True, channels=3)

def draw():
    global frame
    colors(displacements=displacements, frame=frame)
    displacements(colors=colors, frame=frame)
    post(colors=colors, frame=frame)
    screen(color=post)
    readback(color=post)
    imsave(f'png/dreamer/{frame:06}.png', readback.cpu)
    frame+=1


class Window(app.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()
    def on_draw(self, event):
        screen.resize(np.array(self.size)*self.pixel_scale)
        draw()
        self.title=str(self.fps).encode('ascii')

if __name__ == '__main__':
    app.set_interactive()

window = Window('dreamer', size, keys='interactive')
window.measure_fps(callback=lambda x: None)
app.run()
# try:
#     app.run()
# except KeyboardInterrupt:
#     pass
