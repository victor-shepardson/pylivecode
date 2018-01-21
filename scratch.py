import numpy as np
from vispy import app
from livecode import Layer
import IPython

size = 620, 660
# size = 1366, 720

size = np.array(size)

screen = Layer(size, 'display.glsl')
feedback = Layer(size, 'feedback2.glsl', n=3)
filtered = Layer(size, 'filter.glsl', n=2)
readback = Layer(size//4, 'readback.glsl', n=1)

def draw():
    filtered(color=feedback)
    feedback(filtered=filtered)

    readback(color=feedback)
    # print(readback.cpu.shape)

    screen(color=readback)

class Window(app.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()
    def on_draw(self, event):
        screen.resize(window.size)
        draw()
        self.title=str(self.fps).encode('ascii')

if __name__ == '__main__':
    # app.use_app('pyqt5')
    app.set_interactive()

window = Window('pylivecode', size, keys='interactive')
window.measure_fps(callback=lambda x: None)
# try:
#     app.run()
# except KeyboardInterrupt:
#     pass
