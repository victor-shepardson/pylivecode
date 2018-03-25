import numpy as np
from vispy import app
from livecode import Layer
import IPython

size = 620, 660
# size = 1366, 720

size = np.array(size)

screen = Layer(size, 'shader/display.glsl')
feedback = Layer(size, 'shader/feedback2.glsl', n=3)
filtered = Layer(size, 'shader/filter.glsl', n=2)

def draw():
    filtered(color=feedback)
    feedback(filtered=filtered)
    screen(color=feedback)

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

window = Window('pylivecode', size, keys='interactive')
window.measure_fps(callback=lambda x: None)
app.run()
# try:
#     app.run()
# except KeyboardInterrupt:
#     pass
