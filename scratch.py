from vispy import app
from pylivecode import Layer
import IPython

# size = 620, 660
size = 1366, 720

screen = Layer(size, 'display.glsl')
feedback = Layer(size, 'feedback2.glsl', n=2)
filtered = Layer(size, 'filter.glsl', n=2)

def draw():
    filtered(state=feedback.state)
    screen(state=feedback(filtered=filtered.state))

class Window(app.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()
    def on_draw(self, event):
        screen.resize(window.size)
        draw()
window = Window('pylivecode', size)
try:
    app.run()
except KeyboardInterrupt:
    pass
