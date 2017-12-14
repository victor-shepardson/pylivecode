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
    feedback(filtered=filtered.state)
    screen(state=feedback.state)

class Window(app.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()
    def on_draw(self, event):
        screen.resize(window.size)
        draw()

if __name__ == '__main__':
    # app.use_app('pyqt5')
    app.set_interactive()

window = Window('pylivecode', size, keys='interactive')
# try:
#     app.run()
# except KeyboardInterrupt:
#     pass
