from glumpy import app
from pylivecode import Layer

size = 1366, 720

screen = Layer(size, 'display.glsl')
feedback = Layer(size, 'feedback2.glsl', n=3)

window = app.Window(*size)

@window.event
def on_draw(dt):
    screen.resize(window.get_size())
    screen(state=feedback())

try:
    app.run()
except KeyboardInterrupt:
    pass
