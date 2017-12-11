from glumpy import app
from pylivecode import Layer

# size = 620, 660
size = 1366, 720

screen = Layer(size, 'display.glsl')
feedback = Layer(size, 'feedback2.glsl', n=2)
filtered = Layer(size, 'filter.glsl', n=2)

window = app.Window(*size)

@window.event
def on_draw(dt):
    screen.resize(window.get_size())

    filtered(state=feedback.state)
    screen(state=feedback(filtered=filtered.state))

try:
    app.run()
except KeyboardInterrupt:
    pass
