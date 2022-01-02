import numpy as np
from glumpy import app, gl
from livecode import *

size = np.array((1600, 900))
screen_size = np.array((1600, 900))
paths_size = np.array((256, 32))
win_size = screen_size*2

def get_shaders(s):
    return ('shader/lib.glsl', 'shader/'+s+'.glsl')

screen = Layer(screen_size, get_shaders('display-stretch'))
# screen = Layer(screen_size, get_shaders('display-paths-debug'))

feedback = Layer(size, get_shaders('feedback2'), n=3)
filtered = Layer(size, get_shaders('filter'), n=2)
paths = Layer(paths_size, get_shaders('paths'), 
    n=1, scan=True, interpolation=gl.GL_NEAREST)
points = Points(screen_size, paths_size[0]*paths_size[1])

filtered.color = feedback
feedback.filtered = filtered
paths.src = feedback
paths.src_size = size
# screen.color = feedback
# screen.color = paths
screen.color = points

feedback.drag = 0.97

capture = Capture()

frame = 0
def image():
    global frame
    filtered()
    feedback()
    paths()

    positions = paths.cpu[...,:2].reshape(-1, 2)/size*2-1
    cs = np.linspace(0, 1, positions.shape[0])
    colors = np.stack((cs, 1-cs, np.ones(cs.shape), np.ones(cs.shape)), -1)
    # colors = np.ones((positions.shape[0], 4))
    sizes = np.ones((positions.shape[0])) * 4
    points.append((positions, colors, sizes))
    points.draw()

    screen()
    capture.do()
    frame += 1

window = make_window(image, win_size, title='scratch')
@window.event
def on_resize(w,h):
    screen.resize((w,h))

# window.attach(post.program['viewport'])

shell = start_shell(locals())

app.run()
