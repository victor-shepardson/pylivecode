import numpy as np
from glumpy import app
from imageio import imwrite
from livecode import Layer
import IPython

import functools as ft
from multiprocessing import Pool, Array

#####################
size = 1920, 1920

save_images = False
n_saving_procs = 6
maxtasks = 12
#####################

size = np.array(size)
frame = 0

def get_shaders(s):
    return ('shader/dreamer/common.glsl', 'shader/dreamer/'+s)

post = Layer(size, get_shaders('post.glsl'), n=1)
dreamer = Layer(size, get_shaders('dreamer.glsl'), n=2, w=2)
#colors = Layer(size, get_shaders('colors.glsl'), n=2)
#displacements = Layer(size,  get_shaders('displacements.glsl'), n=2)

screen = Layer(size, 'shader/display.glsl')
if save_images:
    readback = Layer(size, 'shader/readback.glsl', n=1, autoread=True, short=True, channels=3)

pool = Pool(n_saving_procs)
tasks = []
def imsave_mp(path, arr):
    global tasks
    tasks.append(pool.apply_async(
        imwrite, (path, arr), dict(compress_level=5),
        error_callback=print))
    while len(tasks) > maxtasks:
        tasks = [t for t in tasks if not t.ready()]

def draw():
    global frame

    dreamer(frame=frame)
    post(colors=dreamer.state[0], frame=frame)
    # displacements(colors=colors, frame=frame)
    # colors(displacements=displacements, frame=frame)
    # post(colors=colors, frame=frame)

    screen(color=post)
    # screen(color=colors)

    if save_images and frame%2:
        readback(color=post)
        imsave_mp(f'png/dreamer/{frame//2:06}.png', readback.cpu)

    frame+=1

app.use('glfw')
config = app.configuration.Configuration()
config.major_version = 3
config.minor_version = 2
config.profile = "core"
window = app.Window(int(size[0]), int(size[1]), 'dreamer', config=config)
# hack: --vsync option unimplemented
app.__backend__.glfw.glfwSwapInterval(1)

@window.event
def on_draw(dt):
    #screen.resize(np.array(window.get_size()))#*self.pixel_scale)
    draw()

app.run()
