import numpy as np
from glumpy import app
from livecode import Layer, make_window, start_shell, imsave_mp, init_imsave_mp
import IPython

import functools as ft

#####################
size = 1920,1080#2100, 2100#1080, 1080#

steps_per_frame = 4

save_images = False
#####################

size = np.array(size)
frame = 0

def get_shaders(s):
    return ('shader/dreamer/common.glsl', 'shader/dreamer/'+s)

post = Layer(size, get_shaders('post.glsl'), n=1)
dreamer = Layer(size, get_shaders('dreamer.glsl'), n=2)
screen = Layer(size, 'shader/display-stretch.glsl')

if save_images:
    init_imsave_mp(n_procs=6, max_tasks=12)
    readback = Layer(size, 'shader/readback.glsl', n=1, autoread=True, short=True, channels=3)
    readback.color = post

post.color = dreamer
screen.color = post

def image():
    global frame

    for _ in range(steps_per_frame):
        dreamer(frame=frame)
        post(frame=frame)

    screen()

    if save_images and frame%2:
        imsave_mp(f'png/dreamer/{frame//2:06}.png', readback().cpu)

    frame+=1

window = make_window(image, size, title='dreamer')
@window.event
def on_resize(w,h):
    screen.resize((w,h))

shell = start_shell(locals())

app.run()
