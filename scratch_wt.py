import sys, logging
import numpy as np
from glumpy import app, gl

from livecode import *

# size = 900, 900
size = 1920, 1920
size = np.array(size)

frame_count = 512
sample_rate = 24000
n_agents = 3

# global primitives need to be wrapped in a Var even if they aren't patched,
# since shell runs in its own thread -- locals() will copy primitives
# they have to be accessed with e.g. next(V.gain)
V = Vars()
V.gain = 0

def get_shaders(s):
    return ('shader/lib.glsl', 'shader/'+s+'.glsl')

# initialization

screen = Layer(size, get_shaders('display'))
feedback = Layer(size, get_shaders('feedback-aux'), n=2)
filtered = Layer(size, get_shaders('filter'), n=2)
readback = Layer(size//8, get_shaders('readback'), n=1, autoread=True)

vwt = VideoWaveTerrain(size, frame_count, n_agents, point_shader=get_shaders('filter-accum'))
vwt.filtered.decay = 0.9

# patching
filtered.color = feedback
feedback.filtered = filtered
feedback.aux = vwt.filtered
feedback.drag = 0.9
readback.color = feedback
screen.color = feedback
# screen.color = cycle((feedback, vwt.filtered), 3)

# draw order
def image():
    filtered()
    vwt.draw()
    feedback()
    vwt.feed(readback().cpu)
    screen()

# sound generator
def sound():
    data = next(V.gain)*vwt.sound()
    return data


# start app
window = make_window(image, size, title='vwt')
@window.event
def on_resize(w,h):
    screen.resize((w,h))

stream = make_stream(sound, frame_count, channels=2, sample_rate=sample_rate)
stream.start_stream()

shell = start_shell(locals())

app.run()
