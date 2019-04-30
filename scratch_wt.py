import sys, logging
import itertools as it
import numpy as np
from glumpy import app, gl
import pyaudio as pa
from livecode import makeWindow, Layer, VideoWaveTerrain, log, cycle, Vars
import IPython
import threading

# primitives need to be wrapped in a Var since shell runs in its own thread
V = Vars()
V.gain = 0

# size = 200, 200
# size = 620, 660
# size = 900, 900
size = 1920, 1920
size = np.array(size)

frame_count = 512

def get_shaders(s):
    return ('shader/lib.glsl', 'shader/'+s+'.glsl')

# initialization

screen = Layer(size, get_shaders('display'))
feedback = Layer(size, get_shaders('feedback-aux'), n=2)
filtered = Layer(size, get_shaders('filter'), n=2)
readback = Layer(size//8, get_shaders('readback'), n=1, autoread=True)

vwt = VideoWaveTerrain(size, frame_count, 3, point_shader=get_shaders('filter-accum'))

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
    readback()
    vwt.feed(readback.cpu)
    screen()

audio = pa.PyAudio()
def sound(in_data, fc, time_info, status):
    assert fc==frame_count, fc
    data = V.gain*vwt.sound()
    return (data, pa.paContinue)

stream = audio.open(
    format=pa.paFloat32,
    channels=2,
    rate=24000,
    output=True,
    stream_callback=sound,
    frames_per_buffer=frame_count,
    start=False
)

window = makeWindow(size, title='vwt')

@window.event
def on_draw(dt):
    window.set_title('fps: {}'.format(window.fps).encode('ascii'))
    image()

@window.event
def on_close():
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sys.exit(0)

@window.event
def on_resize(w,h):
    screen.resize((w,h))

shell = IPython.terminal.embed.InteractiveShellEmbed()
threading.Thread(target=shell.mainloop, kwargs={'local_ns': locals()}).start()

# stream.start_stream()
threading.Thread(target=stream.start_stream).start()

app.run()
