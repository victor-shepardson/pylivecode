import sys, logging
import itertools as it
import numpy as np
from glumpy import app
import pyaudio as pa
from livecode import Layer, VideoWaveTerrain, log, cycle
import IPython

gain = 0

# size = 200, 200
# size = 620, 660
# size = 900, 900
size = 1920, 1920
size = np.array(size)

frame_count = 512

screen = Layer(size, 'shader/display.glsl')
feedback = Layer(size, 'shader/feedback-aux.glsl', n=3)
filtered = Layer(size, 'shader/filter.glsl', n=2)
readback = Layer(size//8, 'shader/readback.glsl', n=1, autoread=True)

vwt = VideoWaveTerrain(size, frame_count, 3)

app.use('glfw')
config = app.configuration.Configuration()
config.major_version = 3
config.minor_version = 2
config.profile = "core"
window = app.Window(int(size[0]), int(size[1]), 'vwt', config=config, vsync=True)

# filtered.color = lambda: feedback
# feedback.filtered = lambda: filtered
# feedback.aux = lambda: vwt.filtered
# feedback.drag = 0.9
# readback.color = lambda: feedback
# screen.color = lambda: feedback

filtered.color = feedback
feedback.filtered = filtered
feedback.aux = vwt.filtered
feedback.drag = 0.9
readback.color = feedback
screen.color = feedback

# screen.color = cycle((feedback, vwt.filtered), 3)

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
    data = gain*vwt.sound()
    return (data, pa.paContinue)

stream = audio.open(
    format=pa.paFloat32,
    channels=2,
    rate=24000,
    output=True,
    stream_callback=sound,
    frames_per_buffer=frame_count
)

@window.event
def on_draw(dt):
    # screen.resize(window.get_size())
    window.set_title('fps: {}'.format(window.fps).encode('ascii'))
    image()

@window.event
def on_close():
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sys.exit(0)

app.run()
stream.start_stream()
