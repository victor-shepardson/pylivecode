import sys, logging

import numpy as np
from glumpy import app
import pyaudio as pa
from livecode import Layer, VideoWaveTerrain
import IPython

# size = 200, 200
# size = 620, 660
size = 900, 900
size = np.array(size)

frame_count = 512

screen = Layer(size, 'shader/display.glsl')
feedback = Layer(size, 'shader/feedback2.glsl', n=3)
filtered = Layer(size, 'shader/filter.glsl', n=2)
readback = Layer(size//4, 'shader/readback.glsl', n=1, autoread=True)

vwt = VideoWaveTerrain(size, frame_count)

app.use('glfw')
config = app.configuration.Configuration()
config.major_version = 3
config.minor_version = 2
config.profile = "core"
window = app.Window(int(size[0]), int(size[1]), 'vwt', config=config, vsync=True)

@window.event
def on_draw(dt):
    vwt.draw()
    filtered(color=feedback)
    feedback(filtered=filtered)

    readback(color=feedback)
    vwt.feed(readback.cpu)

    # screen(color=feedback)
    screen(color=vwt.target.state[0])

audio = pa.PyAudio()
def sound(in_data, fc, time_info, status):
    assert fc==frame_count, fc
    data = vwt.sound()
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
def on_close():
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sys.exit(0)

app.run()
stream.start_stream()
