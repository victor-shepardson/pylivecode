import logging

import numpy as np
from glumpy import app
import pyaudio as pa
from livecode import Layer, VideoWaveTerrain
import IPython

# size = 200, 200
# size = 620, 660
size = 900, 900

size = np.array(size)

screen = Layer(size, 'shader/display.glsl')
feedback = Layer(size, 'shader/feedback2.glsl', n=3)
filtered = Layer(size, 'shader/filter.glsl', n=2)
readback = Layer(size//4, 'shader/readback.glsl', n=1, autoread=True)

vwt = VideoWaveTerrain()

def draw():
    filtered(color=feedback)
    feedback(filtered=filtered)

    readback(color=feedback)
    vwt.feed(readback.cpu)

    screen(color=feedback)

app.use('glfw')
config = app.configuration.Configuration()
config.major_version = 3
config.minor_version = 2
config.profile = "core"
window = app.Window(int(size[0]), int(size[1]), 'vwt', config=config, vsync=True)

@window.event
def on_draw(dt):
    draw()


audio = pa.PyAudio()
def sound(in_data, frame_count, time_info, status):
    try:
        # rb = readback.cpu.reshape(-1, 4)
        # data = (
        #     rb[:frame_count, :2]
        #     )*2-1
        # raise Exception
        data = vwt.step(frame_count)*2-1
    except Exception as e:
        logging.error(e)
        data = np.zeros((frame_count,2))
        # data = np.stack([
        #     np.sin(np.linspace(0,3*2.*np.pi,frame_count,endpoint=False)),
        #     np.sin(np.linspace(0,4*2.*np.pi,frame_count,endpoint=False))
        #     ],1).astype(np.float32)/10.
    return (data, pa.paContinue)

stream = audio.open(
    format=pa.paFloat32,
    channels=2,
    rate=24000,
    output=True,
    stream_callback=sound,
    frames_per_buffer=512
)

@window.event
def on_close():
    stream.stop_stream()
    stream.close()
    audio.terminate()

app.run()
stream.start_stream()
