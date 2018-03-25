import numpy as np
from vispy import app
import pyaudio as pa
from livecode import Layer, VideoWaveTerrain
import IPython

# size = 200, 200
# size = 620, 660
size = 900, 900

size = np.array(size)

oversample = 2

screen = Layer(size, 'shader/display.glsl')
feedback = Layer(size*oversample, 'shader/feedback2.glsl', n=3)
filtered = Layer(size*oversample, 'shader/filter.glsl', n=2)
readback = Layer(size*oversample//4, 'shader/readback.glsl', n=1, autoread=True)

vwt = VideoWaveTerrain()

def draw():
    filtered(color=feedback)
    feedback(filtered=filtered)

    readback(color=feedback)
    vwt.feed(readback.cpu)

    screen(color=feedback)

class Window(app.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()
    def on_draw(self, event):
        screen.resize(np.array(self.size)*self.pixel_scale)
        draw()
        self.title=str(self.fps).encode('ascii')

if __name__ == '__main__':
    # app.use_app('pyqt5')
    app.set_interactive()

window = Window('pylivecode', size, keys='interactive')
window.measure_fps(callback=lambda x: None)

audio = pa.PyAudio()
def sound(in_data, frame_count, time_info, status):
    try:
        # rb = readback.cpu.reshape(-1, 4)
        # data = (
        #     rb[:frame_count, :2]
        #     )*2-1
        data = vwt.step(frame_count)*2-1
    except Exception:
        data = np.zeros((frame_count,2))
        # data = np.stack([
        #     np.sin(np.linspace(0,50*2.*np.pi,frame_count)),
        #     np.sin(np.linspace(0,150*2.*np.pi,frame_count))
        #     ],1)
    return (data, pa.paContinue)

stream = audio.open(
    format=pa.paFloat32,
    channels=2,
    rate=24000,
    output=True,
    stream_callback=sound
)
stream.start_stream()

app.run()

stream.stop_stream()
stream.close()
audio.terminate()
