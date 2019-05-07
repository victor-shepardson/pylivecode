import sys, logging, threading

from glumpy import app

try:
    import IPython
except ImportError:
    logging.warning('shell unavailable; install IPython')

try:
    import pyaudio as pa
    audio = pa.PyAudio()
except ImportError:
    logging.warning('audio unavailable; install pyaudio')

from . utils import *
from . pattern import *
from . midi import *
from . graphics import *
from . vwt import *

# critical bugs/performance issues:
#TODO: fix hidpi + regular display
#TODO: optimize VideoWaveTerrainJIT

# code improvements:
#TODO: parse shaders to set default uniform values (uniform * = ();)
#TODO: allow Points to append any number of points at a time (could draw N at a time)
#TODO: lazy cycling (don't compute unused buffers each frame)
#TODO: "needs_draw" flag on Layers / implicit draw order

# new features:
#TODO: snapshot, push/pop behavior for buffers, programs, layers?
#TODO: capsule shader (points+velocity)
#TODO: distinguish path color from draw color in VideoWaveTerrain(JIT)
#TODO: livecodeable parts of VideoWaveTerrainJIT?


# sugar for setting log level
class _log(type):
    def __getattr__(cls, level):
        logging.getLogger().setLevel(getattr(logging, level))
class log(metaclass=_log):
    pass

#setup glumpy window
def make_window(image_fn, size, title=None, cleanup=True):
    """Return a glumpy app.Window with standard settings"""
    app.use('glfw')
    config = app.configuration.Configuration()
    config.major_version = 3
    config.minor_version = 2
    config.profile = "core"
    window = app.Window(
        int(size[0]), int(size[1]),
        title or '', config=config, vsync=True)

    @window.event
    def on_draw(dt):
        window.set_title('fps: {}'.format(window.fps).encode('ascii'))
        image_fn()

    if cleanup:
        @window.event
        def on_close():
            for stream in streams:
                stream.stop_stream()
                stream.close()
            audio.terminate()
            for shell in shells:
                shell.ex('exit()')
            sys.exit(0)

    return window

# setup pyaudio stream
streams = []
def make_stream(sound_fn, frame_count, channels=2, sample_rate=48000):
    global audio, streams

    def stream_callback(in_data, fc, time_info, status):
        data = sound_fn()
        return (data, pa.paContinue)

    stream = audio.open(
        format=pa.paFloat32,
        channels=channels,
        rate=sample_rate,
        output=True,
        stream_callback=stream_callback,
        frames_per_buffer=frame_count,
        start=False
    )
    streams.append(stream)
    return stream

# setup IPython shell
shells = []
def start_shell(ns):
    """start an IPython shell in a new thread with the given namespace"""
    global shells
    shell = IPython.terminal.embed.InteractiveShellEmbed()
    threading.Thread(target=shell.mainloop, kwargs={'local_ns': ns}).start()
    shells.append(shell)
    return shell
