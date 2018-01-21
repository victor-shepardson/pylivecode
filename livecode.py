import os
import itertools as it
import numpy as np
from vispy import gloo
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

#idea: snapshot, push/pop behavior for buffers, programs and layers

def is_shader_path(sh):
    return isinstance(sh, str) and '{' not in sh

class LiveProgram(gloo.Program):
    def __init__(self, vert=None, frag=None, **kw):
        self.frag = self.vert = self.frag_path = self.vert_path = None
        if is_shader_path(frag):
            self.frag_path = os.path.abspath(frag)
        else:
            self.frag = frag
        if is_shader_path(vert):
            self.vert_path = os.path.abspath(vert)
        else:
            self.vert = vert

        super().__init__(**kw)

        self.needs_reload = True
        class Handler(FileSystemEventHandler):
            def on_modified(handler, e):
                if e.event_type=='modified' and e.src_path in (self.frag_path, self.vert_path):
                    self.needs_reload = True
        self.observers = []
        dirs = set(
            os.path.dirname(p) for p in (self.vert_path, self.frag_path) if p)
        for d in dirs:
            obs = Observer()
            obs.schedule(Handler(), d)
            obs.start()
            self.observers.append(obs)

    def reload(self):
        self.prev_shaders = self._shaders
        frag = self.frag or open(self.frag_path).read()
        vert = self.vert or open(self.vert_path).read()
        self.set_shaders(vert, frag)
        self.needs_reload = False

    def rollback(self):
        self.set_shaders(*self.prev_shaders)

    def draw(self, *args, **kwargs):
        try:
            if self.needs_reload:
                self.reload()
            super().draw(*args, **kwargs)
        except RuntimeError as e:
            print(e)
            self.rollback()

    def cleanup(self): #???
        for ob in self.observers:
            ob.stop()
            ob.join()

class Layer(object):
    def __init__(self, size, shader, n=0, **buffer_args):
        self.program = LiveProgram(vert="""
            attribute vec2 position;
            void main(){
              gl_Position = vec4(position, 0.0, 1.0);
            }""", frag=shader, count=4)
        self.program['size'] = size
        self.program['position'] = [
            (-1, -1), (-1, +1),
            (+1, -1), (+1, +1)
        ]
        self.draw_method = gloo.gl.GL_TRIANGLE_STRIP
        self.target = NBuffer(size, n, **buffer_args)

    def __call__(self, **kwargs):
        with self.target:
            for k,v in kwargs.items():
                try:
                    v = v.state
                except Exception:
                    pass
                self.program[k] = v
            for i,t in enumerate(self.target.history):
                u = 'history_{}'.format(i)
                try:
                    self.program[u] = t
                except IndexError as e:
                    pass
            self.program.draw(self.draw_method)
        return self.state

    def resize(self, size):
        self.target.resize(size)
        self.program['size'] = size

    def __setattr__(self, k, v):
        if k in ['program', 'draw_method', 'target']:
            super().__setattr__(k, v)
        else:
            print('setting uniform {}'.format(k))
            self.program[k] = v

    @property
    def state(self):
        return self.target.state

    @property
    def cpu(self):
        return self.target.cpu

class NBuffer(object):
    def __init__(self, size, n, autoread=False):
        """Circular collection of FrameBuffer
        size: 2d dimensions in pixels
        n: number of buffers:
            n=0 is a dummy
            n=1 is just an FBO
            n=2 ping-pongs (one history buffer available)
            n>2 increases available history (to n-1)
        autoread:
            if True, replace cpu_state when deactivating.
            if False, replace when cpu property is requested since activating
        """
        self.size = size
        self._state = [gloo.FrameBuffer(color=gloo.Texture2D(
            np.zeros((*size[::-1], 4), np.float32),
            wrapping='repeat',
            interpolation='linear',
            internalformat='rgba32f'
            )) for _ in range(n)]
        self.head = 0
        self.n = n
        self.cpu_state = None
        self.autoread = autoread

    def resize(self, size):
        self.size = size
        for buf in self._state:
            buf.resize(*size)

    def read(self):
        assert self.n>0, "nothing to read from NBuffer of length 0"
        # for some reason FrameBuffer.read() doesn't have out_type argument
        # it also appears to just skip the gl.glReadBuffer
        # so not clear whether this is robust at all
        # return self._state[self.head].read()
        return gloo.read_pixels(
            viewport=(0,0,*self.size),
            out_type=np.float32)

    @property
    def cpu(self):
        if self.cpu_state is None:
            self.cpu_state = self.read()
        return self.cpu_state

    @property
    def state(self):
        return self._state[self.head].color_buffer if self.n else None

    @property
    def history(self):
        idxs = (self.head-1-np.arange(self.n-1))%self.n
        return [self._state[i].color_buffer for i in idxs]

    def activate(self):
        if self.n:
            if not self.autoread:
                self.cpu_state = None
            self.head = (self.head+1)%self.n
            self._state[self.head].activate()

    def deactivate(self):
        if self.n:
            self._state[self.head].deactivate()
            if self.autoread:
                self.cpu_state = self.read()

    def __enter__(self):
        self.activate()
    def __exit__(self, *args):
        self.deactivate()
    def __len__(self):
        return self.n

# class VideoWaveTerrain(object):
