import os
import itertools as it
import numpy as np
from vispy import gloo
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def is_path(sh):
    return isinstance(sh, str) and '{' not in sh

class LiveProgram(gloo.Program):
    def __init__(self, vert=None, frag=None, **kw):
        self.frag = self.vert = self.frag_path = self.vert_path = None
        if is_path(frag):
            self.frag_path = os.path.abspath(frag)
        else:
            self.frag = frag
        if is_path(vert):
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
        frag = self.frag or open(self.frag_path).read()
        vert = self.vert or open(self.vert_path).read()
        self.set_shaders(vert, frag)
        self.needs_reload = False

    def draw(self, *args, **kwargs):
        if self.needs_reload:
            self.reload()
        super().draw(*args, **kwargs)

    def cleanup(self): #???
        for ob in self.observers:
            ob.stop()
            ob.join()

class Layer(object):
    def __init__(self, size, shader, n=0):
        self.program = LiveProgram(vert="""
            attribute vec2 position;
            void main(){
              gl_Position = vec4(position, 0.0, 1.0);
            }""", frag=shader, count=4)
        self.program['size'] = size
        self.program['position'] = [
            (-1, -1),
            (-1, +1),
            (+1, -1),
            (+1, +1)
        ]
        self.draw_method = gloo.gl.GL_TRIANGLE_STRIP
        self.target = NBuffer(size, n)

    def __call__(self, **kwargs):
        for k,v in kwargs.items():
            self.program[k] = v
        with self.target:
            for i,t in enumerate(self.target.history):
                u = 'history_{}'.format(i)
                try:
                    self.program[u] = self.target.history[i]
                except IndexError:
                    pass
            self.program.draw(self.draw_method)
        return self.state

    def resize(self, size):
        self.target.resize(size)
        self.program['size'] = size

    @property
    def state(self):
        return self.target.state

class NBuffer(object):
    def __init__(self, size, n):
        self._state = [gloo.FrameBuffer(color=gloo.Texture2D(
            np.zeros((*size[::-1], 4), np.float32)
            )) for _ in range(n)]
        self.head = 0
        self.n = n

    def resize(self, size):
        for buf in self._state:
            buf.resize(*size)

    @property
    def state(self):
        return self._state[self.head].color_buffer if self.n else None

    @property
    def history(self):
        return [
            self._state[i].color_buffer
            for i in (self.head-1-np.arange(self.n-1))%self.n]

    def activate(self):
        if self.n:
            self.head = (self.head+1)%self.n
            self._state[self.head].activate()

    def deactivate(self):
        if self.n:
            self._state[self.head].deactivate()

    def __enter__(self):
        self.activate()
    def __exit__(self, *args):
        self.deactivate()
