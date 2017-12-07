import os
import itertools as it
import numpy as np
from glumpy import gloo, gl
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

#idea: __call__ returns an object which allows nested calls to reuse the same buffers somehow?
#idea: torch layers?
#idea: multitarget?
#TODO: eval main loop

def is_path(sh):
    return isinstance(sh, str) and '{' not in sh

class LiveProgram:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.needs_reload = [os.path.abspath(sh) for sh in kwargs.values() if is_path(sh)]
        class Handler(FileSystemEventHandler):
            def on_modified(handler, e):
                if e.event_type=='modified' and e.src_path in self.needs_reload:
                    self.reload()
        self.observers = []
        dirs = set(os.path.dirname(p) for p in self.needs_reload)
        for d in dirs:
            obs = Observer()
            obs.schedule(Handler(), d)
            obs.start()
            self.observers.append(obs)
        self.program = None
        self.reload()

    def reload(self):
        new_program = gloo.Program(**self.kwargs)
        if self.program:
            for k,_ in it.chain(self.program.all_uniforms, self.program.all_attributes):
                try:
                    new_program[k] = self.program[k]
                except IndexError:
                    print(k)
        self.program = new_program

    def cleanup(self): #???
        for ob in self.observers:
            ob.stop()
            ob.join()

    def __getattr__(self, attr):
        return getattr(self.program, attr)
    def __getitem__(self, k):
        return self.program[k]
    def __setitem__(self, k, v):
        self.program[k] = v

class Layer(object):
    def __init__(self, size, shader, n=0):
        self.program = LiveProgram(vertex="""
            attribute vec2 position;
            void main(){
              gl_Position = vec4(position, 0.0, 1.0);
            }""", fragment=shader, count=4)
        self.program['size'] = size
        self.program['position'] = [
            (-1, -1),
            (-1, +1),
            (+1, -1),
            (+1, +1)
        ]
        self.draw_method = gl.GL_TRIANGLE_STRIP
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
        self._state = [gloo.FrameBuffer(color=[
            np.zeros(
                (*size[::-1], 4), np.float32
            ).view(gloo.TextureFloat2D)
            ]) for _ in range(n)]
        self.head = 0
        self.n = n

    def resize(self, size):
        for buf in self._state:
            buf.resize(*size)

    @property
    def state(self):
        return self._state[self.head].color[0] if self.n else None

    @property
    def history(self):
        return [
            self._state[i].color[0]
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
