import os
from collections import defaultdict
import itertools as it
import numpy as np
from vispy import gloo
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import numba
    from numba import jit, jitclass
except ImportError:
    # dummy numba decorators
    dummy_decorator = lambda *a, **kw: lambda f: f
    jit = jitclass = dummy_decorator
    # for jitclass spec... if spec could be lazy evaluated, wouldn't need
    class dummy:
        def __getitem__(self, _):
            return None
        def __getattr__(self, _):
            return dummy()
    numba = dummy()

#idea: snapshot, push/pop behavior for buffers, programs and layers

def is_shader_path(sh):
    return isinstance(sh, str) and '{' not in sh

class LiveProgram(gloo.Program):
    """gloo.Program which watches its source files for changes"""
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
        """Render to self.target. Keyword args bound to shader."""
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
        """sugar: most attributes fall through to shader program"""
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
        # return self._state[self.head].read()
        # for some reason FrameBuffer.read() doesn't have out_type argument
        # it also appears to just skip the gl.glReadBuffer
        # so not clear whether this is robust at all
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
            self.head = (self.head+1)%self.n
            self._state[self.head].activate()
            if not self.autoread:
                self.cpu_state = None

    def deactivate(self):
        if self.n:
            if self.autoread:
                self.cpu_state = self.read()
            self._state[self.head].deactivate()

    def __enter__(self):
        self.activate()
    def __exit__(self, *args):
        self.deactivate()
    def __len__(self):
        return self.n

@jit(nopython=True)
def lerp(a, b, m):
    return a*(1-m) + b*m

@jit(nopython=True)
def _mcoords(x):
    x_ = int(x)
    return x_, x_+1, x-x_

@jit(nopython=True)
def interp2d(a, p):
    """interpolating ndarray access"""
    x_lo, x_hi, x_m = _mcoords(p[0])
    y_lo, y_hi, y_m = _mcoords(p[1])
    a = a[x_lo:x_hi+1, y_lo:y_hi+1]
    a = lerp(a[0], a[1], x_m)
    return lerp(a[0], a[1], y_m)

# @jit(nopython=True)
# def _vwt_getitem(frames, coord):
#     x,y,z = coord
#     new_z, old_z, z_m = _mcoords(z)
#     new_val = interp2d(frames[new_z], (x,y))
#     if z_m==0:
#         return new_val
#     return lerp(
#         new_val,
#         interp2d(frames[old_z], (x,y)),
#         z_m)
#
# @jit(nopython=True)
# def _displace(x):
#     # assert x.ndim==1
#     return x[:2]
#
# @jit(nopython=True)
# def _step(frames, p, t):
#     if len(frames)==0:
#         return p, t
#     coord = np.empty(3)
#     coord[:2] = p
#     coord[2] = t
#     x = _vwt_getitem(frames, coord)
#     p += _displace(x)
#     p %= frames[0].shape[:2]
#     t += 0.
#     return p, t
#
# @jit(nopython=True)
# def _stepn(frames, p, t, n):
#     for _ in range(n):
#         p, t = _step(frames, p, t)
#     return p, t

# class VideoWaveTerrain(object):
#     # add frames
#     # eviction policy
#     # interpolating x,y,z access
#     # map t -> z
#     # update rule
#     # draw rule
#     def __init__(self, max_len=3):
#         self.frames = ()
#         self.max_len = max_len
#         self.p = np.zeros(2, dtype=np.float32)
#         self.t = 0.
#
#     def feed(self, frame):
#         if frame.ndim!=3:
#             frame = frame.reshape(*frame.shape[:2],-1)
#         # self.frames.insert(0, frame)
#         # if len(self.frames) > self.max_len:
#             # self.frames.pop()
#         self.frames = (frame,)+self.frames[:self.max_len-1]
#
#     def __getitem__(self, coord):
#         return _vwt_getitem(self.frames, coord)
#
#     def step(self, n):
#         self.p, self.t = _stepn(self.frames, self.p, self.t, n)

@jitclass([
    ('terrain', numba.float32[:,:,:]),
    ('shape', numba.float32[2]),
    ('p', numba.float32[2]),
    ('t', numba.float32)
])
class VideoWaveTerrain(object):
    def __init__(self):#, max_len=3):
        self.terrain = np.zeros((1,1,1), dtype=np.float32)
        self.shape = np.ones(2, dtype=np.float32)
        self.p = np.zeros(2, dtype=np.float32)
        self.t = 0.

    def feed(self, frame):
        self.shape = np.float32(frame.shape[:2])
        frame = np.concatenate((frame, frame[0:1]),0)
        frame = np.concatenate((frame, frame[:,0:1]),1)
        self.terrain = frame

    def get(self, x,y,z):
        return interp2d(self.terrain, (x,y))

    def step(self, n):
        for _ in range(n):
            self._step()

    def _step(self):
        val = self.get(self.p[0], self.p[1], self.t)
        self.p += val[:2]
        self.p %= self.shape
        self.t += 0.
