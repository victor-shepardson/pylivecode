import os
import sys
from warnings import warn
from collections import defaultdict, Iterable
import itertools as it
import numpy as np
from glumpy import gloo, gl
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import numba
    from numba import jit, jitclass
except ImportError:
    warn('numba not found')
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
#TODO: ipython in separate thread
#TODO: hidpi + regular display
#TODO: window fps count

def is_nonstr_iterable(arg):
    return isinstance(arg, Iterable) and not isinstance(arg, str)

def wrap_str_or_noniterable(arg):
    return arg if is_nonstr_iterable(arg) else (arg,)

def is_shader_path(sh):
    return isinstance(sh, str) and '{' not in sh

class SourceCode(object):
    """source code as a string or file path"""
    def __init__(self, arg):
        self.changed = True
        self.path = self._code = self.observer = None
        if is_shader_path(arg):
            self.path = os.path.abspath(arg)
            class Handler(FileSystemEventHandler):
                def on_modified(handler, e):
                    if e.event_type=='modified' and e.src_path == self.path:
                        self.reload()
            self.observer = Observer()
            self.observer.schedule(Handler(), os.path.dirname(self.path))
            self.observer.start()
        else:
            self._code = arg
        self.reload()

    def reload(self):
        if self.path:
            with open(self.path) as src:
                self._code = src.read()
        self.changed = True
        # return self._code

    @property
    def code(self):
        self.changed = False
        return self._code

    def __del__(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()


class LiveProgram(object):
    """gloo.Program which watches its source files for changes"""
    def __init__(self, vert=None, frag=None, **kw):
        self.frag_sources = frag and [SourceCode(s) for s in wrap_str_or_noniterable(frag)]
        self.vert_sources = vert and [SourceCode(s) for s in wrap_str_or_noniterable(vert)]
        self.program = None
        self.program_kw = kw
        self.reload()

    def reload(self):
        prev_program = self.program
        try:
            frag = '\n'.join(s.code for s in self.frag_sources)
            vert = '\n'.join(s.code for s in self.vert_sources)
            self.program = gloo.Program(vert, frag, **self.program_kw)
            if prev_program is not None:
                for k,v in it.chain(prev_program.active_uniforms, prev_program.active_attributes):
                    self.program[k] = prev_program[k]
        except Exception as e:
            warn(e)
            if prev_program is not None:
                self.program = prev_program
            else:
                sys.exit(0)

    def needs_reload(self):
        return any(s.changed for s in it.chain(self.frag_sources, self.vert_sources))

    def draw(self, *args, **kwargs):
        if self.needs_reload():
            warn('reloading')
            self.reload()
        self.program.draw(*args, **kwargs)

    def __getitem__(self, key):
        return self.program[key]
    def __setitem__(self, key, value):
        self.program[key] = value
    def __getattr__(self, key):
        if key in ('draw', 'needs_reload', 'reload'):
            return super().__getattr__(key)
        return getattr(self.program, key)
    # def __setattr__(self, *args):
    #     self.program.__setattr__(*args)


class Layer(object):
    def __init__(self, size, shader, n=0, **buffer_args):
        self.program = LiveProgram(vert="""
            in vec2 position;
            void main(){
              gl_Position = vec4(position, 0.0, 1.0);
            }""", frag=shader, count=4, version='330')
        self.program['size'] = size
        dtype = [('position', np.float32, 2)]
        quad_arrays = np.zeros(4, dtype).view(gloo.VertexArray)
        quad_arrays['position'] = [
            (-1, -1), (-1, +1),
            (+1, -1), (+1, +1)
        ]
        self.program.bind(quad_arrays)
        self.draw_method = gl.GL_TRIANGLE_STRIP
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
            gl.glDisable(gl.GL_BLEND)
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
    def __init__(self, size, n,
            autoread=False, short=False, channels=4, wrapping='repeat', interpolation='linear'):
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
        short:
            use int8 texture
        channels:
            number of color channels
        # wrapping:
        #     argument to gloo.Texture2D
        # interpolation:
        #     argument to gloo.Texture2D
        """
        self.size = size
        self.dtype = np.uint8 if short else np.float32
        # internalformat = 'rgba'[:channels]+('8' if short else '32f')
        ttype = gloo.Texture2D if short else gloo.TextureFloat2D
        self._state = [gloo.FrameBuffer(
            color=[np.zeros((*size[::-1], channels), self.dtype).view(ttype)],
            # wrapping=wrapping,
            # interpolation=interpolation,
            ) for _ in range(n)]
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
            out_type=self.dtype)

    @property
    def cpu(self):
        if self.cpu_state is None:
            self.cpu_state = self.read()
        return self.cpu_state

    @property
    def state(self):
        return self._state[self.head].color[0] if self.n else None

    @property
    def history(self):
        idxs = (self.head-1-np.arange(self.n-1))%self.n
        return [self._state[i].color[0] for i in idxs]

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
        r = np.empty((n,2), dtype=np.float32)
        for i in range(n):
            r[i] = self._step()
        return r

    def _step(self):
        val = self.get(self.p[0], self.p[1], self.t)
        self.p += val[:2]-0.5
        self.p %= self.shape
        self.t += 0.
        return val[2:4]
