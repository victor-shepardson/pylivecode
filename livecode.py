import os
import sys
import datetime as dt
import logging
from collections import defaultdict, Iterable
import itertools as it
import numpy as np
from glumpy import gloo, gl, library
from glumpy.graphics.collections import PathCollection
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import numba
    from numba import jit, jitclass
except ImportError:
    logging.warning('numba not found')
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
#TODO: parse shader to set w automatically
#TODO: fix shader reloading

def as_iterable(arg):
    return arg if isinstance(arg, Iterable) and not isinstance(arg, str) else (arg,)

def is_shader_path(sh):
    return isinstance(sh, str) and '{' not in sh

observer = Observer()
observer.start()
class SourceHandler(FileSystemEventHandler):
    """maps paths to sets of SourceCode instances which need reloading"""
    def __init__(self):
        super().__init__()
        self.instances = defaultdict(set)
    def on_modified(self, e):
        insts = self.instances.get(e.src_path) or []
        for i in insts:
            i.reload()
    def add_instance(self, instance):
        self.instances[instance.path].add(instance)
source_handler = SourceHandler()

class SourceCode(object):
    """source code as a string or file path"""
    def __init__(self, arg):
        self.changed = self.path = self._code = self.observer = None
        if is_shader_path(arg):
            self.path = os.path.abspath(arg)
            # self.observer = Observer()
            source_handler.add_instance(self)
            observer.schedule(source_handler, os.path.dirname(self.path))
            # self.observer.schedule(source_handler, os.path.dirname(self.path))
            # self.observer.start()
        else:
            self._code = arg
        print(f'SourceCode with path {self.path}')
        self.reload()

    def reload(self):
        if self.path:
            with open(self.path) as src:
                self._code = src.read()
        self.changed = dt.datetime.now()

    @property
    def code(self):
        return self._code

    def __del__(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()


class LiveProgram(object):
    """gloo.Program which watches its source files for changes"""
    def __init__(self, vert=None, frag=None, **kw):
        self.frag_sources = frag and [SourceCode(s) for s in as_iterable(frag)]
        self.vert_sources = vert and [SourceCode(s) for s in as_iterable(vert)]
        self.program = None
        self.program_kw = kw
        self.reloaded = None
        self.reload()

    def reload(self):
        logging.debug(f'reloading shader program last loaded at {self.reloaded}')
        self.reloaded = dt.datetime.now()
        # try:
        frag = '\n'.join(s.code for s in self.frag_sources)
        vert = '\n'.join(s.code for s in self.vert_sources)
        new_program = gloo.Program(vert, frag, **self.program_kw)
        if self.program is not None:
            for k,v in it.chain(self.program.active_uniforms, self.program.active_attributes):
                logging.debug(f'transferring {k}')
                new_program[k] = self.program[k]
        self.old_program = self.program
        self.program = new_program

    @property
    def sources(self):
        return it.chain(self.frag_sources, self.vert_sources)
    @property
    def needs_reload(self):
        return any(s.changed > self.reloaded for s in self.sources)

    def draw(self, *args, **kwargs):
        try:
            if self.needs_reload:
                self.reload()
            # shaders not actually compiled until draw call
            self.program.draw(*args, **kwargs)
        except Exception as e:
            logging.error(e)
            self.program = self.old_program
            if self.program is None: #if this program never compiled
                sys.exit(0)

    def __getitem__(self, key):
        return self.program[key]
    def __setitem__(self, key, value):
        self.program[key] = value
    def __getattr__(self, key):
        if key in ('draw', 'needs_reload', 'reload', 'sources'):
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
                if isinstance(v, Layer):
                    v = v.state[0]
                self.program[k] = v
            for i, state in enumerate(self.target.history):
                for j, buf in enumerate(state):
                    u = 'history_t{}_b{}'.format(i,j)
                    try:
                        self.program[u] = buf
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
            logging.debug(f'setting uniform {k}')
            self.program[k] = v

    @property
    def state(self):
        return self.target.state

    @property
    def cpu(self):
        return self.target.cpu

class NBuffer(object):
    def __init__(self, size, n, w=1,
            autoread=False, short=False, channels=4, wrapping='repeat', interpolation='linear'):
        """Circular collection of FrameBuffer
        size: 2d dimensions in pixels
        n: number of framebuffers:
            n=0 is a dummy
            n=1 is just an FBO
            n=2 ping-pongs (one history buffer available)
            n>2 increases available history (to n-1)
        w: number of colorbuffers per framebuffer (default 1)
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
            color=[
                np.zeros((*size[::-1], channels), self.dtype).view(ttype)
                for i in range(w)],
            # wrapping=wrapping,
            # interpolation=interpolation,
            ) for j in range(n)]
        self.head = 0
        self.n = n
        self.cpu_state = None
        self.autoread = autoread
        self.readback_buffer = 0

    def resize(self, size):
        self.size = size
        for buf in self._state:
            buf.resize(*size)

    def read(self, b):
        assert self.n>0, "nothing to read from NBuffer of length 0"
        return self.state[b].get()

    @property
    def cpu(self):
        if self.cpu_state is None:
            self.cpu_state = self.read(self.readback_buffer)
        return self.cpu_state

    @property
    def state(self):
        return self._state[self.head].color if self.n else None

    @property
    def history(self):
        idxs = (self.head-1-np.arange(self.n-1))%self.n
        return [self._state[i].color for i in idxs]

    def activate(self):
        if self.n:
            self.head = (self.head+1)%self.n
            self._state[self.head].activate()
            if not self.autoread:
                self.cpu_state = None

    def deactivate(self):
        if self.n:
            if self.autoread:
                self.cpu_state = self.read(self.readback_buffer)
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


#TODO: how often are frames repeated?
@jitclass([
    ('next_terrain', numba.float32[:,:,:]),
    ('cur_terrain', numba.float32[:,:,:]),
    ('last_terrain', numba.float32[:,:,:]),
    ('shape', numba.float32[2]),
    ('p', numba.float32[2]),
    ('t', numba.int64),
    # ('sr', numba.int64),
    ('t_cur_added', numba.int64),
    ('t_last_added', numba.int64),
    ('t_next_added', numba.int64),
    ('t_switched', numba.int64),
])
class VideoWaveTerrainJIT(object):
    def __init__(self):#, max_len=3):
        self.next_terrain = np.zeros((1,1,1), dtype=np.float32)
        self.cur_terrain = np.zeros((1,1,1), dtype=np.float32)
        self.last_terrain = np.zeros((1,1,1), dtype=np.float32)
        self.shape = np.ones(2, dtype=np.float32)
        self.p = np.zeros(2, dtype=np.float32)
        self.t = 0
        # self.sr = 24000
        self.t_next_added = -1
        self.t_cur_added = -2
        self.t_last_added = -3
        self.t_switched = 0

    def feed(self, frame):
        self.shape = np.float32(frame.shape[:2])
        frame = np.concatenate((frame, frame[0:1]),0)
        frame = np.concatenate((frame, frame[:,0:1]),1)
        self.next_terrain = frame.astype(np.float32)
        self.t_next_added = self.t

    def switch(self):
        self.last_terrain = self.cur_terrain
        self.t_last_added = self.t_cur_added
        self.cur_terrain = self.next_terrain
        self.t_cur_added = self.t_next_added
        self.t_switched = self.t

    def get(self, x, y, z):
        return lerp(
            interp2d(self.last_terrain, (x,y)),
            interp2d(self.cur_terrain, (x,y)),
            z
        )

    def step(self, n):
        ps = np.empty((n,2), dtype=np.float32)
        cs = np.empty((n,2), dtype=np.float32)
        for i in range(n):
            p, c = self._step()
            ps[i] = p
            cs[i] = c
        return ps, cs

    def _step(self):
        t = self.t - self.t_switched
        dur = self.t_cur_added - self.t_last_added
        if t >= dur:
            self.switch()
            m = 0
        else:
            m = np.minimum(1.,np.float32(t)/np.maximum(1., np.float32(dur)))
        val = self.get(self.p[0], self.p[1], m)
        delta = val[:2]-0.5 #move on red, green
        delta /= np.linalg.norm(delta) + 1e-15
        self.p += delta
        self.p %= self.shape
        self.t += 1
        # p = np.zeros(3)
        # p[:2] = self.p
        p = self.p/self.shape*2-1
        c = val[2:4]
        return p, c #(x,y,0), (b,a)

class Points(object):
    """adapted from glumpy example:
    https://github.com/glumpy/glumpy/blob/master/examples/gloo-trail.py
    """
    def __init__(self, n):
        self.segments = []
        vert = """
        in vec3  a_position;
        in vec4  a_color;
        in float a_size;
        out vec4  v_color;
        out float v_size;
        void main (void)
        {
            v_size = a_size;
            v_color = a_color;
            if( a_color.a > 0.0)
            {
                gl_Position = vec4(a_position, 1.0);
                gl_PointSize = v_size;
            }
            else
            {
                gl_Position = vec4(-1,-1,0,1);
                gl_PointSize = 0.0;
            }
        }
        """
        frag = """
        in vec4 v_color;
        in float v_size;
        out vec4 fragColor;
        void main()
        {
            if( v_color.a <= 0.0)
                discard;
            vec2 r = (gl_PointCoord.xy - vec2(0.5));
            float d = (length(r)-0.5)*v_size;
            if( d < -1. )
                 fragColor = v_color;
            else if( d > 0 )
                 discard;
            else
                fragColor = v_color*vec4(1.,1.,1.,-d);
        }
        """
        self.program = gloo.Program(vert, frag, version='330')
        self.data = np.zeros(n, [
            ('a_position', np.float32, 2),
            ('a_color', np.float32, 4),
            ('a_size', np.float32, 1)
            ]).view(gloo.VertexArray)
        self.program.bind(self.data)

    def append(self, segment):
        assert len(segment)==len(self.data)
        self.segments.append(segment)

    def draw(self):
        for segment in self.segments:
            self.data['a_position'][:] = segment
            self.data['a_color'][:] = 1.
            self.data['a_size'][:] = 4.
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            self.program.draw(gl.GL_POINTS)

        self.segments = []

class VideoWaveTerrain(object):
    def __init__(self, size, frame_count, short=False):
        self.frame_count = frame_count
        self.vwt = VideoWaveTerrainJIT()
        self.points = Points(frame_count)
        self.target = NBuffer(size, 1, short=short)

    def sound(self):
        try:
            ps, cs = self.vwt.step(self.frame_count)
            cs = cs*2-1
            self.points.append(ps)
        except Exception as e:
            logging.error(e)
            cs = np.zeros((self.frame_count,2))
        return cs

    def draw(self):
        with self.target:
            self.points.draw()

    def __getattr__(self, k):
        return getattr(self.vwt, k)
