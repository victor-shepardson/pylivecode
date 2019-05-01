import sys, os, logging, threading, re
from multiprocessing import Pool
from collections import defaultdict, Iterable
import datetime as dt
import itertools as it
import numpy as np

from glumpy import gloo, gl, library, app
from glumpy.graphics.collections import PathCollection
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import IPython
except ImportError:
    logging.warning('shell unavailable; install IPython')

try:
    from imageio import imwrite
except ImportError:
    logging.warning('image saving unvailable; install imageio')

try:
    import pyaudio as pa
    audio = pa.PyAudio()
except ImportError:
    logging.warning('audio unavailable; install pyaudio')

try:
    import numba
    from numba import jit, jitclass
except ImportError:
    logging.warning('numba not found, JIT code will be slow')
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

# critical bugs/performance issues:
#TODO: fix hidpi + regular display
#TODO: optimize VideoWaveTerrainJIT / debug popping
#TODO: look into gaps in vwt paths
# code improvements:
#TODO: parse shaders to set default uniform values (uniform * = ();)
#TODO: move pyaudio dependency from scratch_wt.py into livecode.py
#TODO: move imsave from dreamer.py to livecode.py
#TODO: allow Points to append any number of points at a time (could still draw fixed N at a time)
#TODO: lazy cycling (don't compute unused buffers each frame)
#TODO: "needs_draw" flag on Layers / implicit draw order
#TODO: livecodeable parts of VideoWaveTerrainJIT
# new features:
#TODO: snapshot, push/pop behavior for buffers, programs, layers?
#TODO: capsule shader (points+velocity)
#TODO: distinguish path color from draw color in VideoWaveTerrain(JIT)

# sugar for setting log level
class _log(type):
    def __getattr__(cls, level):
        logging.getLogger().setLevel(getattr(logging, level))
class log(metaclass=_log):
    pass

# pattern utilites on top of itertools
def cycle(iterable, period=1):
    return it.cycle(it.chain.from_iterable(it.repeat(x, period) for x in iterable))


class Var(object):
    """convert everything to iterables & wrap primitives"""
    def __init__(self, v=None):
        self.set(v)
    def __next__(self):
        return next(self.v)
    def set(self, v):
        if not isinstance(v, it.cycle):
            v = it.cycle((v,))
        self.v = v

class Vars(defaultdict):
    """store Vars and provide syntactic sugar for setting/incrementing"""
    def __init__(self, **kwargs):
        super().__init__(Var)

        for k,v in kwargs.items():
            self.__setattr__(k, v)

    def __getattr__(self, k):
        return next(self[k])

    def __setattr__(self, k, v):
        self[k].set(v)

# file watching/reloading tools
def as_iterable(arg):
    return arg if isinstance(arg, Iterable) and not isinstance(arg, str) else (arg,)

def is_shader_path(sh):
    """assume source code of containing '{', else assume path"""
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
            source_handler.add_instance(self)
            observer.schedule(source_handler, os.path.dirname(self.path))
        else:
            self._code = arg
        logging.debug(f'SourceCode with path {self.path}')
        self.reload()

    def reload(self):
        if self.path:
            with open(self.path) as src:
                self._code = src.read()
        self.update()
        self.changed = dt.datetime.now()

    @property
    def code(self):
        return self._code

    def update(self):
        pass

class GLSLSourceCode(SourceCode):
    collapse_regex = re.compile(
        r'\{.*\}|/\*.*\*/|layout\s*\(\s*location\s*=\s*\d+\s*\)',
        flags=re.DOTALL)

    def update(self):
        # parse for uniforms and outs
        # replace content between curly braces with ; split on ;
        lines = re.sub(GLSLSourceCode.collapse_regex, ';', self._code).split(';')
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if l and not l.startswith('//')]
        def parse_decs(match):
            return {tuple(l.split()[1:3]) for l in lines if l.startswith(match)}
        self.ins = parse_decs('in')
        self.outs = parse_decs('out')
        self.uniforms = parse_decs('uniform')
        # self.lines = lines

# graphics tools on top of glumpy
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


class LiveProgram(object):
    """encapsulates a gloo.Program and watches its source files for changes"""
    def __init__(self, vert=None, frag=None, **kw):
        self.frag_sources = frag and [GLSLSourceCode(s) for s in as_iterable(frag)]
        self.vert_sources = vert and [GLSLSourceCode(s) for s in as_iterable(vert)]
        self.program = None
        self.program_kw = kw
        self.reloaded = None
        self.reload()

    def reload(self):
        logging.debug(f'reloading shader program last loaded at {self.reloaded}')
        self.reloaded = dt.datetime.now()
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
            # shaders not actually compiled by gloo until draw call
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
    """A 2D drawing layer.

    attrs:
        `program`: a LiveProgram shader
        `target`: an NBuffer render target

    Each set attribute will be stored and passed to every `draw` call,
    and when a Layer is passed to `draw`, it is automatically unwrapped to its
    target's state. Thus Layers can be patched together simply by setting the
    attribute matching a `uniform sampler2D` defined in the shader:
        ```
        layer_a.input_tex = layer_b
        layer_b.input_tex = layer_a
        ```
    builds a feedback loop between `layer_a` and `layer_b`.

    Furthermore all buffers `self.target.history` are automatically bound to
    uniforms of the form "history_t{i}_b{j}" (if they exist). So ping-ponging feedback is set up without patching provided `self.n >=2`

    Attributes may be set to an infinite iterator, causing each call to `draw`
    advance the iterator and use the returned value. A finite iterator will be
    automatically converted via `itertools.cycle`.
    """
    def __init__(self, size, shader, n=0, **buffer_args):
        self.program = LiveProgram(vert="""
            in vec2 position;
            out vec2 uv;
            void main(){
              gl_Position = vec4(position, 0.0, 1.0);
              uv = position*.5+.5;
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
        w = sum(len(s.outs) for s in self.program.frag_sources)
        self.target = NBuffer(size, n, w, **buffer_args)
        self.draw_kwargs = {}

    def draw(self, **kwargs):
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

    def __call__(self, **call_kwargs):
        """call `draw` with all stored arguments"""
        self.draw(**{k:next(v) for k,v in self.draw_kwargs.items()}, **call_kwargs)
        return self

    def resize(self, size):
        self.target.resize(size)
        self.program['size'] = size

    def __setattr__(self, k, v):
        """sugar: most attributes fall through to shader program."""
        if k in ['program', 'draw_method', 'target', 'draw_kwargs']:
            super().__setattr__(k, v)
        else:
            # store all uniforms in the Layer object as infinite iterators
            # (may store e.g. default values parsed from the source in the LiveProgram still)
            if not isinstance(v, it.cycle):
                v = it.cycle((v,))
            self.draw_kwargs[k] = v

    @property
    def state(self):
        return self.target.state

    @property
    def cpu(self):
        return self.target.cpu

class NBuffer(object):
    def __init__(self, size, n, w=1,
            autoread=False, short=False, channels=4,
            wrapping=gl.GL_REPEAT, interpolation=gl.GL_LINEAR):
        """Circular collection of FrameBuffer
        size: 2d dimensions in pixels
        n: number of framebuffers (time dimension):
            n=0 is a dummy
            n=1 is just an FBO
            n=2 ping-pongs (one history buffer available)
            n>2 increases available history (to n-1)
        w: number of colorbuffers per framebuffer (default 1)
        autoread:
            if True, replace cpu_state when deactivating.
            if False, replace when cpu property is requested since activating
        short:
            use int8 texture (otherwise float32)
        channels:
            number of color channels
        wrapping:
            gl.GL_CLAMP_TO_EDGE, GL_REPEAT, or GL_MIRRORED_REPEAT
        interpolation:
            gl.GL_NEAREST or GL_LINEAR

        An NBuffer has size[0]*size[1]*n*w*channels total pixels.
        """
        self.size = size
        self.dtype = np.uint8 if short else np.float32
        def gen_tex():
            ttype = gloo.Texture2D if short else gloo.TextureFloat2D
            tex = np.zeros((*size[::-1], channels), self.dtype).view(ttype)
            tex.interpolation = interpolation
            tex.wrapping = wrapping
            return tex
        self._state = [
            gloo.FrameBuffer(color=[
                gen_tex() for i in range(w)
            ]) for j in range(n)]
        self.head = 0
        self.n = n
        self.cpu_state = None
        self.autoread = autoread
        self.readback_buffer = 0

    def resize(self, size):
        """call resize for every constituent buffer"""
        self.size = size
        for buf in self._state:
            buf.resize(*size)

    def read(self, b):
        """read back pixels to numpy from buffer `b`."""
        assert self.n>0, "nothing to read from NBuffer of length 0"
        return self.state[b].get()

    @property
    def cpu(self):
        """return the cpu-side representation of the `readback_buffer`.
        call `read` to get it if necessary.
        """
        if self.cpu_state is None:
            self.cpu_state = self.read(self.readback_buffer)
        return self.cpu_state

    @property
    def state(self):
        """return the ColorBuffers under `self.head`.

        within the NBuffer's context manager, this is the oldest set of buffers
        (to be overwritten). outside the context manager, it is the most recently
        written (to be read).
        """
        return self._state[self.head].color if self.n else None

    @property
    def history(self):
        """return list of all ColorBuffers except at `self.head`, ordered from
        newest to oldest.
        e.g.:
            n = 2, inside context manager: history[0] is
        """
        idxs = (self.head-1-np.arange(self.n-1))%self.n
        return [self._state[i].color for i in idxs]

    def activate(self):
        """prepare to draw:
        increment self.head, invalidate the cpu rep and activate the render target
        """
        gl.glViewport(0, 0, *self.size)
        if self.n:
            # gl.glPushAttrib(gl.GL_VIEWPORT_BIT)
            self.head = (self.head+1)%self.n
            self._state[self.head].activate()
            if not self.autoread:
                self.cpu_state = None

    def deactivate(self):
        """deactivate render target"""
        if self.n:
            # gl.glPopAttrib(gl.GL_VIEWPORT_BIT)
            if self.autoread:
                self.cpu_state = self.read(self.readback_buffer)
            self._state[self.head].deactivate()

    #should readback and ping ponging happen only in context manager?
    def __enter__(self):
        self.activate()
    def __exit__(self, *args):
        self.deactivate()
    def __len__(self):
        return self.n

class Points(object):
    """adapted from glumpy example:
    https://github.com/glumpy/glumpy/blob/master/examples/gloo-trail.py
    """
    def __init__(self, size, n):
        self.segments = []
        self.target = NBuffer(size, 1)
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
        """segment is a tuple of (positions, colors, sizes)"""
        assert len(segment[0])==len(self.data)
        self.segments.append(segment)

    def draw(self):
        with self.target:
            gl.glViewport(0, 0, *self.target.size)
            for p,c,s in self.segments:
                self.data['a_position'][:] = p
                self.data['a_color'][:] = c
                self.data['a_size'][:] = s
                # gl.glClearColor(0,0,0,0.001)
                # gl.glEnable(gl.GL_BLEND)
                # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                self.program.draw(gl.GL_POINTS)
        self.segments = []

# fast serial processing using numba

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

@jit(nopython=True)
def shape2(a):
    return np.float32(a.shape[:2])

#TODO: how often are frames repeated?
@jitclass([
    ('next_terrain', numba.float32[:,:,:]),
    ('cur_terrain', numba.float32[:,:,:]),
    ('last_terrain', numba.float32[:,:,:]),
    ('p', numba.float32[:,:]),
    ('momentum', numba.float32[:,:]),
    ('mdecay', numba.float32),
    ('stepsize', numba.float32),
    ('t', numba.int64),
    # ('sr', numba.int64),
    ('t_cur_added', numba.int64),
    ('t_last_added', numba.int64),
    ('t_next_added', numba.int64),
    ('t_switched', numba.int64),
    ('n', numba.int64)
])
class VideoWaveTerrainJIT(object):
    def __init__(self, n):#, max_len=3):
        self.next_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.cur_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.last_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.p = np.random.random((n, 2)).astype(np.float32)
        self.momentum = np.zeros((n, 2)).astype(np.float32)
        self.mdecay = 0.99
        self.stepsize = 0.03
        self.t = 0
        # self.sr = 24000
        self.t_next_added = -1
        self.t_cur_added = -2
        self.t_last_added = -3
        self.t_switched = 0
        self.n = n

    def feed(self, frame):
        frame = np.concatenate((frame, frame[0:1]), 0)
        frame = np.concatenate((frame, frame[:,0:1]), 1)
        self.next_terrain = frame.astype(np.float32)
        self.t_next_added = self.t

    def switch(self):
        self.last_terrain = self.cur_terrain
        self.t_last_added = self.t_cur_added
        self.cur_terrain = self.next_terrain
        self.t_cur_added = self.t_next_added
        self.t_switched = self.t

    def get(self, p, m):
        return lerp(
            interp2d(self.last_terrain, p*(shape2(self.last_terrain)-1)),
            interp2d(self.cur_terrain, p*(shape2(self.cur_terrain)-1)),
            m
        )

    def step(self, n_steps):
        ps = np.empty((n_steps, self.n, 2), dtype=np.float32)
        cs = np.empty((n_steps, self.n, 4), dtype=np.float32)
        for i in range(n_steps):
            p, c = self._step()
            ps[i] = p
            cs[i] = c
        return ps, cs

    def _step(self):
        t = self.t - self.t_switched
        dur = self.t_cur_added - self.t_last_added
        if t >= dur:
            self.switch()
            m = 0.0
        else:
            m = np.minimum(1.,np.float32(t)/np.maximum(1., np.float32(dur)))
        c = np.empty((self.n, 4))
        for i in range(self.n):
            val = self.get(self.p[i], m)

            # delta = np.sin(np.pi*(val[:2] - val[2:]))
            delta = np.sin(val[:2]*2*np.pi)
            # delta = val[:2]-0.5 #move on (r, g)
            # delta /= np.linalg.norm(delta) + 1e-15

            r = i*np.pi*2/np.float32(self.n)
            x,y = np.cos(r), np.sin(r)
            delta = np.array([x*delta[0]-y*delta[1], x*delta[1]+y*delta[0]])

            self.momentum[i] = self.momentum[i]*self.mdecay + delta
            self.p[i] += self.momentum[i] / (np.linalg.norm(self.momentum[i]) + 1e-12) / (shape2(self.cur_terrain)-1) * self.stepsize
            c[i] = val
        self.p %= 1.0
        self.t += 1
        return self.p, c #(x,y), (b,a)

class VideoWaveTerrain(object):
    def __init__(self, size, frame_count, n=1, short=False, point_shader=None):
        self.frame_count = frame_count
        self.vwt = VideoWaveTerrainJIT(n)
        self.points = Points(size, frame_count*n)
        self.filtered = Layer(size, point_shader, n=2)

    def sound(self):
        try:
            ps, cs = self.vwt.step(self.frame_count)
            samps = np.ascontiguousarray(cs.mean(1)[:,2:]) # slice 2 channels, mean over voices
            ps = ps.reshape(-1,2)*2-1 # points expects (-1, 1) domain
            cs = cs.reshape(-1,4) # corresponding colors
            self.points.append((ps, cs, 8.))
        except Exception as e:
            logging.error(e, exc_info=True)
            samps = np.zeros((self.frame_count, 2))
        return samps

    def draw(self):
        self.points.draw()
        self.filtered.draw(color=self.points.target.state[0])

    def __getattr__(self, k):
        return getattr(self.vwt, k)

# sound tools on top of pyaudio

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

# interactivity tools

shells = []
def start_shell(ns):
    """start an IPython shell in a new thread with the given namespace"""
    global shells
    shell = IPython.terminal.embed.InteractiveShellEmbed()
    threading.Thread(target=shell.mainloop, kwargs={'local_ns': ns}).start()
    shells.append(shell)
    return shell

# recording tools

def init_imsave_mp(n_procs=6, max_tasks=12):
    global imsave_pool, max_imsave_tasks
    max_imsave_tasks = max_tasks
    imsave_pool = Pool(n_procs)

imsave_tasks = []
def imsave_mp(path, arr, compress_level=5):
    global imsave_tasks, imsave_pool
    # add a task to the pool
    imsave_tasks.append(imsave_pool.apply_async(
        imwrite, (path, arr), dict(compress_level=compress_level),
        error_callback=logging.error))
    # block until there are fewer than max_imsave_tasks
    while len(imsave_tasks) > max_imsave_tasks:
        imsave_tasks = [t for t in imsave_tasks if not t.ready()]
