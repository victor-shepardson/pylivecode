"""graphics tools on top of glumpy"""

import sys, logging, re
import itertools as it
import datetime as dt
from queue import Queue
import numpy as np
import torch

from glumpy import gloo, gl, library
from glumpy.graphics.collections import PathCollection

from . utils import *
from . pattern import Var

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

    def is_path(self, sh):
        """assume source code of containing '{', else assume path"""
        return isinstance(sh, str) and '{' not in sh

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

class CPULayer(Var):
    def __init__(self, size, closure, **buffer_args):
        w = 1
        self.buffer = NBuffer(size, 1, w, **buffer_args)
        self.closure = closure #if isinstance(closure, Var) else Var(closure)

    def draw(self):
        if self.closure is not None:
            try:
                self.closure(self.buffer.state[0])
            except Exception as e:
                logging.error(e)
                self.closure = None
        # next(self.closure)(self.buffer.state[0])

    @property
    def state(self):
        return self.buffer.state

    def __next__(self):
        return self.state[0]

    def __call__(self):
        self.draw()
        return self


class Layer(Var):
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
    to advance the iterator and use the returned value. A finite iterator will
    be automatically converted via `itertools.cycle`.
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
                # if isinstance(v, Layer):
                    # v = v.state[0]
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

    def __call__(self, **draw_kwargs):
        """call `draw` with all stored arguments"""
        self.draw(
            **{k:next(v) for k,v in self.draw_kwargs.items()},
            **draw_kwargs
            )
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
            # if not isinstance(v, it.cycle):
                # v = it.cycle((v,))
            if not isinstance(v, Var):
                v = Var(v)
            self.draw_kwargs[k] = v

    def __next__(self):
        return self.state[0]

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
        # self.dtype = np.uint8 if short else np.float32
        self.dtype = torch.uint8 if short else torch.float32
        def gen_tex():
            ttype = gloo.Texture2D if short else gloo.TextureFloat2D
            # tex = np.zeros((*size[::-1], channels), self.dtype).view(ttype)
            tex = torch.zeros((*size[::-1], channels), dtype=self.dtype).numpy().view(ttype)

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
        self.segments = Queue()
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
        self.segments.put(segment)

    def draw(self):
        # consume all segments present at start of drawing
        with self.target:
            gl.glViewport(0, 0, *self.target.size)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            n = self.segments.qsize()
            for _ in range(n):
                p,c,s = self.segments.get()
                self.data['a_position'][:] = p
                self.data['a_color'][:] = c
                self.data['a_size'][:] = s
                # gl.glClearColor(0,0,0,0.001)
                # gl.glEnable(gl.GL_BLEND)
                # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                self.program.draw(gl.GL_POINTS)
