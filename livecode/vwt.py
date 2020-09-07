"""video wave terrain module accelerated with numba"""

import logging, threading
import numpy as np

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

from . pattern import Vars
from . graphics import Layer, Points

njit = jit(nopython=True, fastmath=True)
# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')

@njit
def lerp(a, b, m):
    return a*(1-m) + b*m

@njit
def interp2d(a, p):
    """interpolating ndarray access"""
    x_lo, x_m = divmod(p[0], 1)
    y_lo, y_m = divmod(p[1], 1)
    a = a[x_lo:x_lo+2, y_lo:y_lo+2]
    a = lerp(a[0], a[1], x_m)
    return lerp(a[0], a[1], y_m)

@njit
def shape2(a):
    return np.float32(a.shape[:2])

@njit
def norm2(v):
    return v / (np.sqrt(v[...,0]*v[...,0]+v[...,1]*v[...,1]) + 1e-12)

@njit
def rot2(v, theta):
    return v @ np.sin(theta+np.pi*np.array([[0.5, 0],[1, 0.5]]))
    # x,y = np.cos(theta), np.sin(theta)
    # return np.array([x*v[0]-y*v[1], x*v[1]+y*v[0]])
    # r = np.cos(np.array([0, -np.pi/2])+theta)
    # return np.array([r[0]*v[0]-r[1]*v[1], r[0]*v[1]+r[1]*v[0]])

#TODO: how often are frames repeated?
@jitclass([
    ('next_terrain', numba.float32[:,:,:]),
    ('cur_terrain', numba.float32[:,:,:]),
    ('last_terrain', numba.float32[:,:,:]),
    ('p', numba.float32[:,:]),
    ('c', numba.float32[:,:]),
    # ('buffer_len', numba.int64),
    # ('head', numba.int64),
    # ('pbuffer', numba.float32[:,:,:]),
    # ('cbuffer', numba.float32[:,:,:]),
    ('momentum', numba.float32[:,:]),
    ('pixel_size', numba.float32[:]),
    ('t', numba.int64),
    ('t_cur_added', numba.int64),
    ('t_last_added', numba.int64),
    ('t_next_added', numba.int64),
    ('t_switched', numba.int64),
    ('n', numba.int64)
])
class VideoWaveTerrainJIT(object):
    def __init__(self, n, buffer_len):#, max_len=3):
        # buffers:
        self.next_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.cur_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.last_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.p = np.random.random((n, 2)).astype(np.float32)
        self.c = np.zeros((n, 4), dtype=np.float32)
        # self.buffer_len = buffer_len
        # self.head = 0
        # self.pbuffer = np.empty((buffer_len, n, 2), dtype=np.float32)
        # self.cbuffer = np.empty((buffer_len, n, 4), dtype=np.float32)
        self.momentum = np.zeros((n, 2), dtype=np.float32)
        self.pixel_size = np.ones(2, dtype=np.float32)
        self.t = 0
        self.t_next_added = -1
        self.t_cur_added = -2
        self.t_last_added = -3
        self.t_switched = 0
        # immutable parameters:
        self.n = n

    def feed(self, frame):
        # stage a new frame; if it isn't consumed by `switch` before `feed` is
        # called again, it is lost
        # duplicate edges to simplify wrapping
        # frame = frame[np.arange(frame.shape[0]+1)%frame.shape[0]]
        # frame = frame[:, np.arange(frame.shape[0]+1)%frame.shape[0]]
        # frame = np.concatenate((frame, frame[0:1]), 0)
        # frame = np.concatenate((frame, frame[:,0:1]), 1)
        # stage the next frame of terrain
        self.next_terrain = frame#frame.astype(np.float32)
        self.t_next_added = self.t

    def switch(self):
        # consume the staged next frame of terrain
        # if a new one wasn't staged, the most recent will just be repeated
        self.last_terrain = self.cur_terrain
        self.t_last_added = self.t_cur_added
        self.cur_terrain = self.next_terrain
        self.t_cur_added = self.t_next_added
        self.t_switched = self.t
        self.pixel_size = 1/(shape2(self.cur_terrain)-1)

    def get(self, p, m):
        # sample the terrain volume at postion p and fractional distance m between
        # last and current terrains
        return lerp(
            interp2d(self.last_terrain, p*(shape2(self.last_terrain)-1)),
            interp2d(self.cur_terrain, p*(shape2(self.cur_terrain)-1)),
            m
        )

    def step(self, n_steps, mdecay, stepsize):
        # run for n_steps and return positions + colors for each agent
        # currently allocates the ps and cs arrays each time
        # to avoid that, would need to fix the maximum steps here, and maximumum
        # segement length/pending segment number in Points
        # or could allocate up front and grab a segment here each time (circular buffer -- would need a warning to diagnose overrun)
        ps = np.empty((n_steps, self.n, 2), dtype=np.float32)
        cs = np.empty((n_steps, self.n, 4), dtype=np.float32)
        # if self.head + n_steps > self.buffer_len:
        #     self.head = 0
        # ps = self.pbuffer[self.head:self.head+n_steps]
        # cs = self.cbuffer[self.head:self.head+n_steps]
        # self.head += n_steps

        for i in range(n_steps):
            self._step(mdecay, stepsize)
            ps[i] = self.p
            cs[i] = self.c
        return ps, cs

    def _step(self, mdecay, stepsize):
        # run a single step for each agent
        t = self.t - self.t_switched
        # duration of current frame is estimated by duration of previous frame
        dur = self.t_cur_added - self.t_last_added
        # once as much time has passed as between previous two frames, consume
        # the staged frame (if there is one)
        if t >= dur:# and self.t_next_added > self.t_cur_added:
            self.switch()
            m = 0.0
        else:
            m = np.minimum(1.,np.float32(t)/np.maximum(1., np.float32(dur)))
        # loop over agents
        for i in range(self.n):
            self.c[i] = self.get(self.p[i], m)
            # val = self.get(self.p[i], m)
            # self.c[i] = (self.c[i]-0.5)*mdecay + (1-mdecay)*(val-0.5) + 0.5
        #separating these loops could help compiler to vectorize?
        for i in range(self.n):
            delta = self.c[i, :2]-0.5 #move on (r, g)
            # rotate delta by agent index
            theta = i*np.pi*2/np.float32(self.n)
            delta = rot2(delta, theta)

            self.momentum[i] = self.momentum[i]*mdecay + delta
            self.p[i] += (
                norm2(self.momentum[i]) * self.pixel_size * stepsize
                )
        self.p %= 1.0
        self.t += 1

class VideoWaveTerrain(object):
    def __init__(self, size, frame_count,
            n=1, short=False, point_shader=None, buffer_len=8192
        ):
        self.frame_count = frame_count
        self.vwt = VideoWaveTerrainJIT(n, buffer_len)
        self.points = Points(size, frame_count*n)
        self.filtered = Layer(size, point_shader, n=2)
        self.step_vars = Vars()
        self._lock = threading.Lock()

    def sound(self):
        with self._lock:
            try:
                # step_kwargs = {k:next(v) for k,v in self.step_vars.items()}
                # print(step_kwargs)
                ps, cs = self.vwt.step(
                    self.frame_count,
                    next(self.step_vars.mdecay),
                    next(self.step_vars.stepsize)
                    )
                # slice 2 channels, mean over voices
                samps = cs[:,:,2:].mean(1)#np.ascontiguousarray(cs[:,:,2:].mean(1))
                # collapse time, agent dimensions
                ps = ps.reshape(-1,2)*2-1 # points expects (-1, 1) domain
                cs = cs.reshape(-1,4) # corresponding colors
                self.points.append((ps, cs, 8.))
            except Exception as e:
                logging.error(e, exc_info=True)
                samps = np.zeros((self.frame_count, 2))
            return samps

    def draw(self):
        self.points.draw()
        self.filtered(color=self.points.target.state[0])

    def feed(self, frame):
        # duplicate edges to simplify wrapping
        fs = frame.shape
        frame = (frame
            [np.arange(fs[0]+1)%fs[0]]
            [:, np.arange(fs[1]+1)%fs[1]]
            .astype(np.float32))
        with self._lock:
            self.vwt.feed(frame)

    # def __getattr__(self, k):
        # if k in ('feed',):
            # return getattr(self.vwt, k)
        # return self.__dict__[k]

    def __setattr__(self, k, v):
        if k in ('mdecay', 'stepsize'):
            self.step_vars[k].set(v)
        else:
            super().__setattr__(k, v)
