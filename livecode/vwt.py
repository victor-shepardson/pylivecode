"""video wave terrain module accelerated with numba"""

import logging
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

from . graphics import Layer, Points

njit = jit(nopython=True, fastmath=True)

@njit
def lerp(a, b, m):
    return a*(1-m) + b*m

@njit
def _mcoords(x):
    x_ = int(x)
    return x_, x_+1, x-x_

@njit
def interp2d(a, p):
    """interpolating ndarray access"""
    x_lo, x_hi, x_m = _mcoords(p[0])
    y_lo, y_hi, y_m = _mcoords(p[1])
    a = a[x_lo:x_hi+1, y_lo:y_hi+1]
    a = lerp(a[0], a[1], x_m)
    return lerp(a[0], a[1], y_m)

@njit
def shape2(a):
    return np.float32(a.shape[:2])

@njit
def norm2(v):
    return v / (np.sqrt(v[0]*v[0]+v[1]*v[1]) + 1e-12)

#TODO: how often are frames repeated?
@jitclass([
    ('next_terrain', numba.float32[:,:,:]),
    ('cur_terrain', numba.float32[:,:,:]),
    ('last_terrain', numba.float32[:,:,:]),
    ('p', numba.float32[:,:]),
    ('c', numba.float32[:,:]),
    ('momentum', numba.float32[:,:]),
    ('pixel_size', numba.float32[:]),
    ('t', numba.int64),
    ('t_cur_added', numba.int64),
    ('t_last_added', numba.int64),
    ('t_next_added', numba.int64),
    ('t_switched', numba.int64),
    ('mdecay', numba.float32),
    ('stepsize', numba.float32),
    ('n', numba.int64)
])
class VideoWaveTerrainJIT(object):
    def __init__(self, n):#, max_len=3):
        # buffers:
        self.next_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.cur_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.last_terrain = np.zeros((2,2,4), dtype=np.float32)
        self.p = np.random.random((n, 2)).astype(np.float32)
        self.c = np.zeros((n, 4), dtype=np.float32)
        self.momentum = np.zeros((n, 2), dtype=np.float32)
        self.pixel_size = np.ones(2, dtype=np.float32)
        self.t = 0
        self.t_next_added = -1
        self.t_cur_added = -2
        self.t_last_added = -3
        self.t_switched = 0
        # mutable parameters:
        self.mdecay = 0.95
        self.stepsize = 0.5
        # immutable parameters:
        self.n = n

    def feed(self, frame):
        # stage a new frame; if it isn't consumed by `switch` before `feed` is
        # called again, it is lost
        # duplicate edges to simplify wrapping
        frame = np.concatenate((frame, frame[0:1]), 0)
        frame = np.concatenate((frame, frame[:,0:1]), 1)
        # stage the next frame of terrain
        self.next_terrain = frame.astype(np.float32)
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

    def step(self, n_steps):
        # run for n_steps and return positions + colors for each agent
        # currently allocates an array each time, probably shouldn't do that
        ps = np.empty((n_steps, self.n, 2), dtype=np.float32)
        cs = np.empty((n_steps, self.n, 4), dtype=np.float32)
        for i in range(n_steps):
            # p, c = self._step()
            # ps[i] = p
            # cs[i] = c
            self._step()
            ps[i] = self.p
            cs[i] = self.c
        return ps, cs

    def _step(self):
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
        # again, allocating arrays here is probably bad
        # c = np.empty((self.n, 4))
        for i in range(self.n):
            self.c[i] = self.get(self.p[i], m)

            val = self.c[i]
            delta = val[:2]-0.5 #move on (r, g)
            # delta = norm2(val[:2]-0.5) #move on (r, g)

            # rotate delta by agent index
            r = i*np.pi*2/np.float32(self.n)
            x,y = np.cos(r), np.sin(r)
            delta = np.array([x*delta[0]-y*delta[1], x*delta[1]+y*delta[0]])

            self.momentum[i] = self.momentum[i]*self.mdecay + delta
            self.p[i] += (
                norm2(self.momentum[i]) * self.pixel_size * self.stepsize
                )
        self.p %= 1.0
        self.t += 1

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

    def __getattr__(self, k):
        return getattr(self.vwt, k)

    def __setattr__(self, k, v):
        if k in ('frame_count', 'vwt', 'points', 'filtered'):
            super().__setattr__(k, v)
        elif k in ('mdecay', 'stepsize'):
            setattr(self.vwt, k, v)
