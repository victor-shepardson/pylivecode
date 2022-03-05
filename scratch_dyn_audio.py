from queue import Queue
import numpy as np
from glumpy import app, gl
from livecode import *

import numba

size = np.array((1600, 900))
screen_size = np.array((1600, 900))
paths_size = np.array((128, 128)) # serial, parallel
win_size = screen_size*2

sample_rate = 48000
frame_count = paths_size[0]

Q = Queue()

def get_shaders(s):
    return ('shader/lib.glsl', 'shader/'+s+'.glsl')

# screen = Layer(screen_size, get_shaders('display-dyn'))
screen = Layer(screen_size, get_shaders('display-stretch'))
# screen = Layer(screen_size, get_shaders('display-paths-debug'))

feedback = Layer(size, get_shaders('feedback-trails'), n=3)
filtered = Layer(size, get_shaders('filter'), n=2)
paths = Layer(paths_size, get_shaders('paths'), 
    n=1, scan=True, interpolation=gl.GL_NEAREST)
points = Points(size*2, paths_size[0]*paths_size[1])
# trails = Layer(size, get_shaders('filter-accum'), n=2)


filtered.color = feedback
feedback.filtered = filtered
feedback.aux = points
paths.src = feedback
paths.src_size = size
screen.color = feedback
# screen.color = paths
# trails.color = points
# screen.color = trails
# screen.trails = trails
# screen.terrain = feedback

feedback.drag = 0.97
# trails.decay = 0.8

capture = Capture()

frame = 0
colors = np.ones((paths_size[0]*paths_size[1], 4))
sizes = np.ones((paths_size[0]*paths_size[1])) * 3
def image():
    global frame
    filtered()
    feedback()
    paths.frame = frame
    paths()

    positions = paths.cpu[...,:2].reshape(-1, 2)
    positions*=2
    positions-=1
    points.append((positions, colors, sizes))
    points.draw()

    voices = paths.cpu[...,:2].transpose(1,0,2)
    Q.put(np.power(voices, 4.) * np.pi / 8)

    screen()
    capture.do()
    frame += 1

buf = np.zeros((frame_count, 2), dtype=np.float32)
path_block = np.zeros((*paths_size, 2), dtype=np.float32)
phase = np.zeros((paths_size[1], 2), dtype=np.float32)
path_every = 1
block_counter = 0
def sound():
    global block_counter, path_every, path_block
    # consume 1 paths block per n audio blocks
    # if queue is large, increase n
    # if queue is small, decrease n 
    qs = Q.qsize()
    if qs < 2:
        path_every += 1
    elif qs > 8:
        path_every = max(path_every - 1, 1)
    if block_counter >= path_every and qs > 0:
        path_block = Q.get()

        block_counter = 0 

    # index paths block according to block_counter
    # path_idx = min(
    #     int(block_counter / path_every * paths_size[0]),
    #     paths_size[0])

    # voices = path_block[path_idx]

    sound_block(path_block, phase, buf, block_counter, path_every)
    # print(qs, path_every, path_idx)   

    block_counter += 1

    return buf

# idea: left/right as harmonic spaces (comb filters?)
#       up/down as frequency

@numba.njit(numba.void(
    numba.float32[:,:,:], numba.float32[:,:], numba.float32[:,:], 
    numba.int64, numba.int64))
def sound_block(
    path_block, phase, buf, 
    block_counter, path_every):
    for i in range(frame_count):
        path_idx = min(int(
            (block_counter * frame_count + i) 
            / (path_every * paths_size[0]) 
            * paths_size[0]), paths_size[0]-1)
        phase += path_block[path_idx]
        buf[i,:] = np.mean(np.sin(phase))


window = make_window(image, win_size, title='scratch')
@window.event
def on_resize(w,h):
    screen.resize((w,h))

# window.attach(post.program['viewport'])

stream = make_stream(
    sound, frame_count, channels=2, sample_rate=sample_rate)
stream.start_stream()

# shell = start_shell(locals())

app.run()
