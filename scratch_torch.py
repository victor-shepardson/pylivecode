import sys, logging
import numpy as np
from glumpy import app, gl
import torch

from livecode import *

# size = 900, 900
size = 1920, 1920
size = np.array(size)

# global primitives need to be wrapped in a Var even if they aren't patched,
# since shell runs in its own thread -- locals() will copy primitives
# they have to be accessed with e.g. next(V.gain)
V = Vars()
# V.gain = 0

def get_shaders(s):
    return ('shader/lib.glsl', 'shader/'+s+'.glsl')

# initialization

def closure(pix):
    np_pix = pix
    import torch
    pix = torch.from_numpy(pix)
    idxs = torch.rand(pix.shape)>pix.mean(-1,keepdim=True)
    pix[idxs] = (pix[idxs.roll(1,1)]+.998)%1
    m = torch.rand(pix.shape[0])[:,None,None]**2
    pix = pix*(1-m) + m*pix.roll((1,1),(0,2))
    pix[:,:,3] = (pix.max(-1).values+0.997)%1
    pix[:,:,1] = (pix.min(-1).values+0.003)%1
    np_pix[:] = pix.numpy()

feedback = CPULayer((320, 180), closure, interpolation=gl.GL_NEAREST)

screen = Layer(size, get_shaders('display'))

# patching
# feedback.filtered = filtered
screen.color = feedback

# feedback.drag = (M.cc1/127)**0.25
# M.cc1 = 100

# draw order
def image():
    feedback()
    screen()


# start app
window = make_window(image, size, title='torch')
@window.event
def on_resize(w,h):
    screen.resize((w,h))

shell = start_shell(locals())

app.run()
