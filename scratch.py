import numpy as np
from glumpy import app
from livecode import *

size = np.array((900, 1800))
screen_size = np.array((1600, 900))
win_size = screen_size*2

def get_shaders(s):
    return ('shader/lib.glsl', 'shader/'+s+'.glsl')

screen = Layer(screen_size, get_shaders('display-stretch'))
# post = Layer(
    # screen_size, get_shaders('display_zoetrope'), n=1, short=True)
post = Layer(
    screen_size, get_shaders('display_zoetrope'), n=1, short=True)
feedback = Layer(size, get_shaders('feedback2'), n=3,)
filtered = Layer(size, get_shaders('filter'), n=2)

filtered.color = feedback
feedback.filtered = filtered
post.color = feedback
post.src_size = size
screen.color = post

feedback.drag = 0.97

capture = Capture()

frame = 0
def image():
    global frame
    filtered()
    feedback()
    post(frame=frame)
    screen()
    capture.do()
    frame += 1

window = make_window(image, win_size, title='scratch')
@window.event
def on_resize(w,h):
    screen.resize((w,h))

# window.attach(post.program['viewport'])

shell = start_shell(locals())

app.run()
