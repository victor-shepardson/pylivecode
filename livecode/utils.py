"""Tools for handling files, filesystem events, source code, recording"""

import os, logging
import datetime as dt
from collections import defaultdict, Iterable
from multiprocessing import Pool

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    from imageio import imwrite
except ImportError:
    logging.warning('image saving unvailable; install imageio')

def as_iterable(arg):
    return arg if isinstance(arg, Iterable) and not isinstance(arg, str) else (arg,)

# file watching/reloading tools

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
        if self.is_path(arg):
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
        """override to add language-specific behavior"""
        pass

    def is_path(self, s):
        """override to allow source code as str input"""
        return True

# recording tools
# TODO: combine imsave_mp with Capture

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

class Capture(object):
    """single frame capture. use capture.do() in the draw loop,
    capture.__call__(layer) in the REPL.
    """
    def __init__(self):
        self.needs_capture = False
    def __call__(self, layer):
        self.needs_capture = True
        self.layer = layer
    def do(self):
        if self.needs_capture:
            imwrite(f'capture/{dt.datetime.now()}.png', self.layer.cpu)
            self.needs_capture = False
