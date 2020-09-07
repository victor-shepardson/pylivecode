"""pattern utilites on top of itertools"""

import itertools as it
import functools as ft
import operator, threading
from collections import defaultdict

def cycle(iterable, period=1):
    return it.cycle(it.chain.from_iterable(it.repeat(x, period) for x in iterable))

class Var(object):
    """convert everything to iterables & wrap primitives.
    arithmetic operations will construct computational graphs of variables.
    thread safe.
    """
    def __init__(self, v=0):
        self._lock = threading.Lock()
        self.set(v)
    def __next__(self):
        with self._lock:
            return next(self.v)
    def set(self, v):
        """Set the value, converting it to an infinite iterator"""
        with self._lock:
            if hasattr(v, '__next__'):
                if hasattr(v, '__len__'):
                    v = it.cycle(v)
            else:
                v = it.cycle((v,))
            self.v = v

class Op(Var):
    def __init__(self, *args, fn=None):
        #fn must be supplied, it's kw for syntactic reasons
        self.args = [Var(a) for a in args]
        self.fn = fn
    def __next__(self):
        return self.fn(*[next(a) for a in self.args])

# all operators where the Var is leftmost
for op_name in (
        '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
        '__mod__', '__pow__',
        '__and__', '__or__', '__xor__', '__lshift__', '__rshift__',
        '__abs__', '__neg__', '__pos__'
    ):
    setattr(Var, op_name, ft.partialmethod(Op, fn=getattr(operator, op_name)))
# binary operators where the Var is rightmost
for op_name in (
        '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
        '__mod__', '__pow__',
        '__and__', '__or__', '__xor__', '__lshift__', '__rshift__',
    ):
    r_op_name = '__r'+op_name[2:]
    def r_op(r,l,**kw):
        return Op(l,r,**kw)
    setattr(Var, r_op_name, ft.partialmethod(r_op, fn=getattr(operator, op_name)))


class Vars(defaultdict):
    """defaultdict of Var providing attr access.

    L = Layer(...)
    V = Vars()
    L.param = V.x  #L.param is patched to V.x (default 0).
    ...
    V.x = 1  #new value will be reflected in L

    """
    def __init__(self, **kwargs):
        super().__init__(Var)

        for k,v in kwargs.items():
            self.__setattr__(k, v)

    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k].set(v)
