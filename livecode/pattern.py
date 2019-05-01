"""pattern utilites on top of itertools"""

import itertools as it
from collections import defaultdict

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
