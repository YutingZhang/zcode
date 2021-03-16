__all__ = [
    "serialized_interface",
    "serialized_call",
    "Service",
    "SerializedConnection",
]

import rpyc
import pyarrow
from typing import Callable, Union
from functools import partial


Service = rpyc.Service


def serialize(a):
    return pyarrow.serialize(a).to_buffer().to_pybytes()


def deserialize(b):
    return pyarrow.deserialize(b)


def serialized_interface(f: Callable):      # modifier
    def wrapped_f(*b):
        args, kwargs = deserialize(b[-1])  # allow with self or not
        r = f(*b[:-1], *args, **kwargs)
        return serialize(r)
    return wrapped_f


def serialized_call(*args, **kwargs):
    f = args[0]
    args = tuple(args[1:])
    kwargs = dict(kwargs)
    b = serialize((args, kwargs))
    f: Callable
    s = f(b)
    r = deserialize(s)
    return r


class SerializedConnection:
    def __init__(self, hostname: str, port: int):
        self._conn = rpyc.connect(hostname, port)

    @property
    def root(self):
        return _SerializedConnectionRoot(self._conn)


class _SerializedConnectionRoot:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, item):
        if not hasattr(self._conn.root, item):
            raise KeyError(item)
        func = getattr(self._conn.root, item)
        return partial(serialized_call, func)
