__all__ = [
    "serialized_interface",
    "serialized_call",
    "SerializedConnection",
    "Service", "ThreadedServer", "ForkingServer", "ThreadPoolServer", "OneShotServer"
    "plac",
    "start_function_service",
]

import rpyc
from rpyc import Service, ThreadedServer, ForkingServer, ThreadPoolServer, OneShotServer
import pyarrow
from typing import Callable
from functools import partial
import time
import pickle
import inspect
import plac
from threading import Lock


def serialize(a):
    try:
        return pyarrow.serialize(a).to_buffer().to_pybytes()
    except (KeyboardInterrupt, SystemExit):
        return pickle.dumps(a)


def deserialize(b):
    try:
        return pyarrow.deserialize(b)
    except (KeyboardInterrupt, SystemExit):
        return pickle.loads(b)


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
    def __init__(self, hostname: str, port: int, retry_interval_until_success: float = -1, timeout: float = None):
        config = dict(
            sync_request_timeout=timeout
        )
        if retry_interval_until_success <= 0:
            self._conn = rpyc.connect(hostname, port, config=config)
        else:
            while True:
                try:
                    self._conn = rpyc.connect(hostname, port, config=config)
                    break
                except ConnectionError:
                    time.sleep(retry_interval_until_success)

    @property
    def raw_root(self):
        return self._conn.root

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


class GenericFunctionService(Service):
    def __init__(self, func: Callable):
        super().__init__()
        self._func = func
        self._run_lock = Lock()
        self._signature = inspect.signature(self._func)
        self._arg_parser = plac.parser_from(self._func)

    @serialized_interface
    def exposed_signature(self):
        return self._signature

    @serialized_interface
    def exposed_argparser(self):
        return self._arg_parser

    @serialized_interface
    def exposed_run(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def start_function_service(func: Callable, port: int):
    s = ThreadedServer(GenericFunctionService(func), port=port)
    s.start()


