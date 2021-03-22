__all__ = [
    "serialized_interface",
    "serialized_call",
    "SerializedConnection",
    "Service", "ThreadedServer", "ForkingServer", "ThreadPoolServer", "OneShotServer",
    "plac",
    "start_function_service",
    "BasicFunctionServiceConnection",
    "FunctionServiceConnection"
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
import sys
import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def serialize(a):
    try:
        return pyarrow.serialize(a).to_buffer().to_pybytes()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        return pickle.dumps(a)


def deserialize(b):
    try:
        return pyarrow.deserialize(b)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
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
        try:
            self._arg_parser = plac.parser_from(self._func)
        except TypeError:
            # mysterious behavior. worked when try twice
            self._arg_parser = plac.parser_from(self._func)
        self._arg_parser.func = None
        self._arg_parser.obj = None
        self._num_request = 0

    @serialized_interface
    def exposed_signature(self):
        return self._signature

    @serialized_interface
    def exposed_argparser(self):
        return self._arg_parser

    @serialized_interface
    def exposed_run(self, *args, **kwargs):
        with self._run_lock:
            self._num_request += 1
            try:
                logger.info(" - Request:", self._num_request, ": START")
                results = self._func(*args, **kwargs)
                logger.info(" - Request:", self._num_request, ": END")
            except:
                logger.warning(" - Request:", self._num_request, ": Interrupted")
                raise
            return results


def start_function_service(func: Callable, port: int):
    s = ThreadedServer(GenericFunctionService(func), port=port)
    s.start()


class BasicFunctionServiceConnection:
    def __init__(self, *args, **kwargs):
        self._conn = SerializedConnection(*args, **kwargs)

    def run(self, *args, **kwargs):
        self._conn.root.run(*args, **kwargs)


class FunctionServiceConnection(BasicFunctionServiceConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signature = self._conn.root.signature()
        self._arg_parser = self._conn.root.argparser()
        self._arg_parser.obj = self._arg_parser.func = self.run

    @property
    def arg_parser(self) -> plac.ArgumentParser:
        return self._arg_parser

    @property
    def signature(self) -> inspect.Signature:
        return self._signature

    def run_from_cmd_args(self, args=None):
        if args is None:
            args = sys.argv[1:]
        cmd, results = self.arg_parser.consume(args)
        return results
