__all__ = [
    'ZipFileStorage',
    'PickledBytes',
    'advanced_serialize',
    'advanced_deserialize',
]

import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pickle
from typing import Iterable, Optional, Callable, Any, Dict
from functools import partial
from z_python_utils.classes import SizedWrapperOfIterable
from zutils.async_executors import SubmitterThrottle

try:
    import pyarrow
    _has_pyarrow = True
except ModuleNotFoundError:
    _has_pyarrow = False


if _has_pyarrow:

    def advanced_serialize(a):
        try:
            return pyarrow.serialize(a).to_buffer().to_pybytes()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            return pickle.dumps(a)

    def advanced_deserialize(b):
        try:
            return pyarrow.deserialize(b)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            return pickle.loads(b)


class PickledBytes(bytes):
    pass


class ZipFileStorage:
    def __init__(
            self, file: str, mode='r', *args, num_workers: int = 4,
            serialization_func: Callable[[Any], bytes] = None, deserialization_func: Callable[[bytes], Any] = None,
            use_advanced_serialization: bool = False,
            async_buffer_size: int = 256,
            **kwargs
    ):
        """
        refer the zipfile.ZipFile
        """
        self._mode = mode
        self._zf = zipfile.ZipFile(file, mode, *args, **kwargs)
        self._zipfile_executor = ThreadPoolExecutor(max_workers=1)
        self._pickle_executor = ThreadPoolExecutor(max_workers=num_workers)
        self._pickle_submitter = SubmitterThrottle(self._pickle_executor, async_buffer_size)
        self._cache = dict()
        self._cache_access_lock = Lock()
        self._set_item_lock = Lock()
        self._zip_lock = Lock()
        self._key2ext = dict()
        self._init_key2ext_mapping()
        self._prefetch_size = num_workers * 2
        self._is_closed = False
        if use_advanced_serialization:
            default_serialization_func = advanced_serialize
            default_deserialization_func = advanced_deserialize
        else:
            default_serialization_func = partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)
            default_deserialization_func = pickle.loads

        self._serialization_func = (
            serialization_func if serialization_func else default_serialization_func
        )
        self._deserialization_func = deserialization_func if deserialization_func else default_deserialization_func

    def _init_key2ext_mapping(self):
        for fn in self._zf.namelist():
            ext = None
            if fn.endswith(".pkl"):
                ext = ".pkl"
            elif fn.endswith(".int.txt"):
                ext = ".int.txt"
            elif fn.endswith(".str.txt"):
                ext = ".str.txt"
            elif fn.endswith(".bytes"):
                ext = ".bytes"
            if ext:
                self._key2ext[fn[:-len(ext)]] = ext

    def __getitem__(self, key):
        key = str(key)
        ext = self._key2ext[key]
        value = deserialize_from_zip(
            key, ext, self._zf, self._cache, self._cache_access_lock, self._zip_lock, self._deserialization_func
        )
        return value

    def __setitem__(self, key, value):
        with self._set_item_lock:
            self._set_item(key, value)

    def _set_item(self, key, value):
        assert self._mode != 'r', "cannot set item in read mode"
        key = str(key)
        with self._cache_access_lock:
            self._cache[key] = cached_info = [value, None]
        r = self._pickle_submitter.submit(
            serialize_and_add_to_zip_queue, key, value, self._zipfile_executor,
            self._zf, self._cache, self._cache_access_lock, self._zip_lock,
            self._key2ext, self._serialization_func
        )
        cached_info[1] = r

    def batch_set(self, keys, values):
        for k, v in zip(keys, values):
            self[k] = v

    def batch_set_from_pairs(self, key_value_pairs):
        for k, v in key_value_pairs:
            self[k] = v

    def open(self, key, mode='r', *args, **kwargs):
        key = str(key)
        return self._zf.open(key + self._key2ext[key], mode, *args, **kwargs)

    def keys(self):
        return self._key2ext.keys()

    def __contains__(self, item):
        return str(item) in self.keys()

    def values(self):
        return SizedWrapperOfIterable(self._values(), len(self))

    def _values(self):
        for _, value in self.iterate_items(self.keys()):
            yield value

    def items(self):
        return self.iterate_items(self.keys())

    def __iter__(self):
        return self.keys()

    def __len__(self):
        with self._zip_lock, self._cache_access_lock:
            return len(self._zf.infolist()) + len(self._cache)

    def iterate_items(self, keys: Iterable):
        item_iterable = self._iterate_items(keys)
        return SizedWrapperOfIterable(item_iterable, len(self))

    def _iterate_items(self, keys: Iterable):
        prefetched = dict()

        def _get_prefetched_item(_j):
            _key, _r = prefetched.pop(_j)
            _value = _r.result()
            return _key, _value

        n = j = 0
        for key in keys:

            key = str(key)

            ext = self._key2ext[key]
            prefetched[n] = key, self._pickle_submitter.submit(
                deserialize_from_zip,
                key, ext, self._zf, self._cache, self._cache_access_lock, self._zip_lock,
                self._deserialization_func,
                self._zipfile_executor
            )

            n = n + 1
            if n > self._prefetch_size:
                yield _get_prefetched_item(j)
                j += 1

        for i in range(j, n):
            yield _get_prefetched_item(i)

    def infolist(self):
        return self._zf.infolist()

    def join(self):
        with self._cache_access_lock:
            all_results = [r for _, r in self._cache.values()]
        for r in all_results:
            if r is not None:
                r.result()

    def close(self):
        if self._is_closed:
            return
        self.join()
        self._pickle_executor.shutdown(wait=True)
        self._zipfile_executor.shutdown(wait=True)
        self._zf.close()
        self._is_closed = True
        self._pickle_executor = None
        self._zipfile_executor = None

    def __del__(self):
        self.close()


def serialize_and_add_to_zip_queue(
        key, value, zipfile_executor: ThreadPoolExecutor,
        zf: zipfile.ZipFile, cache: dict, cache_access_lock: Lock, zip_lock: Lock,
        key2ext: Dict[str, str],
        serialization_func: Callable[[Any], bytes]
):
    if isinstance(value, int):
        ext = ".int.txt"
        s = str(value).encode(encoding='UTF-8')
    elif isinstance(value, str):
        ext = ".str.txt"
        s = value.encode(encoding='UTF-8')
    elif isinstance(value, PickledBytes):
        ext = ".pkl"
        s = value
    elif isinstance(value, bytes):
        ext = ".bytes"
        s = value
    else:
        ext = ".pkl"
        s = serialization_func(value)
    with cache_access_lock:
        key2ext[key] = ext
    zipfile_executor.submit(add_to_zip, key, ext, s, zf, cache, cache_access_lock, zip_lock)


def add_to_zip(
        key, ext: str, s, zf: zipfile.ZipFile, cache: dict, cache_access_lock: Lock, zip_lock: Lock
):
    with zip_lock:
        zf.writestr(str(key) + ext, s)
    with cache_access_lock:
        cache.pop(key)


def read_from_zip(key, ext: str, zf: zipfile.ZipFile, zip_lock: Lock):
    with zip_lock:
        s = zf.read(str(key) + ext)
    return s


def deserialize_from_zip(
        key: str, ext: str, zf: zipfile.ZipFile, cache: dict, cache_access_lock: Lock, zip_lock: Lock,
        deserialization_func: Callable[[bytes], Any],
        zipfile_executor: Optional[ThreadPoolExecutor] = None
):
    with cache_access_lock:
        if key in cache:
            return cache[key][0]

    if zipfile_executor is None:
        s = read_from_zip(key, ext, zf, zip_lock)
    else:
        # make sure to do unzip in a single thread
        r = zipfile_executor.submit(zf.read, key, ext, zf, zip_lock)
        s = r.result()
    if ext == ".int.txt":
        value = int(s.decode(encoding='UTF-8'))
    elif ext == ".str.txt":
        value = s.decode(encoding='UTF-8')
    elif ext == ".bytes":
        value = s
    else:
        value = deserialization_func(s)
    return value
