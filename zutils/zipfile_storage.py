__all__ = [
    'ZipFileStorage'
]

import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pickle


class ZipFileStorage:
    def __init__(self, file: str, mode='r', *args, num_workers: int = 4, **kwargs):
        """
        refer the zipfile.ZipFile
        """
        self._mode = mode
        self._zf = zipfile.ZipFile(file, mode, *args, **kwargs)
        self._zipfile_executor = ThreadPoolExecutor(max_workers=1)
        self._pickle_executor = ThreadPoolExecutor(max_workers=num_workers)
        self._cache = dict()
        self._cache_access_lock = Lock()
        self._key2ext = dict()
        self._init_key2ext_mapping()

    def _init_key2ext_mapping(self):
        for fn in self._zf.namelist():
            ext = None
            if fn.endswith(".pkl"):
                ext = ".pkl"
            elif fn.endswith(".int.txt"):
                ext = ".int.txt"
            elif fn.endswith(".str.txt"):
                ext = ".str.txt"
            if ext:
                self._key2ext[fn[:-len(ext)]] = ext

    def __getitem__(self, key):
        key = str(key)
        with self._cache_access_lock:
            if key in self._cache:
                return self._cache[key][0]
        ext = self._key2ext[key]
        s = self._zf.read(key + ext)
        if ext == ".int.txt":
            value = int(s.decode(encoding='UTF-8'))
        elif ext == ".str.txt":
            value = s.decode(encoding='UTF-8')
        else:
            value = pickle.loads(s)
        return value

    def __setitem__(self, key, value):
        assert self._mode != 'r', "cannot set item in read mode"
        key = str(key)
        r = self._pickle_executor.submit(
            serialize_and_add_to_zip_queue, key, value, self._zipfile_executor,
            self._zf, self._cache, self._cache_access_lock
        )
        with self._cache_access_lock:
            self._cache[key] = value, r

    def batch_set(self, keys, values):
        for k, v in zip(keys, values):
            self[k] = v

    def open(self, key, mode='r', *args, **kwargs):
        key = str(key)
        return self._zf.open(key + self._key2ext[key], mode, *args, **kwargs)

    def keys(self):
        return self._key2ext.keys()

    def values(self):
        for k in self.keys():
            yield self[k]

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self._zf.infolist())

    def infolist(self):
        return self._zf.infolist()

    def join(self):
        with self._cache_access_lock:
            all_results = [r for _, r in self._cache.values()]
        for r in all_results:
            r.result()

    def __del__(self):
        self.join()
        self._pickle_executor.shutdown(wait=True)
        self._zipfile_executor.shutdown(wait=True)
        self._zf.close()


def serialize_and_add_to_zip_queue(
        key, value, zipfile_executor: ThreadPoolExecutor,
        zf: zipfile.ZipFile, cache: dict, cache_access_lock: Lock
):
    if isinstance(value, int):
        ext = ".int.txt"
        s = str(value).encode(encoding='UTF-8')
    elif isinstance(value, str):
        ext = ".str.txt"
        s = value.encode(encoding='UTF-8')
    else:
        ext = ".pkl"
        s = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    zipfile_executor.submit(add_to_zip, key, ext, s, zf, cache, cache_access_lock)


def add_to_zip(key, s, ext: str, zf: zipfile.ZipFile, cache: dict, cache_access_lock: Lock):
    zf.writestr(key + ext, s)
    with cache_access_lock:
        cache.pop(key)
