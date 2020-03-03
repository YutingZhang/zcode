from collections import OrderedDict
import os
import pyarrow
import random
import snappy
import shutil
from typing import Callable
from z_python_utils.io import mkdir_p
from threading import Lock, Thread
import time


__all__ = [
    'folder_to_record',
    "IndexedRecord",
    "FolderToIndexedRecord"
]


# serializer
def dumps_single_object(a):
    s = pyarrow.serialize(a).to_buffer().to_pybytes()
    s = snappy.compress(s)
    return s


def loads_single_object(s):
    s = snappy.decompress(s)
    a = pyarrow.deserialize(s)
    return a


def folder_to_record(src_folder: str, record_path: str, show_progress_bar: bool = False):
    ftl = FolderToIndexedRecord(src_folder, record_path)
    ftl.sync(show_progress_bar=show_progress_bar)


def default_encoder(a):
    return a


def default_decoder(s):
    return s


class IndexedRecord:
    def __init__(
            self, record_path: str, write: bool = False,
            encoder: Callable[[object], bytes] = default_encoder,
            decoder: Callable[[object], bytes] = default_decoder,
            verbose: bool = False,
    ):
        self._lock = Lock()
        self._record_path = os.path.abspath(record_path)
        self._write = write
        self._key2meta = OrderedDict()
        self._modified = False
        self._data_file = None
        self._is_data_file_pos_end = False
        self._encoder = encoder
        self._decoder = decoder
        self._verbose = verbose

        if self._write:
            mkdir_p(self._record_path)
            if not os.path.exists(self.index_file_path) or not os.path.exists(self.data_file_path):
                with open(self.index_file_path, 'wb') as f:
                    f.write(dumps_single_object([]))
                with open(self.data_file_path, 'wb'):
                    pass

        if self._verbose:
            print('[Record: %s] open' % self._record_path)

        self.sync_from_disk()

    def __del__(self):
        self.sync_indexes_to_disk()

    def sync_from_disk(self):

        if self._verbose:
            print('[Record: %s] sync from disk' % self._record_path)

        with self._lock:
            assert os.path.exists(self.index_file_path), 'index file does not exist: %s' % self.index_file_path
            assert os.path.exists(self.data_file_path), 'data file does not exist: %s' % self.data_file_path

            if self._write:
                self._data_file = open(self.data_file_path, 'rb+')
            else:
                self._data_file = open(self.data_file_path, 'rb')
            self._is_data_file_pos_end = False

            key2meta = loads_single_object(open(self.index_file_path, 'rb').read())
            assert isinstance(key2meta, list), "corrupted dataset, index is not a list of tuples"
            key2meta = OrderedDict(key2meta)
            self._key2meta = key2meta
            self._modified = False

    @property
    def index_file_path(self) -> str:
        return os.path.join(self._record_path, 'index')

    @property
    def data_file_path(self) -> str:
        return os.path.join(self._record_path, 'data')

    def sync_indexes_to_disk(self, force: bool = False):
        if not force and not self._modified:
            return

        if self._verbose:
            print('[Record: %s] dump indexes' % self._record_path)

        with self._lock:
            with open(self.index_file_path, 'wb') as f:
                f.write(dumps_single_object(list(self._key2meta.items())))
            self._modified = False

    def _df_seek_end(self):
        if self._is_data_file_pos_end:
            return
        self._data_file.seek(0, 2)
        self._is_data_file_pos_end = True

    def _df_seek_abs(self, loc):
        if not self._data_file.tell() == loc:
            self._data_file.seek(loc, 0)
        self._is_data_file_pos_end = False

    def __getitem__(self, key):
        with self._lock:
            if key not in self._key2meta:
                raise KeyError("Key does not exists: %s" % key)
            start_loc, data_length = self._key2meta[key][-1]
            self._df_seek_abs(start_loc)
            s = self._data_file.read(data_length)
        a = self._decoder(s)
        return a

    def __setitem__(self, key, value):
        with self._lock:
            meta_list = self._key2meta.pop(key, [])
            self._key2meta[key] = meta_list   # make sure the ordering of the index is the same as the data in the file
            s = self._encoder(value)
            assert isinstance(s, bytes), "stream must be bytes"
            meta_list.append((self._data_file.tell(), len(s)))
            self._modified = True
            self._df_seek_end()
            self._data_file.write(s)

    def keys(self):
        return self._key2meta.keys()

    def __len__(self):
        return len(self._key2meta)

    def __iter__(self):
        return iter(self._key2meta)

    def __contains__(self, key):
        return key in self._key2meta

    def items(self):
        for key in iter(self):
            yield key, self[key]

    def values(self):
        for key in iter(self):
            yield key, self[key]

    def save_as(self, record_path: str):
        if record_path == self._record_path:
            return self.refresh_data_file()

        with self._lock:
            ir = IndexedRecord(record_path, write=True)
            items = self.items()
            if self._verbose:
                from tqdm import tqdm
                items = tqdm(items, total=len(self))
            for k, v in items:
                ir[k] = v

    def has_dirty_entry(self):
        with self._lock:
            return all(len(v) == 1 for v in self._key2meta.values())

    def refresh_data_file(self):
        base_tmp_record_path = self._record_path + '/staged'
        tmp_record_path = base_tmp_record_path
        while os.path.exists(tmp_record_path):
            tmp_record_path = base_tmp_record_path + "." + str(random.randint(10000))
        self.save_as(tmp_record_path)
        for fn in os.listdir(tmp_record_path):
            shutil.move(os.path.join(tmp_record_path, fn), os.path.join(self._record_path, fn))
        os.rmdir(tmp_record_path)


class FolderToIndexedRecord:
    def __init__(
            self, src_folder: str, record_path: str, remove_files: bool = False
    ):
        self._src_folder = src_folder
        self._record_path = record_path
        self._remove_files = remove_files
        self._ir = IndexedRecord(self._record_path, write=True, verbose=True)

        self._sync_lock = Lock()
        self._sync_context_lock = Lock()
        self._in_sync_context = False
        self._sync_loop_thread = None

    def sync(self, overwrite_existing: bool = False, show_progress_bar: bool = False):
        with self._sync_lock:
            all_filenames = list(fn for fn in os.listdir(self._src_folder) if not fn.startswith('.'))
            locked_filenames = set()
            all_valid_filenames = set()
            for fn in all_filenames:
                if fn.endswith('.lock'):
                    locked_filenames.add(fn[:-len('.lock')])
                else:
                    all_valid_filenames.add(fn)

            all_filenames = all_valid_filenames - locked_filenames
            if not overwrite_existing:
                all_filenames = set(fn for fn in all_filenames if fn not in self._ir)

            if not all_filenames:
                return 0

            if show_progress_bar:
                from tqdm import tqdm
                print("[Folder to LMDB] %s -> %s" % (self._src_folder, self._record_path))
                all_filenames_iterable = tqdm(all_filenames)
            else:
                all_filenames_iterable = all_filenames

            for fn in all_filenames_iterable:
                full_fn = os.path.join(self._src_folder, fn)
                with open(full_fn, 'rb') as f:
                    s = f.read()
                self._ir[fn] = s
                if self._remove_files:
                    try:
                        os.remove(full_fn)
                    except (OSError, FileExistsError):
                        pass

            return len(all_filenames)

    def _sync_loop(self):
        num_zero_updates = 0
        while self._in_sync_context:
            if num_zero_updates > 0:
                time.sleep(2 ** min(3, num_zero_updates-1))
            num_files = self.sync(overwrite_existing=False)
            if not num_files:
                num_zero_updates += 1
            else:
                num_zero_updates = 0
        self.sync(overwrite_existing=False)

    def __enter__(self):
        assert not self._in_sync_context, "already in sync loop context"
        with self._sync_context_lock:
            # enter live mode
            self._in_sync_context = True
            self._sync_loop_thread = Thread(target=self._sync_loop)
            self._sync_loop_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # exit live mode
        with self._sync_context_lock:
            self._in_sync_context = False
            self._sync_loop_thread.join()
            self._sync_loop_thread = None
