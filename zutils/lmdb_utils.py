import lmdb
import os
from z_python_utils.io import mkdir_p


def folder_to_lmdb(src_folder: str, lmdb_path: str, show_progress_bar: bool = False):
    all_filenames = list(fn for fn in os.listdir(src_folder) if not fn.startswith('.'))
    mkdir_p(lmdb_path)
    env = lmdb.open(lmdb)

    if show_progress_bar:
        from tqdm import tqdm
        print("[Folder to LMDB] %s -> %s" % (src_folder, lmdb_path))
        all_filenames_iterable = tqdm(all_filenames)
    else:
        all_filenames_iterable = all_filenames

    with env.begin(write=True, buffers=False) as txn:
        for fn in all_filenames_iterable:
            with open(os.path.join(src_folder, fn), 'rb') as f:
                s = f.read()
            txn.put(fn, s)
        txn.commit()


class FolderToLMDB:
    def __init__(self, src_folder: str, lmdb_path: str, remove_files: bool = False):
        self._src_folder = src_folder
        self._lmdb_path = lmdb_path
        mkdir_p(lmdb_path)

        self._env = lmdb.open(lmdb_path)
        self._env_begin = self._env.begin(write=True, buffers=False)
        self._txn = self._env_begin.__enter__()

    def __del__(self):
        self._txn.commit()
        self._env_begin.__exit__(None, None, None)

    def copy_to_lmdb(self, show_progress_bar: bool = False):
        all_filenames = list(fn for fn in os.listdir(self._src_folder) if not fn.startswith('.'))
        if show_progress_bar:
            from tqdm import tqdm
            print("[Folder to LMDB] %s -> %s" % (self._src_folder, self._lmdb_path))
            all_filenames_iterable = tqdm(all_filenames)
        else:
            all_filenames_iterable = all_filenames

        for fn in all_filenames_iterable:
            with open(os.path.join(self._src_folder, fn), 'rb') as f:
                s = f.read()
            self._txn.put(fn, s)
