import lmdb
import os
from z_python_utils.io import mkdir_p


def folder_to_lmdb(src_folder: str, lmdb_path: str, overwrite_existing: bool = False, show_progress_bar: bool = False):
    ftl = FolderToLMDB(src_folder, lmdb_path)
    ftl.copy_to_lmdb(overwrite_existing=overwrite_existing, show_progress_bar=show_progress_bar)


class FolderToLMDB:
    def __init__(self, src_folder: str, lmdb_path: str, remove_files: bool = False):
        self._src_folder = src_folder
        self._lmdb_path = lmdb_path
        self._remove_files = remove_files
        mkdir_p(lmdb_path)

        self._env = lmdb.open(lmdb_path)
        self._env_begin = self._env.begin(write=True, buffers=False)
        self._txn = self._env_begin.__enter__()

    def __del__(self):
        self._txn.commit()
        self._env_begin.__exit__(None, None, None)

    def copy_to_lmdb(self, overwrite_existing: bool = False, show_progress_bar: bool = False):
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
            existing_keys = set(key for key, _ in self._txn.cursor())
            all_filenames = set(fn for fn in all_filenames if fn not in existing_keys)

        if show_progress_bar:
            from tqdm import tqdm
            print("[Folder to LMDB] %s -> %s" % (self._src_folder, self._lmdb_path))
            all_filenames_iterable = tqdm(all_filenames)
        else:
            all_filenames_iterable = all_filenames

        for fn in all_filenames_iterable:
            full_fn = os.path.join(self._src_folder, fn)
            with open(full_fn, 'rb') as f:
                s = f.read()
            self._txn.put(fn, s)
            if self._remove_files:
                try:
                    os.remove(full_fn)
                except (OSError, FileExistsError):
                    pass

    def __enter__(self):
        # enter live mode
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # exit live mode
        self.copy_to_lmdb()

