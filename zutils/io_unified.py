__all__ = [
    'RemoteOrLocalFile', 'better_smart_open',
    'local_or_zip_as_folder',
    'FolderOrZipReader', 'FolderOrZipReaderFolder',
]


import os
from z_python_utils.os import run_system
from tempfile import mkdtemp
from z_python_utils.io import rm_f, better_smart_open
import re
from typing import Optional, List
import logging
import zipfile
import codecs
import fnmatch
import glob
from pathlib import Path

logger = logging.getLogger(__name__)


class RemoteOrLocalFile:
    def __init__(self, path: str, tmp_dir: str = '/tmp'):
        self.path = path
        self.tmp_dir = tmp_dir
        self.is_remote_path = re.match(r'^[a-zA-Z0-9]+://', path)
        self.local_dir: Optional[str] = None
        self.filename = os.path.basename(self.path)
        self.count = 0

    @property
    def _local_path(self) -> str:
        return os.path.join(self.local_dir, self.filename)

    def local(self) -> str:
        if self.local_dir is None:
            self.count = 0
        self.count += 1
        if self.local_dir is None:
            if self.is_remote_path:
                logger.info('Pulling remote file(s): ' + self.path)
                self.local_dir = mkdtemp(prefix='RemoteOrLocalFile', dir=self.tmp_dir)
                if self.path.startswith('s3://'):
                    run_system("aws cp --recursive '{:s}' '{:s}'".format(
                        self.path, os.path.join(self.local_dir, self.filename)
                    ))
                else:
                    raise ValueError('unsupported remote protocol')
            else:
                self.local_dir = self.path
        return self._local_path

    def release_once(self):
        self.count -= 1
        self.count = max(self.count, 0)
        self.clear_local_cache_if_needed()

    def release(self):
        self.count = 0
        self.clear_local_cache_if_needed()

    def clear_local_cache_if_needed(self):
        if self.is_remote_path and self.count <= 0 and self.local_dir is not None:
            rm_f(self.local_dir)
            self.local_dir = None

    def __del__(self):
        self.release()

    def __enter__(self):
        return self.local()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_once()


class FolderOrZipReader:
    def __init__(self, path: str):
        self.path = path
        self.zf = None
        self.zf_all_fn = None
        if path.endswith('.zip'):
            self.zf = zipfile.ZipFile(path, 'r')
            self.zf_all_fn = self.zf.namelist()
            self.path_type = 'zip'
        else:
            self.path_type = 'folder'

        self.root_folder = FolderOrZipReaderFolder(self)
        self.chdir = self.root_folder.chdir
        self.listdir = self.root_folder.listdir

    def open(self, filename: str, mode: str = 'r', encoding: Optional[str] = 'utf-8'):
        filename = os.path.abspath(os.path.join('/', filename))[1:]
        if mode == 'b':
            mode = 'rb'
        elif not mode:
            mode = 'r'
        assert mode in {'r', 'rb'}, 'must be a read mode'
        if self.path_type == 'folder':
            return open(os.path.join(self.path, filename), mode=mode, encoding=encoding)
        elif self.path_type == 'zip':
            f_binary = self.zf.open(filename, mode='r')
            if mode == 'r':
                f = codecs.iterdecode(f_binary, 'utf-8')
            else:
                f = f_binary
            return f
        else:
            raise ValueError('Unsupported path type')


class FolderOrZipReaderFolder:
    def __init__(self, fozr: FolderOrZipReader, subfolder: str = ''):
        self._fozr = fozr
        self._subfolder = os.path.abspath(os.path.join('/', subfolder))[1:]

    def chdir(self, subpath: str):
        return type(self)(self._fozr, os.path.join(self._subfolder, subpath))

    @property
    def path(self) -> str:
        return os.path.join(self._fozr.path, self._subfolder)

    @property
    def path_type(self) -> str:
        return self._fozr.path_type

    def _all_fn_for_zip(self):
        if not self._subfolder:
            fns = self._fozr.zf_all_fn
        else:
            _fn_prefix = self._subfolder + '/'
            fns = [_x[len(_fn_prefix):] for _x in self._fozr.zf_all_fn if _x.startswith(_fn_prefix)]
        return fns

    def listdir(self) -> List[str]:
        if self._fozr.path_type == 'folder':
            subfolder_full_path = os.path.join(self._fozr.path, self._subfolder)
            files_in_this_folder = [
                ((_x + '/') if os.path.isdir(os.path.join(subfolder_full_path, _x)) else _x)
                for _x in os.listdir(subfolder_full_path)
            ]
        elif self._fozr.path_type == 'zip':
            candidate_fns = self._all_fn_for_zip()
            files_in_this_folder = []
            for _x in candidate_fns:
                _y = _x.split('/')
                if len(_y) == 1 or (len(_y) == 2 and not _y[1]):
                    files_in_this_folder.append(_x)
        else:
            raise ValueError('Unsupported path type')

        return files_in_this_folder

    def glob(self, pathname: str):
        if self._fozr.path_type == 'folder':
            return glob.glob(pathname, root_dir=self.path)
        elif self._fozr.path_type == 'zip':
            return fnmatch.filter(self._all_fn_for_zip(), pathname)
        else:
            raise ValueError('Unsupported path type')

    def open(self, filename: str):
        return self._fozr.open(os.path.join(self._subfolder, filename))


def local_or_zip_as_folder(path: str) -> FolderOrZipReaderFolder:
    return FolderOrZipReader(path).chdir('')