__all__ = ['RemoteFile', 'better_smart_open']


import os
from z_python_utils.os import run_system
from tempfile import mkdtemp
from z_python_utils.io import rm_f, better_smart_open
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RemoteFile:
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
        if self.local_dir is None and self.is_remote_path:
            logger.info('Pulling remote file(s): ' + self.path)
            self.local_dir = mkdtemp(prefix='RemoteFile', dir=self.tmp_dir)
            if self.path.startswith('s3://'):
                run_system("aws cp --recursive '{:s}' '{:s}'".format(
                    self.path, os.path.join(self.local_dir, self.filename)
                ))
            else:
                raise ValueError('unsupported remote protocol')
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
