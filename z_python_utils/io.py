import os
import sys
import stat
import logging
from typing import List, Tuple, Callable, Union
from shutil import rmtree
from functools import partial


def path_full_split(p):
    s = list()
    a = p
    if not a:
        return s
    if a[-1] == "/":
        s.append("")
        while a and a[-1] == "/":
            a = a[:-1]
    while True:
        [a, b] = os.path.split(a)
        if b:
            s.append(b)
        else:
            if a:
                s.append(a)
            else:
                break
    s.reverse()
    return s


def relative_symlink(src, dst, overwrite=False):
    pdir_src = os.path.dirname(src)
    pdir_dst = os.path.dirname(dst)
    src_rel_dst = os.path.relpath(pdir_src, pdir_dst)
    src_rel = os.path.join(src_rel_dst, os.path.basename(src))
    if overwrite and os.path.exists(dst):
        try:
            os.remove(dst)
        except OSError:
            pass
    if os.path.exists(dst):
        raise OSError('destination exists: %s' % dst)
    os.symlink(src_rel, dst)


def make_file_readonly(fn):
    s = os.stat(fn)[stat.ST_MODE]
    os.chmod(fn, s & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))  # make it read only


def mkdir_p(dir_path):
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            pass


def mkpdir_p(fn):
    mkdir_p(os.path.dirname(fn))


class LoggerSet:

    _counter = 0

    @classmethod
    def create_logger(cls, filename: str, level=logging.INFO, fmt='%(levelname)s: %(message)s'):

        _logger = logging.getLogger("logger.%d" % cls._counter)
        _logger.setLevel(level)
        _logger.propagate = False

        formatter = logging.Formatter(fmt)

        if filename is not None and filename:

            mkpdir_p(filename)
            fh = logging.FileHandler(filename)
            fh.setFormatter(formatter)
            _logger.addHandler(fh)

            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            _logger.addHandler(sh)

        cls._counter += 1

        return _logger


create_logger = LoggerSet.create_logger


class TempIndicatorFile:
    def __init__(self, filename: str):
        self._fn = filename

    def __enter__(self):
        mkpdir_p(self._fn)
        with open(self._fn, "w"):
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.remove(self._fn)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass


class RemoveFilesWhenExit:
    def __init__(self, paths: Union[str, List[str], Tuple[str]]):
        if isinstance(paths, str):
            paths = [paths]
        paths = list(paths)
        self.paths = paths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.paths is None:
            return
        for p in self.paths:
            try:
                rmtree(os.path.abspath(p))
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                pass

    def abort(self):
        self.paths = []


def _wrapper_remove_files_when_finished(*args, paths_to_remove_when_finish=None, **kwargs):
    func = args[0]
    the_args = args[1:]
    with RemoveFilesWhenExit(paths_to_remove_when_finish):
        return func(*the_args, **kwargs)


def remove_files_when_finish(func):
    return partial(_wrapper_remove_files_when_finished, func)


def call_if_not_exisit(*args, **kwargs):
    assert len(args) >= 3, "call_with_file_existence(output_path, extra_paths, fn, ...)"
    output_path = args[0]
    assert isinstance(output_path, str), "the first argument (output_path) must be a str"
    extra_paths = args[1]
    fn = args[2]
    assert isinstance(fn, Callable), "the third argument (fn) must be a callable"

    if extra_paths:
        for p in extra_paths:
            if os.path.exists(p):
                return

    if os.path.exists(output_path):
        return

    running_file = output_path + ".RUNNING"

    if os.path.exists(output_path + ".RUNNING"):
        return

    with TempIndicatorFile(running_file):
        return fn(*args[3:], **kwargs)

