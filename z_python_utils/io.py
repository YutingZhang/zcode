from collections import namedtuple, Iterable
import time
import inspect
from copy import copy
from collections import deque, OrderedDict
import os
import datetime
from inspect import isfunction, ismethod
import sys
import subprocess
import traceback
import io
import stat
from recursive_utils.recursive_utils import *
import logging
import threading
from concurrent import futures
from typing import Type, Tuple, List, Union, Callable


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


def relative_symlink(src, dst):
    pdir_src = os.path.dirname(src)
    pdir_dst = os.path.dirname(dst)
    src_rel_dst = os.path.relpath(pdir_src, pdir_dst)
    src_rel = os.path.join(src_rel_dst, os.path.basename(src))
    # if os.path.exists(dst):
    #     os.remove(dst)
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

