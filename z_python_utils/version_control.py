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
from .time import timestamp_for_filename
from .functions import call_until_success

# get git versioning info
def git_version_dict(dir_path=None):
    if dir_path is None or not dir_path:
        dir_path = os.getcwd()
        prefixed_command = ""
    else:
        dir_path = os.path.abspath(dir_path)
        prefixed_command = "cd '%s'; " % dir_path

    gv_dict = dict()

    gv_dict["DATE_TIME"] = timestamp_for_filename()

    gv_dict["ARGV"] = copy(sys.argv)

    is_git = call_until_success(
        OSError, subprocess.call,
        "%sgit branch" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        executable="bash",
    ) == 0
    if not is_git:
        return gv_dict

    # git checksum of HEAD
    proc = call_until_success(
        OSError, subprocess.Popen,
        "%sgit rev-parse HEAD" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
        executable="bash",
    )
    head_checksum = proc.stdout.read().decode("utf-8")
    proc.communicate()
    proc.wait()
    gv_dict["HEAD_CHECKSUM"] = head_checksum.strip()

    # git diff with HEAD
    proc = call_until_success(
        OSError, subprocess.Popen,
        "%sgit diff" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
        executable="bash",
    )
    diff_with_head = proc.stdout.read().decode("utf-8")
    proc.communicate()
    proc.wait()
    if not diff_with_head.strip():
        diff_with_head = None
    gv_dict["DIFF_WITH_HEAD"] = diff_with_head

    # user files in trackback
    file_cache = []
    for a in traceback.extract_stack():
        fn = a.filename
        if not os.path.exists(fn):
            # not a file
            continue
        if not fn.startswith(dir_path):
            # not a file in the specified dir
            continue

        is_tracked = call_until_success(
            OSError, subprocess.call,
            "%sgit ls-files --error-unmatch '%s'" % (prefixed_command, fn),
            shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            executable="bash",
        ) == 0
        is_different = True

        if is_tracked:
            proc = call_until_success(
                OSError, subprocess.Popen,
                "%sgit diff '%s'" % (prefixed_command, fn),
                shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
                executable="bash",
            )
            diff_for_this_file = proc.stdout.read().decode("utf-8")
            proc.communicate()
            proc.wait()
            diff_for_this_file = diff_for_this_file.strip()
            is_different = bool(diff_for_this_file)

        if is_different:
            # if different in git, save the file content
            with open(fn, 'r') as f:
                c = f.read()
                file_cache.append((fn, c))
        else:
            # if clean in git
            file_cache.append((fn, None))

    gv_dict["TRACEBACK_FILE_CACHE"] = file_cache

    return gv_dict
