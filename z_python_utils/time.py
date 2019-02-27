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


def time_stamp_str():
    return datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f')


def timestamp_for_filename():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')


class StopWatch(OrderedDict):

    def __init__(self, prefix=""):
        super().__init__([(0, time.time())])
        self._prefix = prefix
        self._lap_n = 0

    def lap(self, tag=None):
        self._lap_n += 1
        if tag is None:
            tag = self._lap_n
        if tag in self:
            del self[tag]
        self[tag] = time.time()
        return tag

    def lap_and_print(self, tag=None):
        self.print(self.lap(tag))

    def _print(self, tag, t1, t2):
        t0 = self[0]
        print("%s: %g sec (diff: %g)" % (self._prefix + str(tag), t2 - t0, t2 - t1))

    def print(self, tag):
        t2 = self[tag]
        all_tags = list(self.keys())
        ind2 = all_tags.index(tag)
        if ind2 > 0:
            t1 = self[all_tags[ind2-1]]
        else:
            t1 = t2
        self._print(tag, t1, t2)

    def print_all(self):
        t1 = self[0]
        for tag, t2 in self:
            self._print(tag, t1, t2)
            t1 = t2


class IfTimeout:

    def __init__(self, timeout):
        self.start_time = time.time()
        self.ignored_time = 0.
        if timeout is None:
            self.target_time = None
        else:
            self.target_time = self.start_time + timeout
        self.interval = None
        self.paused_at = None

    def is_timeout(self):
        if self.target_time is None:
            return False
        else:
            cur_time = time.time()
            _ignored_time = self.ignored_time
            if self.paused_at is not None:
                _ignored_time += cur_time - self.paused_at
            if cur_time - self.target_time - _ignored_time > 0:
                if self.interval is None:
                    self.interval = cur_time - self.start_time - _ignored_time
                return True
            else:
                return False

    def add_ignored_time(self, time_amount):
        self.ignored_time += time_amount

    def pause(self):
        if self.paused_at is None:
            self.paused_at = time.time()

    def resume(self):
        if self.paused_at is not None:
            self.add_ignored_time(time.time() - self.paused_at)
            self.paused_at = None


class PeriodicRun:

    def __init__(self, interval, func):
        self.interval = interval
        self.func = func
        self.countdown = None
        self.extra_true_conditions = []
        self.reset()

    def add_extra_true_condition(self, extra_true_condition, extra_func=None):
        if extra_true_condition is None:
            def extra_true_condition_default(**kwargs): return False
            extra_true_condition = extra_true_condition_default
        if extra_func is None:
            extra_func = self.func
        self.extra_true_conditions.append((extra_true_condition, extra_func))

    def add_ignored_time(self, time_amount):
        self.countdown.add_ignored_time(time_amount)

    def reset(self):
        self.countdown = IfTimeout(timeout=self.interval)

    def _run(self, *args, **kwargs):
        output = args[0](*args[1:], **kwargs)
        self.reset()
        return True, output

    def _run_if_timeout(self, *args, **kwargs):
        if self.countdown.is_timeout():
            return self._run(*args, **kwargs)
        else:
            return False, None

    def __call__(self, *args, **kwargs):
        return self.run_if_timeout(*args, **kwargs)

    def run_if_timeout(self, *args, **kwargs):
        return self.run_if_timeout_with_prefixfunc(lambda **kwargs: None, *args, **kwargs)

    def run_if_timeout_with_prefixfunc(self, *args, **kwargs):
        for cond, func in reversed(self.extra_true_conditions):   # later added, have higher priority
            if cond():
                args[0]()
                return self._run(func, *args[1:], **kwargs)

        def the_func(*args, **kwargs):
            args[0]()
            return self.func(*args[1:], **kwargs)
        return self._run_if_timeout(the_func, *args, **kwargs)

    def pause(self):
        self.countdown.pause()

    def resume(self):
        self.countdown.resume()
