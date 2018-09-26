from collections import namedtuple, OrderedDict, Iterable
import time
import inspect
from copy import copy
from collections import deque
import os
import datetime
from inspect import isfunction, ismethod
import sys
import subprocess
import traceback
import io
import stat
from zutils.recursive_utils import *
import logging


def time_stamp_str():
    return datetime.datetime.now().strftime('%Y-%m/%d-%H:%M:%S.%f')


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


def convert2set(a):
    if isinstance(a, set):
        return a
    if a is None:
        return set()
    if isinstance(a, list) or isinstance(a, tuple):
        return set(a)
    else:
        return {a}


def dict2namedtuple(d, tuple_name=None):
    if tuple_name is None:
        tuple_name = "lambda_namedtuple"
    return namedtuple(tuple_name, d.keys())(**d)


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


class FlagEater:

    def __init__(self, flags=None, default_value=False, default_pos_value=True):
        if isinstance(flags, FlagEater):
            self._flags = flags._flags
            self._default_value = flags._default_value
            self._default_pos_value = flags._default_pos_value
        else:
            self._default_value = default_value
            self._default_pos_value = default_pos_value
            self._flags = dict()
            self.add_flags(flags)

    def add_flags(self, flags):
        if flags is None:
            return
        if isinstance(flags, dict):
            self._flags = {**self._flags, **flags}
        elif isinstance(flags, (list, tuple, set)):
            for k in flags:
                self._flags[k] = self._default_pos_value
        else:
            self._flags[flags] = self._default_pos_value

    def pop(self, key):
        return self._flags.pop(key, self._default_value)

    def finalize(self):
        assert not self._flags, "Not all flags are eaten"


def call_func_with_ignored_args(func, *args, **kwargs):

    remaining_args = deque(args)
    remaining_kwargs = copy(kwargs)
    actual_args = list()
    actual_kwargs = OrderedDict()
    for k, v in inspect.signature(func).parameters.items():
        value_has_been_set = False
        if v.kind == v.POSITIONAL_ONLY:
            if remaining_args:
                actual_args.append(remaining_args.popleft())
                value_has_been_set = True
        elif v.kind == v.POSITIONAL_OR_KEYWORD:
            if remaining_args:
                actual_args.append(remaining_args.popleft())
                value_has_been_set = True
            elif remaining_kwargs and k in remaining_kwargs:
                actual_kwargs[k] = remaining_kwargs.pop(k)
                value_has_been_set = True
        elif v.kind == v.VAR_POSITIONAL:
            actual_args.extend(remaining_args)
            value_has_been_set = True
        elif v.kind == v.KEYWORD_ONLY:
            if remaining_kwargs and k in remaining_kwargs:
                actual_kwargs[k] = remaining_kwargs.pop(k)
                value_has_been_set = True
        elif v.kind == v.VAR_KEYWORD:
            actual_kwargs = {**actual_kwargs, **remaining_kwargs}
            value_has_been_set = True
        else:
            raise ValueError("Internal errors: unrecognized parameter kind")

        if not value_has_been_set:
            assert v.default != v.empty, "necessary argument is missing: %s" % v.name

    return func(*actual_args, **actual_kwargs)


def first_in_dict(d):
    for k in d:
        return d[k]


def robust_index(a, i):
    if a is None:
        return None
    else:
        return a[i]


def rbool(a):
    try:
        c = bool(a)
    except OSError:
        raise
    except:
        c = True
    return c


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


def even_partition(total_num, partition_num):
    smaller_size = total_num//partition_num
    larger_num = total_num - smaller_size * partition_num
    smaller_num = partition_num - larger_num
    p = [smaller_size+1] * larger_num + [smaller_size] * smaller_num
    return p


def even_partition_indexes(total_num, partition_num):
    subset_num = even_partition(total_num, partition_num)
    epi = list()
    for i in range(len(subset_num)):
        epi.extend([i] * subset_num[i])
    return epi


def value_class_for_with(init_value=None):

    class value_for_with:

        current_value = init_value
        value_stack = [init_value]

        def __init__(self, value):
            self._value = value

        def __enter__(self):
            self.value_stack.append(self._value)
            value_for_with.current_value = self._value
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.value_stack.pop()
            value_for_with.current_value = self.value_stack[-1]
            return False

    return value_for_with


class dummy_class_for_with:

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def get_new_members(inherent_class, base_class):
    assert issubclass(inherent_class, base_class), "must be inherent class and base class"
    target_mem = dict(inspect.getmembers(inherent_class))
    base_mem = dict(inspect.getmembers(base_class))
    new_mem = list(filter(lambda a: a[0] not in base_mem or base_mem[a[0]] is not a[1], target_mem.items()))
    return new_mem


def update_class_def_per_ref(target_class, ref_class, target_base_class=None, ref_base_class=None):
    # this is for merge two class definition branch into one
    if target_base_class is None:
        target_base_class = object

    reserved_mem_names = set(dict(get_new_members(target_class, target_base_class)).keys())

    if ref_base_class is None:
        ref_base_class = object

    new_mem = dict(get_new_members(ref_class, ref_base_class))
    override_mem_names = set(new_mem.keys()) - set(reserved_mem_names)

    for k in override_mem_names:
        setattr(target_class, k, new_mem[k])


def link_with_instance(self, another):
    # this is for merges the defintion of another instance

    my_attr_dict = list(filter(
        lambda kk: not (kk.startswith('__') and kk.endswith('__')),
        dir(self)))

    for k in dir(another):
        if k.startswith('__') and k.endswith('__'):
            continue
        if k in my_attr_dict:
            continue
        v = getattr(another, k)
        if not (isfunction(v) or ismethod(v) or callable(v)):
            continue
        setattr(
            self, k, (lambda vv: lambda *arg, **kwargs: vv(*arg, **kwargs))(v))


class ArgsSepFunc:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.my_args = args
        self.my_kwargs = kwargs

    def set_args(self, *args, **kwargs):
        self.my_args = args
        self.my_kwargs = kwargs

    def __call__(self):
        return self.func(*self.my_args, **self.my_kwargs)

# --------------------------------


class ClsWithCustomInit:

    init_scope = value_class_for_with()

    def __init__(self):
        self.args = self._args_dummy
        if getattr(type(self), "init") is not ClsWithCustomInit.init:
            custom_init = getattr(self, 'init')
            custom_init_need_no_args = not inspect.signature(custom_init).parameters
            if custom_init_need_no_args:
                self.__call_custom_init_func(custom_init)
            else:
                self._custom_init = custom_init
                self.args = self._args

    def __call_custom_init_func(self, func, *args, **kwargs):
        with self.init_scope(self):
            return self._call_custom_init_func(func, *args, **kwargs)

    def _call_custom_init_func(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _args(self, *args, **kwargs):
        delattr(self, "args")
        self.__call_custom_init_func(self._custom_init, *args, **kwargs)
        delattr(self, "_custom_init")
        return self

    def _args_dummy(self):
        delattr(self, "args")
        return self

    def init(self, *args, **kwargs):
        pass

    @property
    def _is_custom_initialized(self):
        return not hasattr(self, "_custom_init")

    def _assert_custom_initialized(self):
        assert self._is_custom_initialized, "need to use args() to specify extra arguments"

    @property
    def _is_in_custom_init(self):
        return self in self.init_scope.value_stack

# --------------------------------------------------------------------


class IntervalSearch:
    def __init__(self, split_points, leftmost_val=0):
        self._sp = sorted(split_points)
        self._leftmost_val = leftmost_val

    def __getitem__(self, item):
        a = self._sp

        if not a:
            return 0

        n = len(a)

        left = 0
        right = n + 1
        left_val = self._leftmost_val

        mid = (left+right) // 2
        while mid != left:
            b = a[mid-1]
            if b <= item:
                left = mid
                left_val = b
            else:
                right = mid
            mid = (left + right) // 2

        return mid, (item-left_val)  # (interval_id, loc in interval)

    def __len__(self):
        return len(self._sp)

    @property
    def splitting_points(self):
        return self._sp

    @property
    def leftmost_val(self):
        return self._leftmost_val



# -------------------------------------------------------------------------

def canonicalize_slice(s, end=None):
    return slice(
        0 if s.start is None else s.start,
        end if s.end is None else s.end,
        1 if s.step is None else s.step,
    )

# -------------------------------------------------------------------------

def float_to_rational(a, max_denominator=10):
    if a == int(a):
        return int(a), 1

    if a < 0:
        the_sign = -1
        a = -a
    else:
        the_sign = 1

    d = int(a)
    f = a - d

    min_diff = 1
    the_frac = None
    for i in range(2, max_denominator+1):
        for j in range(1, i):
            cur_diff = abs(f - j/i)
            if min_diff > cur_diff:
                min_diff = cur_diff
                the_frac = (j, i)
    output = (int(the_sign * (the_frac[0] + d * the_frac[1])), int(the_frac[1]))
    return output


# ------------------------------------------------------------------------------

def self_memory_usage():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss
    return mem_usage

# -----------------------------------------------------------------------------
# Based on: https://stackoverflow.com/users/2069807/mrwonderful


def structured_dump(obj, nested_level=0, file=sys.stdout):

    if file is None or file == "str" or file is str:
        str_io = io.StringIO()
        structured_dump(obj, nested_level=nested_level, file=str_io)
        return str_io.getvalue()

    spacing = '   '
    if isinstance(obj, dict):
        print('%s{' % (nested_level * spacing), file=file)
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print('%s%s:' % ((nested_level + 1) * spacing, k), file=file)
                structured_dump(v, nested_level + 1, file=file)
            else:
                print('%s%s: %s' % ((nested_level + 1) * spacing, k, v), file=file)
        print('%s}' % (nested_level * spacing), file=file)
    elif isinstance(obj, list):
        print('%s[' % ((nested_level) * spacing), file=file)
        for v in obj:
            if isinstance(v, Iterable) and not isinstance(v, str):
                structured_dump(v, nested_level + 1, file=file)
            else:
                print('%s%s' % ((nested_level + 1) * spacing, v), file=file)
        print('%s]' % ((nested_level) * spacing), file=file)
    else:
        print('%s%s' % (nested_level * spacing, obj), file=file)


# ------------------------------------------------------------------------------

def match_args_to_callable(callable_or_signature, *args, **kwargs):

    if isinstance(callable_or_signature, inspect.Signature):
        sig = callable_or_signature
    else:
        sig = inspect.signature(callable_or_signature)

    matched_dict = OrderedDict()
    the_args = list(args)
    the_kwargs = OrderedDict(kwargs)
    for k, p in sig.parameters.items():
        assert isinstance(k, str), "internal error: k must be a str"
        assert isinstance(p, inspect.Parameter), "internal error: p must be a inspect.Parameter"
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            matched_dict[k] = list(the_args)
            the_args.clear()
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            for kk in the_kwargs:
                assert kk not in matched_dict, "duplicted key"
            matched_dict[k] = OrderedDict(the_kwargs)
            the_kwargs.clear()
        elif the_args:
            assert p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD), \
                "require a positional parameter"
            matched_dict[k] = the_args[0]
            del the_args[0]
        else:
            if k in the_kwargs:
                matched_dict[k] = the_kwargs[k]
            elif p.default is not inspect.Parameter.empty:
                matched_dict[k] = p.default
            else:
                raise ValueError("required arguments does not exist: %s" % k)

        return matched_dict

# ------------------------------------------------------------------------------


class NonSelfAttrDoesNotExist:
    pass


def get_nonself_attr_for_type(cls: type, name, target_type=None):
    assert hasattr(cls, name), "no such attr"
    a0 = getattr(cls, name)
    mro = inspect.getmro(cls)
    for t in mro:
        if target_type is not None and not issubclass(t, target_type):
            continue
        if issubclass(cls, t):
            continue
        if hasattr(t, name):
            a = getattr(t, name)
            if a is not a0:
                return a
    raise NonSelfAttrDoesNotExist

# ------------------------------------------------------------------------------


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
        os.makedirs(dir_path)


def mkpdir_p(fn):
    mkdir_p(os.path.dirname(fn))


# ------------------------------------------------------------------------------
# get git versioning info


def timestamp_for_filename():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')


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

    is_git = subprocess.call(
        "%sgit branch" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) == 0
    if not is_git:
        return gv_dict

    # git checksum of HEAD
    head_checksum = subprocess.Popen(
        "%sgit rev-parse HEAD" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE
    ).stdout.read().decode("utf-8")
    gv_dict["HEAD_CHECKSUM"] = head_checksum.strip()

    # git diff with HEAD
    diff_with_head = subprocess.Popen(
        "%sgit diff" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE
    ).stdout.read().decode("utf-8")
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

        is_tracked = subprocess.call(
            "%sgit ls-files --error-unmatch '%s'" % (prefixed_command, fn),
            shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        ) == 0
        is_different = True

        if is_tracked:
            diff_for_this_file = subprocess.Popen(
                "%sgit diff '%s'" % (prefixed_command, fn),
                shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE
            ).stdout.read().decode("utf-8")
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


# logger ------------------------------------------------


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

