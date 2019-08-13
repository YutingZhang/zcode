import inspect
from copy import copy
from collections import deque, OrderedDict
import sys
import threading
from concurrent import futures
from typing import Type, Tuple, List, Union, Callable


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


def call_until_success(
        exceptions_to_ignore: Union[Type[BaseException], Tuple[Type[BaseException]]], *args, **kwargs
):
    while True:
        try:
            return args[0](*args[1:], **kwargs)
        except exceptions_to_ignore:
            print("call_until_success: failed try again", file=sys.stderr)


# call with timeout warning ----------------------------

class _CallWithTimeoutCallback:

    thread_executor_pools = deque()
    thread_executor_pools_lock = threading.Lock()

    @classmethod
    def call_with_timeout_callback(cls, timeout, timeout_callback, *args, **kwargs):
        if isinstance(timeout, (list, tuple)):
            timeout_list = list(zip(timeout, timeout_callback))
        else:
            timeout_list = [(timeout, timeout_callback)]

        # borrow/create executor
        with cls.thread_executor_pools_lock:
            if not cls.thread_executor_pools:
                cls.thread_executor_pools.append(futures.ThreadPoolExecutor(max_workers=1))
            te = cls.thread_executor_pools.pop()

        run_lock = threading.Lock()
        run_lock.acquire()
        r = te.submit(cls._call_with_timeout_callback_callback, timeout_list, run_lock)
        fn = args[0]
        out = fn(*args[1:], **kwargs)
        run_lock.release()
        r.result()

        # return executor
        with cls.thread_executor_pools_lock:
            cls.thread_executor_pools.append(te)

        return out

    @staticmethod
    def _call_with_timeout_callback_callback(
            timeout_list: List[Tuple[float, Union[Callable, str]]], run_lock: threading.Lock
    ):
        timeout_list = sorted(timeout_list, key=lambda x: x[0])
        t0 = 0
        for t1, t_callback in timeout_list:
            dt = max(0, t1 - t0)
            if run_lock.acquire(timeout=dt):
                # run finished already
                break
            if isinstance(t_callback, str):
                print(t_callback, file=sys.stderr)
            else:
                t_callback()
            t0 = t1


call_with_timeout_callback = _CallWithTimeoutCallback.call_with_timeout_callback


def do_nothing(*args, **kwargs):
    pass


# insert code blocks to regular functions -----------------------------------------------------------------


class KeyErrorInCodeBlocks(KeyError):
    pass


class CodeBlocks:

    _code_blocks_stack = []

    def __init__(self, _undefined_as_do_nothing=False, **kwargs):
        self._undefined_as_do_nothing = _undefined_as_do_nothing
        for k, v in kwargs.items():
            assert isinstance(k, str) and k, "code block name must be a str"
            assert k[0] != '_', "code block name must not start with '_'"
            setattr(self, k, v)

    def __getitem__(self, item: str):

        assert isinstance(item, str) and item, "item must be a non-empty string"
        assert item[0] != '_', "item should not start with '_'. no support on calling private function"

        if hasattr(self, item):
            code_block = getattr(self, item)
        else:
            if self._undefined_as_do_nothing:
                return
            else:
                raise KeyErrorInCodeBlocks("No such code blocks: %s" % item)

        frame = inspect.currentframe()
        try:
            caller_locals = frame.f_back.f_locals
        finally:
            del frame

        non_private_caller_locals = dict(filter(lambda _k, _: _k[0] != '_', caller_locals.items()))
        output_variable_dict = call_func_with_ignored_args(
            code_block, non_private_caller_locals
        )
        if output_variable_dict is None:
            return

        for k, v in output_variable_dict.items():
            assert isinstance(k, str) and k, "output variable name must be a str"
            assert k[0] != '_', "should not try to modify the private variables in the caller"
            caller_locals[k] = v

    def __enter__(self):
        type(self)._code_blocks_stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        type(self)._code_blocks_stack.pop()

    @classmethod
    def _call_in_context_stack(cls, item: str):
        for code_blocks in cls._code_blocks_stack[::-1]:
            try:
                return code_blocks[item]
            except KeyErrorInCodeBlocks:
                pass


insert_code_block = CodeBlocks._call_in_context_stack

