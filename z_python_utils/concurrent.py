from typing import Union, Callable, Optional
from concurrent import futures
from threading import Lock
from collections import deque
from functools import partial
import time
import tempfile
import pickle
import os
from shutil import rmtree


class FileCachedFunctionJob:

    def __init__(self, *args, **kwargs):
        assert len(args) >= 1, 'first argument must be given'
        self._folder = tempfile.mkdtemp(prefix='FileCachedFunctionJob')
        with open(os.path.join(self._folder, 'content.pkl'), 'wb') as f:
            pickle.dump((args, kwargs), f)

    def __call__(self):
        with open(os.path.join(self._folder, 'content.pkl'), 'rb') as f:
            args, kwargs = pickle.load(f)
        rmtree(self._folder)
        args[0](args[1:], kwargs)


class WorkerExecutor:
    def __init__(self, max_workers: int, use_thread_pool=False, pickle_to_file=False):
        self._max_workers = max_workers
        self._use_thread_pool = use_thread_pool
        self._executor = None
        self._results = deque()
        self._lock = Lock()
        self._pickle_to_file = False if use_thread_pool else pickle_to_file

    def join(self, wait_callback: Optional[Callable]=None, timeout: Optional[float]=None, shutdown: bool=False):
        with self._lock:
            need_wait_callback = wait_callback is not None
            infinity = float('inf')
            if timeout is None:
                timeout = infinity
            t0 = time.time()
            t1 = t0
            while self._results and t1 - t0 <= timeout:
                r = self._results.popleft()
                if need_wait_callback:
                    try:
                        r.result(timeout=0.1)
                        t1 = time.time()
                        continue
                    except futures.TimeoutError:
                        wait_callback()
                        need_wait_callback = False

                if timeout == infinity:
                    r.result()
                else:
                    t1 = time.time()
                    remaining_time = timeout - (t1 - t0)
                    if remaining_time > 0:
                        try:
                            r.result(timeout=remaining_time)
                        except futures.TimeoutError:
                            break
                t1 = time.time()

            if shutdown:
                if self._executor is not None:
                    self._executor.shutdown(wait=False)

    def __call__(self,  *args, **kwargs):
        return self.submit(*args, **kwargs)

    def submit(self, *args, **kwargs):
        if self._pickle_to_file:
            return self._submit(FileCachedFunctionJob(*args, **kwargs))
        else:
            return self._submit(*args, **kwargs)

    def _submit(self, *args, **kwargs):
        with self._lock:
            if self._executor is None:
                if self._use_thread_pool:
                    self._executor = futures.ThreadPoolExecutor(max_workers=self._max_workers)
                else:
                    self._executor = futures.ProcessPoolExecutor(max_workers=self._max_workers)
            r = self._executor.submit(args[0], *args[1:], **kwargs)
            self._results.append(r)
        return r

    def join_and_shutdown(self, wait_callback: Union[Callable, None]=None):
        self.join(wait_callback=wait_callback, shutdown=True)

    def __del__(self):
        self.join()
        if self._executor is not None:
            self._executor.shutdown()


class _ImmediateResult:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self.result()


class ImmediateExecutor:
    def __call__(self, *args, **kwargs):
        return _ImmediateResult(args[0](*args[1:], **kwargs))

    def submit(self, *args, **kwargs):
        return self(*args, **kwargs)

    def join(self, *args, **kwargs):
        pass

    def join_and_shutdown(self, *args, **kwargs):
        pass


ProcessWorkerExecutor = partial(WorkerExecutor, use_thread_pool=False)
ThreadWorkerExecutor = partial(WorkerExecutor, use_thread_pool=True)


immediate_executor = ImmediateExecutor()


class ProcessPoolExecutorWithProgressBar:

    def __init__(
            self, num_workers: int=0, num_tasks: Optional[int]=None, title: Optional[str]=None):

        self._pbar = None
        self._num_workers = num_workers
        self._num_tasks = num_tasks
        self._title = title
        self._results = deque()
        if self._num_workers <= 0:
            self._executor = immediate_executor
        else:
            self._executor = futures.ProcessPoolExecutor(max_workers=num_workers)

        if self._need_pbar:
            if self._title:
                print("[%s] " % self._title, end="")
            if self._num_workers > 0:
                print("Submit tasks", end="")
            else:
                print("Run tasks", end="")
            if self._num_tasks:
                print(": ")
                self._create_pbar(total=self._num_tasks)
            else:
                print(" ...")

        self._open_for_submit = True

    @property
    def _need_pbar(self):
        return self._num_tasks is None or self._num_tasks >= 0

    def _create_pbar(self, total: int):
        if not self._need_pbar:
            return
        self._close_pbar()
        from tqdm import tqdm
        self._pbar = tqdm(total=total)

    def _close_pbar(self):
        if not self._need_pbar:
            return
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def _inc_pbar(self):
        if not self._need_pbar:
            return
        if self._pbar is not None:
            self._pbar.update(1)

    def submit(self, *args, **kwargs):
        assert self._open_for_submit, "executor is joined/joining"
        r = self._executor.submit(*args, **kwargs)
        if self._num_workers > 0:
            self._results.append(r)
        self._inc_pbar()
        return r

    def submit_dummy(self):
        self._inc_pbar()

    def join(self):
        self._close_pbar()
        if self._num_workers <= 0:
            return

        if self._need_pbar:
            if self._title:
                print("[%s] " % self._title, end="")
            print("Run tasks: ")

        self._create_pbar(total=len(self._results))
        while self._results:
            r = self._results.popleft()
            r.result()
            self._inc_pbar()
        self._close_pbar()

    def __del__(self):
        self._close_pbar()

