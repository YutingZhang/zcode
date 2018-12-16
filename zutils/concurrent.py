from typing import Union, Callable, Optional
from concurrent import futures
from threading import Lock
from collections import deque
from functools import partial
import time


class WorkerExecutor:
    def __init__(self, max_workers: int, use_thread_pool=False):
        self._max_workers = max_workers
        self._use_thread_pool = use_thread_pool
        self._executor = None
        self._results = deque()
        self._lock = Lock()

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

    def join(self, *args, **kwargs):
        pass

    def join_and_shutdown(self, *args, **kwargs):
        pass


ProcessWorkerExecutor = partial(WorkerExecutor, use_thread_pool=False)
ThreadWorkerExecutor = partial(WorkerExecutor, use_thread_pool=True)


immediate_executor = ImmediateExecutor()
