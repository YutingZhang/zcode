from typing import Union, Callable
from concurrent import futures
from threading import Lock
from collections import deque
from functools import partial


class WorkerExecutor:
    def __init__(self, max_workers: int, use_thread_pool=False):
        self._max_workers = max_workers
        self._use_thread_pool = use_thread_pool
        self._executor = None
        self._results = deque()
        self._lock = Lock()

    def join(self, wait_callback: Union[Callable, None]=None, shutdown=False):
        with self._lock:
            need_wait_callback = wait_callback is not None
            while self._results:
                r = self._results.popleft()
                if need_wait_callback:
                    try:
                        r.result(timeout=0.1)
                    except futures.TimeoutError:
                        wait_callback()
                        need_wait_callback = False
                        r.result()
                else:
                    r.result()
            if shutdown:
                if self._executor is not None:
                    self._executor.shutdown()

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


ProcessWorkerExecutor = partial(WorkerExecutor, use_thread_pool=False)
ThreadWorkerExecutor = partial(WorkerExecutor, use_thread_pool=True)


immediate_executor = ImmediateExecutor()
