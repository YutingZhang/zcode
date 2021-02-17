from typing import Union, Callable, Optional
from concurrent import futures
from threading import Lock, Thread
from collections import deque
from functools import partial
import time
import tempfile
import pickle
import os
from shutil import rmtree
import random


class FileCachedFunctionJob:

    def __init__(self, *args, **kwargs):
        assert len(args) >= 1, 'first argument must be given'
        self._folder = tempfile.mkdtemp(prefix='FileCachedFunctionJob-')
        with open(os.path.join(self._folder, 'content.pkl'), 'wb') as f:
            pickle.dump((args, kwargs), f)

    def __call__(self):
        with open(os.path.join(self._folder, 'content.pkl'), 'rb') as f:
            args, kwargs = pickle.load(f)
        self._remove_cached()
        args[0](*args[1:], **kwargs)

    def _remove_cached(self):
        if not os.path.exists(self._folder):
            return
        try:
            rmtree(self._folder)
        except FileNotFoundError:
            pass

    def __del__(self):
        self._remove_cached()


class WorkerExecutor:

    def __init__(self, max_workers: int, use_thread_pool=False, pickle_to_file=False):
        self._max_workers = max_workers
        self._use_thread_pool = use_thread_pool
        self._executor = None
        self._results = deque()
        self._lock = Lock()
        self._pickle_to_file = False if use_thread_pool else pickle_to_file

    def join(
            self, wait_callback: Optional[Callable] = None, timeout: Optional[float] = None, shutdown: bool = False,
            raise_timeout_exception: bool = False
    ):
        with self._lock:
            is_timeout = False
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
                            is_timeout = True
                            break
                t1 = time.time()

            if shutdown:
                if self._executor is not None:
                    self._executor.shutdown(wait=False)

            if raise_timeout_exception and is_timeout:
                raise TimeoutError()

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
        return self._result


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
            self, num_workers: int = 0, num_tasks: Optional[int] = None, title: Optional[str] = None,
            use_thread_pool: bool = False, store_results: bool = False
    ):

        self._pbar = None
        self._num_workers = num_workers
        self._num_tasks = num_tasks
        self._title = title
        self._results = deque()
        if self._num_workers <= 0:
            self._executor = immediate_executor
        else:
            if use_thread_pool:
                self._executor = futures.ThreadPoolExecutor(max_workers=num_workers)
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

        self._store_results = store_results
        self._result_vals = []

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
        else:
            if self._store_results:
                self._result_vals.append(r.result())
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
            r_val = r.result()
            if self._store_results:
                self._result_vals.append(r_val)
            del r_val
            self._inc_pbar()
        self._close_pbar()

    def __del__(self):
        self._close_pbar()

    def get_results(self):
        assert self._store_results, "results are not stored"
        return self._result_vals


class DetachableExecutorWrapper:
    """
    the executor wrapped can be deleted before joining the executor
    a guard thread will take care of the joining in the background
    """

    def __init__(self, executor, join_func_name: str = 'join'):
        self._executor = executor
        self._join_func_name = join_func_name

    def __del__(self):
        join_func = getattr(self._executor, self._join_func_name)
        _DetachableExecutorWrapperAux.put_into_trash_queue(join_func)

    def __getattr__(self, item):
        if hasattr(self._executor, item):
            return getattr(self._executor, item)
        raise AttributeError("Attribute does not exist: %s" % item)


class _DetachableExecutorWrapperAux:

    guard_instance = None
    guard_instance_lock = Lock()

    garbage_executor_pool = dict()
    garbage_executor_pool_lock = Lock()
    garbage_executor_collection_lock = Lock()

    active = False
    active_gc_loop_lock = Lock()
    first_cycle_ready = False

    def __init__(self):
        with type(self).active_gc_loop_lock:
            self._thread = Thread(
                target=type(self)._garbage_collection_loop
            )
            self._thread.start()
        while not type(self).first_cycle_ready:
            pass

    @classmethod
    def put_into_trash_queue(cls, executor_join_func):

        with cls.guard_instance_lock:
            if cls.guard_instance is None:
                cls.guard_instance = cls()

        with cls.garbage_executor_pool_lock:
            executor_id = -1
            while executor_id < 0 or executor_id in cls.garbage_executor_pool:
                executor_id = random.randint(0, 2147483647)
            cls.garbage_executor_pool[executor_id] = executor_join_func
        cls._try_to_unlock_gc_loop()

    @classmethod
    def _try_to_unlock_gc_loop(cls):
        with cls.garbage_executor_pool_lock:    # try to make sure not to mess up with pop from pool
            try:
                cls.garbage_executor_collection_lock.release()
            except (KeyboardInterrupt, SystemError):
                raise
            except:
                pass

    @classmethod
    def _garbage_collection_loop(cls):
        with cls.active_gc_loop_lock:
            cls.garbage_executor_collection_lock.acquire()
            cls.first_cycle_ready = True
            cls.garbage_executor_collection_lock.acquire()
            while True:
                with cls.garbage_executor_pool_lock:
                    garbage_ids = list(cls.garbage_executor_pool)
                    if not garbage_ids:
                        break

                for executor_id in garbage_ids:
                    with cls.garbage_executor_pool_lock:
                        executor_join_func = cls.garbage_executor_pool.pop(executor_id)
                    executor_join_func()
                    del executor_join_func
            cls._try_to_unlock_gc_loop()

        cls.first_cycle_ready = False

    def __del__(self):
        type(self)._try_to_unlock_gc_loop()
        self._thread.join()
        type(self)._try_to_unlock_gc_loop()


def async_detechable_thread_call(*args, **kwargs):
    thread = Thread(target=args[0], args=args[1:], kwargs=kwargs)
    thread.start()
    _ = DetachableExecutorWrapper(thread, join_func_name='join')


def _heart_beat(interval: float, callback: Callable, running_lock: Lock, alive_lock: Lock):
    while not alive_lock.acquire(timeout=interval):
        with running_lock:
            if alive_lock.acquire(timeout=0):
                # if acquired, it is dead
                return
            callback()


class HeartBeat:

    _all_threads = dict()
    _all_threads_lock = Lock()

    def __init__(self, interval: float, callback: Callable):
        self._interval = interval   # in sec
        self._callback = callback

    def start(self):
        with type(self)._all_threads_lock:
            if id(self) in type(self)._all_threads:
                return
        running_lock = Lock()
        alive_lock = Lock()
        thread_dict = dict(
            running_lock=running_lock, alive_lock=alive_lock
        )
        thread = Thread(target=_heart_beat, kwargs=dict(thread_dict))
        thread_dict["thread"] = thread
        with type(self)._all_threads_lock:
            if id(self) in type(self)._all_threads:
                return
            type(self)._all_threads[id(self)] = thread_dict
        alive_lock.acquire()
        thread.start()

    def stop(self):
        with type(self)._all_threads_lock:
            if id(self) not in type(self)._all_threads:
                return
            thread_dict = type(self)._all_threads.pop(id(self))
        running_lock = thread_dict["running_lock"]
        alive_lock = thread_dict["alive_lock"]
        thread = thread_dict["thread"]
        del thread_dict
        alive_lock: Lock
        thread: Thread
        with running_lock:
            alive_lock.release()
        thread.join()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        self.stop()


def main():
    import time
    executor = WorkerExecutor(max_workers=1)
    dew = DetachableExecutorWrapper(executor)
    dew.submit(time.sleep, 3)


if __name__ == '__main__':
    main()
