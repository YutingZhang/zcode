import sys
from typing import Union, Callable, Optional, Type
from concurrent import futures
import threading
from collections import deque
from functools import partial, lru_cache
import time
import tempfile
import pickle
import os
from shutil import rmtree
import random
import multiprocessing.managers as mpm
from z_python_utils.classes import ObjectPool
from z_python_utils.functions import run_and_print_trackback_if_exception


# FIXME: this is a money patch
# See: https://www.py4u.net/discuss/151784

def mpm_AutoProxy(token, serializer, manager=None, authkey=None,
              exposed=None, incref=True, manager_owned=False):
    '''
    Return an auto-proxy for `token`
    '''
    _Client = mpm.listener_client[serializer][1]

    if exposed is None:
        conn = _Client(token.address, authkey=authkey)
        try:
            exposed = mpm.dispatch(conn, None, 'get_methods', (token,))
        finally:
            conn.close()

    if authkey is None and manager is not None:
        authkey = manager._authkey
    if authkey is None:
        authkey = mpm.process.current_process().authkey

    ProxyType = mpm.MakeProxyType('AutoProxy[%s]' % token.typeid, exposed)
    proxy = ProxyType(token, serializer, manager=manager, authkey=authkey,
                      incref=incref, manager_owned=manager_owned)
    proxy._isauto = True
    return proxy


mpm.AutoProxy = mpm_AutoProxy


# -----------------------------

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
        self._lock = threading.Lock()
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

    def join_and_shutdown(self, wait_callback: Union[Callable, None] = None):
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
    def submit(self, *args, **kwargs):
        return _ImmediateResult(args[0](*args[1:], **kwargs))

    def __call__(self, *args, **kwargs):
        return self.submit(*args, **kwargs)

    def join(self, *args, **kwargs):
        pass

    def join_and_shutdown(self, *args, **kwargs):
        pass

    def shutdown(self, *args, **kwargs):
        pass


ProcessWorkerExecutor = partial(WorkerExecutor, use_thread_pool=False)
ThreadWorkerExecutor = partial(WorkerExecutor, use_thread_pool=True)


immediate_executor = ImmediateExecutor()


class SubmitterThrottle:
    def __init__(
            self,
            executor: Union[futures.ThreadPoolExecutor, futures.ProcessPoolExecutor],
            bandwidth: Union[int, None],
            done_callback: Optional[Callable] = None
    ):
        self._executor = executor
        if bandwidth is None:
            bandwidth = float('inf')
        self._bandwidth = bandwidth
        self._op_lock = threading.RLock()
        self._pending_futures = set()
        self._task_count = 0
        self._throttle_lock = threading.Lock()
        self._done_callback = done_callback
        self._joined_lock = threading.Lock()

    def claim_done(self, future_obj, task_id: int):
        with self._op_lock:
            self._pending_futures.remove(task_id)
            if len(self._pending_futures) == self._bandwidth:
                self._throttle_lock.release()
            if self._done_callback is not None:
                self._done_callback(task_id, future_obj)
            if not self._pending_futures:
                self._joined_lock.release()

    def submit(self, *args, **kwargs):
        self._throttle_lock.acquire()
        r = self._executor.submit(*args, **kwargs)
        with self._op_lock:
            self._pending_futures.add(self._task_count)
            self._joined_lock.acquire(blocking=False)
            if len(self._pending_futures) <= self._bandwidth:
                self._throttle_lock.release()
            task_id = self._task_count
            self._task_count += 1
        # the following cannot be in the op_lock context. because when the task is done fast,
        # the claim done will be called directly in a sync manner
        r.add_done_callback(partial(self.claim_done, task_id=task_id))
        return r

    def join(self):
        with self._joined_lock:
            pass


class ProcessPoolExecutorWithProgressBar:

    def __init__(
            self, num_workers: int = 0, num_tasks: Optional[int] = None, title: Optional[str] = None,
            use_thread_pool: bool = False, store_results: bool = False, throttling_bandwidth: int = 0,
            **kwargs
    ):
        """
        Executor that show a progress bar
        :param num_workers:
        :param num_tasks:
        :param title:
        :param use_thread_pool:
        :param store_results:
        :param throttling_bandwidth: 0 for automatic, None for infinite
        """

        self._pbar = None
        self._num_workers = num_workers
        self._num_tasks = num_tasks
        self._title = title
        self._use_thread_pool = use_thread_pool
        if self._num_workers <= 0:
            self._executor = immediate_executor
        else:
            if use_thread_pool:
                self._executor = futures.ThreadPoolExecutor(max_workers=num_workers, **kwargs)
            else:
                self._executor = futures.ProcessPoolExecutor(max_workers=num_workers, **kwargs)

        if self._need_pbar:
            if self._title:
                print("[%s] " % self._title, end="")
            if self._num_workers > 0:
                print(f"Run tasks ({self._num_workers} {'thread' if use_thread_pool else 'process'} workers)", end="")
            else:
                print("Run tasks (main thread)", end="")
            if self._num_tasks:
                print(": ")
                self._create_pbar(total=self._num_tasks)
            else:
                print(" ...")

        if self._num_workers > 0:
            if throttling_bandwidth is not None:
                if throttling_bandwidth <= 0:
                    throttling_bandwidth = self._num_workers * 2
            self._submitter_throttle = SubmitterThrottle(
                self._executor, throttling_bandwidth, done_callback=self.done_callback
            )
            self._submit_func = self._submitter_throttle.submit
        else:
            self._submitter_throttle = None
            self._submit_func = self._executor.submit

        self._store_results = store_results
        self._result_vals = dict()
        self._task_count = 0

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
        if not hasattr(self, '_need_pbar') and not self._need_pbar:
            return
        if hasattr(self, '_pbar') and self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def _inc_pbar(self):
        if not self._need_pbar:
            return
        if self._pbar is not None:
            self._pbar.update(1)

    def done_callback(self, task_id: int, future_obj):
        if self._store_results:
            self._result_vals[task_id] = future_obj.result()
        self._inc_pbar()

    def submit(self, *args, **kwargs):
        assert self._open_for_submit, "executor is joined/joining"
        r = self._submit_func(*args, **kwargs)
        if self._num_workers <= 0:
            self.done_callback(self._task_count, r)
        self._task_count += 1
        return r

    def submit_dummy(self):
        self._inc_pbar()

    def join(self):
        self._open_for_submit = False
        if hasattr(self, '_submitter_throttle') and self._submitter_throttle is not None:
            self._submitter_throttle.join()
        self._close_pbar()

    def __del__(self):
        self.join()
        self.shutdown()

    def shutdown(self, *args, **kwargs):
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(*args, **kwargs)
            self._executor = None

    def get_results(self):
        assert self._store_results, "results are not stored"
        if self._submitter_throttle is not None:
            self._submitter_throttle.join()
        return [self._result_vals[task_id] for task_id in range(self._task_count)]


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
    guard_instance_lock = threading.Lock()

    garbage_executor_pool = dict()
    garbage_executor_pool_lock = threading.Lock()
    garbage_executor_collection_lock = threading.Lock()

    active = False
    active_gc_loop_lock = threading.Lock()
    first_cycle_ready = False

    def __init__(self):
        with type(self).active_gc_loop_lock:
            self._thread = threading.Thread(
                target=type(self)._garbage_collection_loop
            )
            self._thread.setDaemon(True)
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
    thread = threading.Thread(target=args[0], args=args[1:], kwargs=kwargs)
    thread.start()
    _ = DetachableExecutorWrapper(thread, join_func_name='join')


class CachedExecutorWrapper:
    """
    the executor wrapped can use lru cache for submitted jobs
    """

    def __init__(self, executor, cache_size: int = 1):
        self._executor = executor
        self._cache_size = cache_size
        if self._cache_size > 0:
            self._cached_submit = lru_cache(maxsize=self._cache_size)(executor.submit)
        else:
            self._cached_submit = self._async_run_plain
        self._cached_submit_lock = threading.Lock()

    def __getattr__(self, item):
        if hasattr(self._executor, item):
            return getattr(self._executor, item)
        raise AttributeError("Attribute does not exist: %s" % item)

    def submit(self, *args, **kwargs):
        with self._cached_submit_lock:
            return self._cached_submit(*args, **kwargs)


def _heart_beat(interval: float, callback: Callable, running_lock: threading.Lock, alive_lock: threading.Lock):
    is_first_iter = True
    while True:
        with running_lock:
            if not is_first_iter and alive_lock.acquire(timeout=0):
                # if acquired, it is dead
                return
            callback()
            is_first_iter = False
        if alive_lock.acquire(timeout=interval):
            # if acquired, it is dead
            return


class HeartBeat:

    _all_threads = dict()
    _all_threads_lock = threading.Lock()

    def __init__(
            self, interval: float, callback: Callable,
            final_callback: Optional[Callable] = None,
            stopping_callback: Optional[Callable] = None,
    ):
        self._interval = interval   # in sec
        self._callback = callback
        self._final_callback = final_callback
        self._stopping_callback = stopping_callback

    def start(self):
        with type(self)._all_threads_lock:
            if id(self) in type(self)._all_threads:
                return
        running_lock = threading.Lock()
        alive_lock = threading.Lock()
        thread_dict = dict(
            running_lock=running_lock, alive_lock=alive_lock
        )
        thread = threading.Thread(
            target=_heart_beat, args=(self._interval, self._callback), kwargs=dict(thread_dict),
        )
        thread_dict["thread"] = thread
        with type(self)._all_threads_lock:
            if id(self) in type(self)._all_threads:
                return
            type(self)._all_threads[id(self)] = thread_dict
        alive_lock.acquire()
        thread.setDaemon(True)
        thread.start()

    def stop(self, finalized: bool = True):
        with type(self)._all_threads_lock:
            if id(self) not in type(self)._all_threads:
                return
            thread_dict = type(self)._all_threads.pop(id(self))
        running_lock = thread_dict["running_lock"]
        alive_lock = thread_dict["alive_lock"]
        thread = thread_dict["thread"]
        del thread_dict
        alive_lock: threading.Lock
        thread: threading.Thread
        with running_lock:
            alive_lock.release()
        thread.join()
        if finalized:
            if self._final_callback is not None:
                self._final_callback()
        if self._stopping_callback is not None:
            self._stopping_callback()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        finalized = exc_type is None
        self.stop(finalized=finalized)

    def __del__(self):
        self.stop(finalized=False)


class ExecutorBaseManager(mpm.BaseManager):
    pass


class _FailedToGet:
    pass


class CrossProcessFuture:
    def __init__(self, results_holder, result_id: int):
        self._results_holder: _CrossProcessResultsHolderRemote = results_holder
        self._result_id = result_id
        self._result = None
        self._lock = threading.Lock()

    def result(self):
        try:
            with self._lock:
                result_holder = self._results_holder
            if result_holder is not None:
                d = self._results_holder.pop(self._result_id)
                with self._lock:
                    if isinstance(d, _FailedToGet):
                        err_msg = "internal error: result is not ready and cannot be obtained"
                        print(err_msg, file=sys.stderr)
                        assert result_holder is None, err_msg
                self.set_result(d)
            return self._result
        except (KeyboardInterrupt, SystemError):
            raise
        except:
            import traceback
            print("CrossProcessFuture Error: error while getting results", file=sys.stderr)
            traceback.print_exc()
            raise

    def set_result(self, r):
        with self._lock:
            self._result = r
            self._results_holder = None

    def __del__(self):
        with self._lock:
            if self._results_holder is not None:
                self._results_holder.remove(self._result_id)


class _CrossProcessResultsHolder:
    def __init__(self):
        self._results_lock = threading.Lock()
        self._results = dict()
        self._available_result_index = set()

    def add(self, r) -> int:
        with self._results_lock:
            if self._available_result_index:
                result_id = self._available_result_index.pop()
            else:
                result_id = len(self._results)
            self._results[result_id] = [r, None]
        return result_id

    def set_future(self, result_id: int, future_obj: CrossProcessFuture):
        with self._results_lock:
            if result_id not in self._results:
                return      # remark: this should not happen
            self._results[result_id][1] = future_obj

    def get(self, result_id: int):
        with self._results_lock:
            if result_id not in self._results:
                return _FailedToGet()
            return self._results[result_id][0].result()

    def remove(self, result_id: int):
        with self._results_lock:
            if result_id not in self._results:
                return
            self._available_result_index.add(result_id)
            self._results.pop(result_id)

    def pop(self, result_id: int):
        with self._results_lock:
            if result_id not in self._results:
                return _FailedToGet()
            self._available_result_index.add(result_id)
            r = self._results.pop(result_id)
            d = r[0].result()
            return d

    def flush_all_results(self):
        with self._results_lock:
            future_obj: Union[None, CrossProcessFuture]
            for r, future_obj in self._results.values():
                if future_obj is None:
                    continue
                future_obj.set_result(r.result())
            self._results.clear()


class _CrossProcessResultsHolderRemote:
    def __init__(self, results_holder_id):
        self._results_holder_id = results_holder_id

    @property
    def results_holder(self) -> _CrossProcessResultsHolder:
        return CrossProcessPoolExecutor.result_holder_pool.get(self._results_holder_id)

    def remove(self, result_id: int):
        return self.results_holder.remove(result_id)

    def pop(self, result_id: int):
        return self.results_holder.pop(result_id)


class CrossProcessPoolExecutor:

    result_holder_pool = ObjectPool()

    def __init__(self, executor_type: Union[Type, Callable, str], *args, **kwargs):
        if isinstance(executor_type, str):
            if executor_type == "ProcessPoolExecutor":
                executor_type = futures.ProcessPoolExecutor
            elif executor_type == "ThreadPoolExecutor":
                executor_type = futures.ThreadPoolExecutor
            else:
                executor_type = globals()[executor_type]
        self._executor = executor_type(*args, **kwargs)
        self._results_holder_id = self.result_holder_pool.add(_CrossProcessResultsHolder())
        self.__results_holder_remote: Optional[_CrossProcessResultsHolderRemote] = None
        self._lock = threading.Lock()
        self._result_manager = ExecutorManager()
        # self._result_manager.start()

    @property
    def _results_holder(self) -> _CrossProcessResultsHolder:
        return self.result_holder_pool.get(self._results_holder_id)

    def get_results_holder_id(self):
        return self._results_holder_id

    def set_results_holder_remote(self, results_holder_remote: _CrossProcessResultsHolderRemote):
        self.__results_holder_remote = results_holder_remote

    @property
    def _results_holder_remote(self) -> _CrossProcessResultsHolderRemote:
        assert self.__results_holder_remote is not None, "result_holder_remote has not been set up"
        return self.__results_holder_remote

    def submit(self, *args, **kwargs):
        with self._lock:
            bound_func = partial(run_and_print_trackback_if_exception, *args, **kwargs)
            r = self._executor.submit(bound_func)
            result_id = self._results_holder.add(r)   # TODO: interact with the real result holder
            rf = self._result_manager.ExecutorResultFuture(self._results_holder_remote, result_id)
            self._results_holder.set_future(result_id, rf)
        return rf

    def shutdown(self, wait: bool = True):
        with self._lock:
            self._executor.shutdown(wait=wait)
            if wait:
                self._results_holder.flush_all_results()

    def __del__(self):
        self.shutdown()
        # self._result_manager.shutdown()
        self.result_holder_pool.pop(self._results_holder_id)


ExecutorBaseManager.register(
    "Executor", CrossProcessPoolExecutor,
    exposed=['submit', 'get_results_holder_id', 'set_results_holder_remote', 'shutdown']
)
ExecutorBaseManager.register(
    "ExecutorResultFuture", CrossProcessFuture,
    exposed=['result']
)
ExecutorBaseManager.register(
    "ExecutorResultsHolderRemote", _CrossProcessResultsHolderRemote,
    exposed=['remove', 'pop']
)


class ExecutorManager(ExecutorBaseManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start()

    def __del__(self):
        if hasattr(ExecutorBaseManager, "__del__"):
            getattr(ExecutorBaseManager, "__del__")(self)


class ManagedCrossProcessPoolExecutor:

    manager_pool = ObjectPool()

    def __init__(self, executor_type: Union[Type, Callable, str], *args, **kwargs):
        # self._pid = os.getpid()
        self._manager_id = self.manager_pool.add(ExecutorManager())
        self._executor: CrossProcessPoolExecutor = self.manager.Executor(executor_type, *args, **kwargs)
        results_holder_id = self.executor.get_results_holder_id()
        results_holder_remote = self.manager.ExecutorResultsHolderRemote(results_holder_id)
        self.executor.set_results_holder_remote(results_holder_remote)
        self.submit = self._executor.submit
        self.shutdown = self._executor.shutdown

    @property
    def manager(self) -> ExecutorManager:
        return self.manager_pool.get(self._manager_id)

    @property
    def executor(self) -> CrossProcessPoolExecutor:
        return self._executor

    def __del__(self):
        self.submit = None
        self.shutdown = None
        self._executor = None
        if hasattr(self, "_manager_id"):
            try:
                self.manager.shutdown()
            except (KeyboardInterrupt, SystemError):
                raise
            except:
                pass
            self.manager_pool.pop(self._manager_id, None)


MCPPoolExecutor = ManagedCrossProcessPoolExecutor
MCPProcessPoolExecutor = partial(ManagedCrossProcessPoolExecutor, "ProcessPoolExecutor")
MCPThreadPoolExecutor = partial(ManagedCrossProcessPoolExecutor, "ThreadPoolExecutor")


def main():
    import time
    executor = WorkerExecutor(max_workers=1)
    dew = DetachableExecutorWrapper(executor)
    dew.submit(time.sleep, 3)


if __name__ == '__main__':
    main()
