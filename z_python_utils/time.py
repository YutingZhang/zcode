import time
from collections import OrderedDict, deque
import datetime
from typing import Sized, List, Union, Callable, Any, Iterable, Optional
import threading


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


def tic_toc_progress(a, interval: float = 1, name: str = None, print_func=print):

    tic_toc_print = PeriodicRun(interval, print_func)

    n = len(a) if isinstance(a, Sized) else None

    for i, x in enumerate(a):
        line = ''
        if name:
            line += name + ": "
        line += '%d' % i
        if n is not None:
            line += '/ %d' % n
        tic_toc_print(line)
        yield x


def print_time_lapse(t):
    print("Time lapsed: ", datetime.timedelta(seconds=t))


class ParallelRunIfTimeout:

    def __init__(
            self, time_points: Union[List[float], float],
            callback_func: Optional[Callable[[float], Any]] = None,
            main_func: Optional[Callable] = None
    ):
        if not time_points:
            time_points = None
        else:
            if not isinstance(time_points, Iterable):
                time_points = [time_points]
            time_points = sorted(time_points)
        self._time_points = time_points
        if self._time_points is None:
            return
        self.__overall_lock = None
        self.__start_end_call_lock = None
        self._timer_locks = deque()
        if callback_func is None:
            callback_func = print_time_lapse
        self._callback_func = callback_func
        self._thread = None
        self._main_func = main_func

    @property
    def _overall_lock(self) -> threading.Lock:
        assert self.__overall_lock is not None, "lock is not initialized"
        return self.__overall_lock

    @property
    def _start_end_call_lock(self) -> threading.Lock:
        if self.__start_end_call_lock is None:
            self.__start_end_call_lock = threading.Lock()
            self.__overall_lock = threading.Lock()
        return self.__start_end_call_lock

    def _timer(self):
        starting_time = datetime.datetime.now()
        for t in self._time_points:
            with self._overall_lock:
                lck = self._timer_locks[0]
            lck: threading.Lock
            timeout_sec = (starting_time + datetime.timedelta(seconds=t) - datetime.datetime.now()).total_seconds()
            if lck.acquire(timeout=timeout_sec):
                with self._overall_lock:
                    if self._thread is not None:
                        self._timer_locks.popleft()
                    return
            with self._overall_lock:
                self._callback_func(t)

    def start(self):
        with self._start_end_call_lock:
            if self._time_points is None:
                return
            with self._overall_lock:
                if self._thread is not None:
                    return
                self._timer_locks.clear()
                for _ in range(len(self._time_points)):
                    lck = threading.Lock()
                    lck.acquire()
                    self._timer_locks.append(lck)
                self._thread = threading.Thread(target=self._timer)
                self._thread.start()

    def end(self):
        with self._start_end_call_lock:
            if self._time_points is None:
                return
            with self._overall_lock:
                if self._thread is None:
                    return
                lck: threading.Lock
                for lck in self._timer_locks:
                    lck.release()
            self._thread.join()
            with self._overall_lock:
                self._timer_locks.clear()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def __call__(self, *args, **kwargs):
        assert self._main_func is not None, "main function is not set"
        with self:
            self._main_func(*args, **kwargs)

