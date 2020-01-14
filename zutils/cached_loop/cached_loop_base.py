import typing


__all__ = ['CachedLoopBase', 'iter_to_kwargs_for_cached_loop']


def _default_submit(*args, **kwargs):
    return args[0](*args[1:], **kwargs)


def _default_get(obj):
    return obj


class CachedLoopBase:
    def __init__(
            self, function: typing.Union[typing.Callable, typing.List[typing.Callable], typing.Tuple[typing.Callable]],
            indexes: typing.Iterable, cache_size: typing.Optional[int] = None,
            batched: bool = False,
            submit_func=_default_submit, get_func=_default_get
    ):
        if isinstance(function, (tuple, list)):
            # a list of functions, call it in an interleaving manner for each iteration
            assert function, "function must not be an empty list/tuple"
            for i, func in enumerate(function):
                assert callable(func), "the %d function is not callable"
        elif callable(function):
            function = (function,)
        else:
            raise AssertionError("function must be a Callable or a list/tuple of Callable")
        self.functions = tuple(function)

        assert isinstance(indexes, typing.Iterable), "indexes must be an iterable"
        self.indexes = indexes

        if cache_size is None:
            cache_size = len(function)
        else:
            assert isinstance(cache_size, int), "cache_size must be an integer. when <1 means no cache"
            cache_size = max(cache_size, 0)
        self.cache_size = cache_size

        self.batched = bool(batched)

        self.submit_func = submit_func
        self.get_func = get_func

        if isinstance(self.indexes, typing.Sized):
            self.__len__ = lambda: len(self.indexes)

    def iter(self):
        return self.__iter__()

    def __iter__(self):
        return CachedLoopIter(self)


class CachedLoopIter:
    def __init__(self, bottleneck_loop: CachedLoopBase):
        self.meta = bottleneck_loop
        self.cache = dict()
        self.index_iter = iter(self.meta.indexes)
        self.cache_idx = 0
        self.next_idx = 0
        self.finished = False
        self.current_function_id = 0
        self.num_functions = len(self.meta.functions)
        self._fill_cache()

    def _fill_cache(self):
        while not self.finished and len(self.cache) < self.meta.cache_size:
            self._cache_next()

    def _cache_next(self):
        if self.finished:
            return
        try:
            batch_indexes = next(self.index_iter)
        except StopIteration:
            self.finished = True
            return

        func = self.meta.functions[self.current_function_id]
        self.current_function_id = (self.current_function_id + 1) % self.num_functions

        if self.meta.batched:
            r = []
            for i in batch_indexes:
                r.append(self.meta.submit_func(func, i))
        else:
            r = self.meta.submit_func(func, batch_indexes)

        self.cache[self.cache_idx] = r
        self.cache_idx += 1

    def __next__(self):
        if self.next_idx not in self.cache:
            self._cache_next()
            if self.next_idx not in self.cache and self.finished:
                raise StopIteration
        r = self.cache.pop(self.next_idx)
        self._fill_cache()
        self.next_idx += 1
        if self.meta.batched:
            a = []
            for r_k in r:
                a.append(self.meta.get_func(r_k))
            a = tuple(a)
        else:
            a = self.meta.get_func(r)
        return a


class _Iter2DataFunc:
    def __init__(self, iterable: typing.Iterable, size: typing.Optional[int] = None):
        self._iter = iter(iterable)
        self._current = 0
        self._size = size
        self._stopped = False
        self._next_val = None
        self._update_next()

    def _update_next(self):
        if self._stopped:
            raise AssertionError('the iter is used up')
        try:
            self._next_val = next(self._iter)
        except StopIteration:
            self._stopped = True
            self._next_val = None

    def __call__(self, item):
        assert item == self._current, "this is a fake random access. it can only get the current value."
        v = self._next_val
        self._update_next()
        self._current += 1
        return v

    def __iter__(self):
        while not self._stopped:
            yield self._current
        raise StopIteration


def iter_to_kwargs_for_cached_loop(a):
    assert isinstance(a, typing.Iterable), "a must be an iterable"

    func = _Iter2DataFunc(a)

    return dict(function=func, indexes=func)
