from typing import Callable
from zutils.recursive_utils import recursive_apply, recursive_flatten_to_list
from threading import Lock


__all__ = ['modify_callable_to_cache_results', 'ResultCachingFunction']


def modify_callable_to_cache_results(func: Callable, max_cache_size: int = 1, try_to_modify_input: bool = True):
    if try_to_modify_input and hasattr(func, '__call__'):
        result_caching_func = ResultCachingFunction(getattr(func, '__call__'), max_cache_size=max_cache_size)
        setattr(func, '__call__', result_caching_func)
        return func
    else:
        return ResultCachingFunction(func, max_cache_size=max_cache_size)


def _get_single_obj_id(obj):
    if isinstance(obj, (int, float, bool, str)):
        return obj
    else:
        return id(obj),


def get_hash_and_id_struct_for_recursive_object(obj):
    id_struct = recursive_apply(None, _get_single_obj_id, obj)
    my_hash = hash(tuple(recursive_flatten_to_list(None, id_struct)))
    return my_hash, id_struct


class ResultCachingFunction:
    def __init__(self, func: Callable, max_cache_size: int = 1):

        self._func = func

        self._max_cache_size = max_cache_size

        self._cache_lock = Lock()
        with self._cache_lock:
            self._cache = dict()
            self._cache_freshness = dict()
            self._current_freshness = 0

    def set_max_cache_size(self, max_cache_size: int):
        with self._cache_lock:
            self._max_cache_size = max_cache_size
        self._prune_cache()

    def get_max_cache_size(self):
        return self._max_cache_size

    def clear_cache(self):
        self._cache.clear()

    def _refresh_item_cache(self, arg_hash, cache_entry_id):
        with self._cache_lock:
            self._cache_freshness[(arg_hash, cache_entry_id)] = self._current_freshness
            self._current_freshness += 1
            max_freshness = max(self._max_cache_size * 2, 1000000)
            if self._current_freshness > max_freshness:
                new_cche_freshness = dict(
                    (k, the_freshness) for the_freshness, (k, _)
                    in enumerate(sorted(self._cache.items(), key=lambda x: x[1]))
                )
                self._cache_freshness.clear()
                self._cache_freshness.update(new_cche_freshness)
                self._current_freshness = len(self._cache_freshness)

        self._prune_cache()

    def _prune_cache(self):
        with self._cache_lock:
            while len(self._cache_freshness) > self._max_cache_size:
                k = min(self._cache_freshness.items(), key=lambda x: x[1])[0]
                del self._cache_freshness[k]

    def __call__(self, *args, **kwargs):

        # indexing arguments
        arg_hash, args_id_struct = get_hash_and_id_struct_for_recursive_object((args, kwargs))

        out = None

        with self._cache_lock:
            # retrieve or create entry
            if arg_hash in self._cache:
                the_cache_list = self._cache[arg_hash]
            else:
                self._cache[arg_hash] = the_cache_list = []

            cache_entry_id = None
            for i, cache_entry in enumerate(the_cache_list):
                if cache_entry[0] == args_id_struct:
                    cache_entry_id = i
                    out = cache_entry[1]
                    break

        if cache_entry_id is None:
            out = self._func(*args, **kwargs)
            with self._cache_lock:
                cache_entry_id = len(the_cache_list)
                the_cache_list.append((args_id_struct,out))

        self._refresh_item_cache(arg_hash, cache_entry_id)

        return out

