from .cached_loop_base import CachedLoopBase
import typing
from z_python_utils.async_executors import WorkerExecutor


__all__ = ['CachedLoopExecutor']


def _get_result(r):
    return r.result()


class CachedLoopExecutor(CachedLoopBase):

    def __init__(
            self,
            function: typing.Union[typing.Callable, typing.List[typing.Callable], typing.Tuple[typing.Callable]],
            indexes: typing.Iterable, max_workers: int, use_thread_pool: bool = False, pickle_to_file: bool = False,
            cache_size: typing.Optional[int] = None,
            batched: bool = False
    ):
        if max_workers is not None and max_workers >= 1:
            self.executor = WorkerExecutor(max_workers=max_workers, use_thread_pool=use_thread_pool, pickle_to_file=pickle_to_file)
            if cache_size is None:
                cache_size = max_workers * 2
            super().__init__(
                function=function, indexes=indexes, batched=batched, cache_size=cache_size,
                submit_func=self.executor.submit, get_func=_get_result
            )
        else:
            cache_size = 0
            super().__init__(
                function=function, indexes=indexes, batched=batched, cache_size=cache_size,
            )

