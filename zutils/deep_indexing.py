from typing import Dict, Tuple, Any, Union, Callable
import copy


__all__ = [
    'apply_to_index_path',
    'INDEX_WILDCARD',
    'KEY_WILDCARD',
]


class _INDEX_WILDCARD:
    pass


INDEX_WILDCARD = _INDEX_WILDCARD()


class _KEY_WILDCARD:
    pass


KEY_WILDCARD = _KEY_WILDCARD()


class _NotSpecified:
    pass


INDEX_PATH = Tuple[Any]


def apply_to_index_path(a, func: Callable, index_path2extra_args: Dict[INDEX_PATH, Tuple[Any]]):

    out = copy.copy(a)
    for index_path, extra_args in index_path2extra_args.items():
        _apply_to_single_index_path(
            a, out, func, index_path, extra_args
        )
    return out


def _apply_to_single_index_path(
        src, dst, func: Callable, index_path: INDEX_PATH, extra_args: Tuple[Any]
):

    if not isinstance(index_path, tuple):
        index_path = tuple(index_path)

    assert index_path, "at least one index needs to be in the index path"

    pre_dst = _NotSpecified
    k = _NotSpecified
    for i, k in enumerate(index_path):
        pre_dst = dst
        index_iter = None
        if isinstance(k, _INDEX_WILDCARD):
            index_iter = range(len(src))
        elif isinstance(k, _KEY_WILDCARD):
            index_iter = src

        if index_iter is None:
            if dst[k] is src[k]:
                dst[k] = copy.copy(dst[k])
            src = src[k]
            dst = dst[k]
        else:
            for subkey in src:
                _apply_to_single_index_path(
                    src, dst, func, (subkey,) + index_path[i+1:], extra_args
                )
            return

    assert pre_dst is not _NotSpecified, "at least one index needs to be in the index path"

    pre_dst[k] = func(src, *extra_args)
