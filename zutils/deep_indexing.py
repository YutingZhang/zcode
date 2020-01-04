from typing import Dict, Tuple, Any, Union, Callable
import copy


__all__ = [
    'apply_to_index_paths',
    'apply_to_single_index_path',
    'INDEX_WILDCARD',
    'KEY_WILDCARD',
]


class _IndexWildcard:
    pass


INDEX_WILDCARD = _IndexWildcard()


class _KeyWildcard:
    pass


KEY_WILDCARD = _KeyWildcard()


class _NotSpecified:
    pass


INDEX_PATH = Tuple[Any]


def apply_to_index_paths(a, func: Callable, index_path2extra_args: Dict[INDEX_PATH, Tuple[Any]]):

    b = a
    for index_path, extra_args in index_path2extra_args.items():
        b = apply_to_single_index_path(
            b, func, index_path, extra_args
        )
    return b


def apply_to_single_index_path(
        src, func: Callable, index_path: INDEX_PATH, extra_args: Tuple[Any]
):

    if not isinstance(index_path, tuple):
        index_path = tuple(index_path)

    assert index_path, "at least one index needs to be in the index path"

    tuple_type = None
    if isinstance(src, tuple):
        tuple_type = type(src)
        dst = list(src)
    else:
        dst = copy.copy(src)

    k = index_path[0]
    index_iter = None
    if isinstance(k, _IndexWildcard):
        index_iter = range(len(src))
    elif isinstance(k, _KeyWildcard):
        index_iter = src

    if index_iter is not None:
        for subkey in src:
            dst[subkey] = apply_to_single_index_path(
                src, func, (subkey,) + index_path[1:], extra_args
            )
    else:

        if len(index_path) > 1:
            dst[k] = apply_to_single_index_path(
                src[k], func, index_path[1:], extra_args
            )
        else:
            dst[k] = func(src[k], *extra_args)

    if tuple_type is not None:
        dst = tuple_type(dst)

    return dst


def set_to_index_path(a, index_path: INDEX_PATH, value: Any):
    raise NotImplementedError


def get_from_index_path(a, index_path: INDEX_PATH):
    raise NotImplementedError

