from . import recursive_utils as ru
import numpy as np


__all__ = ['into_ndarray', 'into_structured_ndarray']


class _IntoNDArray:
    @staticmethod
    def condition_func(*args):
        a = args[0]
        return isinstance(a, np.ndarray)

    @staticmethod
    def condition_func_for_structured(*args):
        a = args[0]
        return isinstance(a, np.ndarray) and a.dtype.names is not None

    @staticmethod
    def recursive_apply_func(*args):
        s = args[0].shape
        field_names = args[0].dtype.names
        out = np.ndarray(shape=s, dtype=args[0].dtype)
        for index in np.ndindex(*s):
            if field_names is None:
                for index in np.ndindex(*s):
                    sub_args = [
                        a[index] for a in args
                    ]
                    out[index] = ru.CustomRecursiveApplyScope.current_bound_recursive_apply(
                        *sub_args
                    )
            else:
                for index in np.ndindex(*s):
                    for fn in field_names:
                        sub_args = [
                            a[index][fn] for a in args
                        ]
                        out[index][fn] = ru.CustomRecursiveApplyScope.current_bound_recursive_apply(
                            *sub_args
                        )
        return out


def into_structured_ndarray():
    return ru.CustomRecursiveApplyScope(
        _IntoNDArray.condition_func_for_structured, _IntoNDArray.recursive_apply_func
    )


def into_ndarray():
    return ru.CustomRecursiveApplyScope(
        _IntoNDArray.condition_func, _IntoNDArray.recursive_apply_func
    )
