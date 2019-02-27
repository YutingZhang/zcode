from . import recursive_utils as ru
import numpy as np


__all__ = ['into_structured_ndarray']


class _IntoStructuredNDArray:
    @staticmethod
    def condition_func(*args):
        a = args[0]
        return isinstance(a, np.ndarray) and a.dtype.names is not None

    @staticmethod
    def recursive_apply_func(*args):
        s = args[0].shape
        field_names = args[0].dtype.names
        out = np.ndarray(shape=s, dtype=args[0].dtype)
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
        _IntoStructuredNDArray.condition_func, _IntoStructuredNDArray.recursive_apply_func
    )
