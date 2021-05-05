import numpy as np
from functools import partial
from typing import Callable


def onehot_to_int(onehot_array, axis=1, dtype=np.int64, keepdims=False):

    s = onehot_array.shape
    num = s[axis]
    nonzero_indexes = np.nonzero(onehot_array)
    index_arr = np.array(np.arange(num))
    all_indexes = index_arr[nonzero_indexes[axis]]
    if keepdims:
        s[axis] = 1
    else:
        s = s[0:axis] + s[axis+1:]
    all_indexes = np.reshape(all_indexes, s)

    return all_indexes


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors by adding a new axis
    expand dims if only 1 tensor
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    Source: improved from mxnet RCNN example
    """
    if isinstance(tensor_list, np.ndarray):
        return tensor_list
    if len(tensor_list) == 1:
        return tensor_list[0][np.newaxis, :]

    ndim = max(len(tensor.shape) for tensor in tensor_list)
    dimensions = [len(tensor_list)]  # first dim is batch size
    for dim in range(ndim):
        dimensions.append(max([(tensor.shape[dim] if len(tensor.shape) > dim else 0) for tensor in tensor_list]))

    tensor_list = [*tensor_list]
    for i in range(len(tensor_list)):
        extra_ndim = ndim - len(tensor_list[i].shape)
        if extra_ndim > 0:
            tensor_list[i] = np.reshape(tensor_list[i], tensor_list[i].shape + (1,) * extra_ndim)

    dtype = tensor_list[0].dtype
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)

    for ind, tensor in enumerate(tensor_list):
        sub_tensor_ind = (ind,) + tuple(slice(s) for s in tensor.shape)
        all_tensor[sub_tensor_ind] = tensor

    return all_tensor


class _BatchifyTuple:

     def __init__(self, *batchify_fns):
         self._batchify_fns = batchify_fns

     def __call__(self, d):
         return tuple(
             bfe(d_i) for bfe, d_i in zip(self._batchify_fns, zip(*d))
         )


def standardize_batchify_fn(
        num_data_fields, batchify_fn=None, data_vstack_pad=None, tensor_vstack_func=tensor_vstack
):
    # batchify func
    if batchify_fn is None:
        if data_vstack_pad is None:
            data_vstack_pad = 0
        if not isinstance(data_vstack_pad, (list, tuple)):
            data_vstack_pad = (data_vstack_pad,) * num_data_fields
        batchify_fn = tuple(partial(tensor_vstack_func, pad=dvp) for dvp in data_vstack_pad)
    else:
        assert data_vstack_pad is None, "must not specify data_vstack_pad with batchify func"
    if not isinstance(batchify_fn, Callable) and isinstance(batchify_fn, (list, tuple)):
        batchify_fn_elts = tuple(
            (dvp if isinstance(dvp, Callable) else partial(tensor_vstack_func, pad=dvp)) for dvp in batchify_fn
        )

        batchify_fn = _BatchifyTuple(*batchify_fn_elts)

    return batchify_fn


def as_numpy_object_vec(a):
    n = len(a)
    b = np.empty([n], dtype=object)
    for i, elt in enumerate(a):
        b[i] = elt
    return b


class FindIntervalsForPoint:
    def __init__(self, left, right):
        self._interval_seps, self._indexes = self.get_intervals_and_indexes(left, right)

    @staticmethod
    def get_intervals_seps_and_indexes(left, right):
        left = np.asarray(left)
        right = np.asarray(right)
        n = len(right)
        assert len(left) == n, "left and right must have the same length"
        if not n:
            return np.zeros([0], dtype=left.dtype), np.zeros([0], dtype=object)

        x = np.concatenate([left, right])
        si = np.argsort(x, axis=0)
        sx = x[si]
        ux, ui = np.unique(sx, unique_indices=True)
        ui = np.append(ui, 2 * n)
        indexes = np.empty([len(ui)], dtype=object)
        indexes[0] = set()
        overlapped_set = set()
        for j in np.arange(len(ux)):
            for i in si[np.arange(ui[j], ui[j + 1])]:
                if i < n:
                    overlapped_set.add(i)
                else:
                    overlapped_set.remove(i - n)
            indexes[j] = set(overlapped_set)

        interval_seps = ux
        return interval_seps, indexes

    def interval_ids(self, y):
        y = np.asarray(y)
        js = np.searchsorted(self._interval_seps, y, side='right', sorter=self._si_left)
        return self._indexes[js]

