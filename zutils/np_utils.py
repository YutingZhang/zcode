import numpy as np


def onthot_to_int(onehot_array, axis=1, dtype=np.int64, keepdims=False):

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
    if len(tensor_list) == 1:
        return tensor_list[0][np.newaxis, :]

    ndim = len(tensor_list[0].shape)
    dimensions = [len(tensor_list)]  # first dim is batch size
    for dim in range(ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))

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

