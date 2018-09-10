import numpy as np


class RecordedDataFetcher:
    def __init__(self, data_mod, batch_size, debug_text=None):
        self.iter = 0
        self.data_mod = data_mod
        self.batch_size = batch_size
        self.debug_text = debug_text

    def next_batch(self):
        self.iter += 1
        if self.debug_text is not None:
            print("%s [iter %d]" % (str(self.debug_text), self.iter))
        raw_data = self.data_mod(self.batch_size)
        if isinstance(raw_data, np.ndarray):
            raw_data = (raw_data,)
        data = []
        data_shape = []
        for i in range(len(raw_data)):
            raw_data_i = raw_data[i]
            assert len(raw_data_i) == self.batch_size, "the data batch_size is wrong"
            is_np_i = isinstance(raw_data_i, np.ndarray)
            if is_np_i:
                data_i = raw_data_i
                data_shape_i = np.tile(
                    np.reshape(np.array(data_i.shape[1:], np.int64), [1, -1]),
                    [self.batch_size, 1]
                )
            else:
                data_i = []
                data_shape_i = []
                for j in range(self.batch_size):
                    d = np.array(raw_data_i[j])
                    data_i.append(d)
                    data_shape_i.append(d.shape)
                data_shape_i = np.array(data_shape_i, np.int64)
                max_data_shape_i = np.max(data_shape_i, axis=0)
                for j in range(self.batch_size):
                    if all(max_data_shape_i == data_shape_i[j]):
                        continue
                    data_ij = np.zeros(max_data_shape_i, data_i[j].dtype)
                    data_ij[tuple(slice(c) for c in data_shape_i[j])] = data_i[j]
                    data_i[j] = data_ij
                data_i = np.array(data_i)
            data.append(data_i)
            data_shape.append(data_shape_i)
        return tuple(data + data_shape)

