# UNFINISHED

import mxnet as mx
import numpy as np
from zutils.np_utils import tensor_vstack


class MXDataIterFromLoader(mx.io.DataIter):

    def __init__(self, dataloader, batch_size, data_fields=None, label_fields=None, clear_epoch=None):
        super().__init__(batch_size=batch_size)

        # set up data loader and batch size
        self._dataloader = dataloader
        self._batch_size = batch_size

        if clear_epoch is None:
            if self.num_images is None:
                clear_epoch = False
            else:
                clear_epoch = True
        if clear_epoch:
            assert self.num_images is not None, \
                "cannot clear_epoch if the underlying dataloader has not definte number of samples"
        self._clear_epoch = clear_epoch
        self._reached_last_iter = False

        # set up data fields and label_fields
        if data_fields is None:
            if label_fields is None:
                # all fields as data
                data_fields = list(self._dataloader.output_keys())
                label_fields = []
            else:
                # infer data from label
                label_fields = list(label_fields)
                data_fields = [k for k in self._dataloader.output_keys() if k not in label_fields]
        else:
            data_fields = list(data_fields)
            if label_fields is None:
                # infer label fields
                label_fields = [k for k in self._dataloader.output_keys() if k not in data_fields]
            else:
                label_fields = list(label_fields)
        assert not bool(set(data_fields) - set(self._dataloader.output_keys())), \
            "not all data_fields exist"
        assert not bool(set(label_fields) - set(self._dataloader.output_keys())), \
            "not all label_fields exist"
        self._data_field_indexes = self._index_in_dataloader_fields(data_fields)
        self._label_field_indexes = self._index_in_dataloader_fields(label_fields)

        # data and label meta
        self._data_key_size = [
            (self._dataloader.output_keys()[i], self._dataloader.output_shapes()[i]) for i in self._data_field_indexes
        ]
        self._label_key_size = [
            (self._dataloader.output_keys()[i], self._dataloader.output_shapes()[i]) for i in self._label_field_indexes
        ]

        def _fill_batch_size(_key_size_p):
            for _i in range(len(_key_size_p)):
                _k, _s = _key_size_p[_i]
                if _s is not None and _s:
                    assert _s[0] is None or _s[0] == self._batch_size, "conflicted batch_size"
                    _s = (self._batch_size,) + tuple(_s[1:])
                _key_size_p[_i] = _k, _s

        _fill_batch_size(self._data_key_size)
        _fill_batch_size(self._label_key_size)

        self._unknown_data_size = any((v is None or any(e is None for e in v)) for _, v in self._data_key_size)
        self._unknown_label_size = any((v is None or any(e is None for e in v)) for _, v in self._label_key_size)

        # status variable
        self._data = None
        self._label = None
        self._pad = None

        # prefetch if needed
        self._is_data_cached = False
        if self._unknown_data_size or self._unknown_label_size:
            # get first batch to fill in provide_data and provide_label
            self.next()
            self._is_data_cached = True

    def _index_in_dataloader_fields(self, field_names):
        if isinstance(field_names, str):
            return self._dataloader.output_keys().index(field_names)
        else:
            return list(self._dataloader.output_keys().index(a) for a in field_names)

    @property
    def provide_data(self):
        if self._unknown_data_size:
            return [(k[0], v.shape) for k, v in zip(self._data_key_size, self._data)]
        else:
            return self._data_key_size

    @property
    def provide_label(self):
        if self._unknown_label_size:
            return [(k[0], v.shape) for k, v in zip(self._label_key_size, self._label)]
        else:
            return self._label_key_size

    @property
    def num_images(self):
        if hasattr(self._dataloader, "num_samples"):
            return self._dataloader.num_samples()
        else:
            return None

    #@property
    #def batch_size(self):
    #    return self._batch_size

    @property
    def epoch(self):
        return self._dataloader.epoch()

    @property
    def clear_epoch(self):
        return self._clear_epoch

    @property
    def eof(self):
        return self._clear_epoch and self._reached_last_iter

    def reset(self):
        self._dataloader.reset()
        self._reached_last_iter = False

    def __next__(self):
        return self.next()

    def current_batch(self):
        return mx.io.DataBatch(
            data=self._data, label=self._label, pad=self._pad,
            provide_data=self.provide_data, provide_label=self.provide_label
        )

    def next(self):

        if self._is_data_cached:
            self._is_data_cached = False
            return self.current_batch()

        if self.eof:
            raise StopIteration

        self._reached_last_iter = self.num_images is not None and (
                self._dataloader.num_samples() - self._dataloader.num_samples_finished() <= self.batch_size
        )

        if self.eof:
            the_batch_size = self._dataloader.num_samples() - self._dataloader.num_samples_finished()
        else:
            the_batch_size = self.batch_size

        pad = self.batch_size - the_batch_size

        d = list(self._dataloader(the_batch_size))

        _output_types = self._dataloader.output_types()
        for i in set(self._data_field_indexes + self._label_field_indexes):
            ot = _output_types[i]
            if isinstance(ot, str):
                ot = getattr(np, ot)
            d_i = np.array(tensor_vstack(d[i]), dtype=ot)
            if pad:
                padding_i = np.tile(d_i[:1], [pad] + [1] * (d_i.ndim-1))
                d_i = np.concatenate(
                    [d_i, padding_i], axis=0
                )
            d[i] = d_i

        self._data = [d[i] for i in self._data_field_indexes]
        self._label = [d[i] for i in self._label_field_indexes]
        self._pad = pad

        return self.current_batch()



