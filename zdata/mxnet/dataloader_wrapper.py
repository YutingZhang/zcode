import mxnet as mx
import numpy as np
import math
from zutils.np_utils import tensor_vstack
from functools import partial
from typing import Callable


class MXDataIterFromLoader(mx.io.DataIter):

    def __init__(
            self, dataloader, batch_size, data_fields=None, label_fields=None, clear_epoch=None,
            data_vstack_pad=0, label_vstack_pad=0
    ):
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

        # vstack pad
        if not isinstance(data_vstack_pad, (list, tuple)):
            data_vstack_pad = [data_vstack_pad] * len(self._data_field_indexes)
        if not isinstance(label_vstack_pad, (list, tuple)):
            label_vstack_pad = [label_vstack_pad] * len(self._label_field_indexes)
        assert len(data_vstack_pad) == len(self._data_field_indexes), \
            "data_vstack_pad should have the same len as data_fields"
        assert len(label_vstack_pad) == len(self._label_field_indexes), \
            "label_vstack_pad should have the same len as label_fields"
        self._data_vstack_pad = data_vstack_pad
        self._label_vstack_pad = label_vstack_pad

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

        def _to_mxndarray(_i, _pad_val):
            ot = _output_types[_i]
            if isinstance(ot, str):
                ot = getattr(np, ot)
            d_i = np.array(tensor_vstack(d[_i], pad=_pad_val), dtype=ot)
            if pad:
                padding_i = np.tile(d_i[:1], [pad] + [1] * (d_i.ndim-1))
                d_i = np.concatenate(
                    [d_i, padding_i], axis=0
                )
            return mx.nd.array(d_i)

        self._data = [
            _to_mxndarray(i, pad_val) for i, pad_val in zip(self._data_field_indexes, self._data_vstack_pad)
        ]
        self._label = [
            _to_mxndarray(i, pad_val) for i, pad_val in zip(self._label_field_indexes, self._label_vstack_pad)
        ]
        self._pad = pad

        return self.current_batch()


class DummyGluonDataloaderFromLoader:

    def __init__(
            self, dataloader, batch_size, data_fields=None, clear_epoch=None, batchify_fn=None, data_vstack_pad=None,
            reset_at_iter_end=False
    ):

        # set up data loader and batch size
        self._dataloader = dataloader
        self._batch_size = batch_size
        self._reset_at_iter_end = reset_at_iter_end

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
            data_fields = list(self._dataloader.output_keys())

        assert not bool(set(data_fields) - set(self._dataloader.output_keys())), \
            "not all data_fields exist"

        self._data_field_indexes = self._index_in_dataloader_fields(data_fields)

        # batchify func
        if batchify_fn is None:
            if data_vstack_pad is None:
                data_vstack_pad = 0
            if not isinstance(data_vstack_pad, (list, tuple)):
                data_vstack_pad = (data_vstack_pad,) * len(self._data_field_indexes)
            batchify_fn = tuple(partial(tensor_vstack, pad=dvp) for dvp in data_vstack_pad)
        else:
            assert data_vstack_pad is None, "must not specify data_vstack_pad with batchify func"
        if not isinstance(batchify_fn, Callable) and isinstance(batchify_fn, (list, tuple)):
            batchify_fn_elts = tuple(
                (dvp if isinstance(dvp, Callable) else partial(tensor_vstack, pad=dvp)) for dvp in batchify_fn
            )
            batchify_fn = lambda d: tuple(
                bfe(d[i]) for i, bfe in zip(range(len(batchify_fn_elts)), batchify_fn_elts)
            )
        self._batchify_fn = batchify_fn

    def _index_in_dataloader_fields(self, field_names):
        if isinstance(field_names, str):
            return self._dataloader.output_keys().index(field_names)
        else:
            return list(self._dataloader.output_keys().index(a) for a in field_names)

    @property
    def num_images(self):
        if hasattr(self._dataloader, "num_samples"):
            return self._dataloader.num_samples()
        else:
            return None

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epoch(self):
        return self._dataloader.epoch()

    @property
    def clear_epoch(self):
        return self._clear_epoch

    @property
    def eof(self):
        return self._reached_last_iter

    def reset(self):
        self._dataloader.reset()
        self._reached_last_iter = False

    @property
    def num_batch(self):
        return math.ceil(self.num_images / self.batch_size)

    def __iter__(self):
        self._reached_last_iter = True
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.num_batch

    def next(self):

        if self.eof:
            if self._reset_at_iter_end:
                self._dataloader.reset()
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

        _data = self._batchify_fn(tuple(d[i] for i in self._data_field_indexes))

        def _to_typed_batch(_i, _d_i):
            ot = _output_types[_i]
            if isinstance(ot, str):
                ot = getattr(np, ot)
            d_i = np.array(_d_i, dtype=ot)
            if pad:
                padding_i = np.tile(d_i[:1], [pad] + [1] * (d_i.ndim-1))
                d_i = np.concatenate(
                    [d_i, padding_i], axis=0
                )
            return d_i

        data = tuple(
            _to_typed_batch(i, d_i) for i, d_i in zip(self._data_field_indexes, _data)
        )

        return data


GluonDataloaderFromLoader = DummyGluonDataloaderFromLoader

