# UNFINISHED

import mxnet as mx


class MXDataIterFromLoader(mx.io.DataIter):

    def __init__(self, dataloader, batch_size, data_fields=None, label_fields=None):
        super().__init__()

        # set up data loader and batch size
        self._dataloader = dataloader
        self._batch_size = batch_size

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
        self._data_field_indexes = self._dataloader.index_in_fields(data_fields)
        self._label_field_indexes = self._dataloader.index_in_fields(label_fields)

        # status variable
        self._cur = 0
        self._data = None
        self._label = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def provide_data(self):
        # *** remember to assign batch size
        return [
            (self._dataloader.output_keys()[i], self._dataloader.output_shapes()[i]) for i in self._data_field_indexes
        ]

    @property
    def provide_label(self):
        # *** remember to assign batch size
        return [
            (self._dataloader.output_keys()[i], self._dataloader.output_shapes()[i]) for i in self._label_field_indexes
        ]

    def reset(self):
        self._dataloader.reset()

    def __next__(self):
        pass

