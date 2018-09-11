from abc import ABCMeta, abstractmethod
import numpy as np


class Net:

    __metaclass__ = ABCMeta

    def __init__(self):
        self._dataset = None
        self._cur_pos = 0
        self._cur_epoch = 0
        self._cur_iter = 0
        self._num_samples = None
        self._num_fields = None
        self._total_pos = 0

    def set_dataset(self, dataset):
        self._dataset = list(dataset)
        self._num_fields = len(self._dataset)
        self._num_samples = self._dataset[0].shape[0]

    def __call__(self, *args, **kwargs):
        return self.next_batch(*args, **kwargs)

    def num_samples(self):
        return self._num_samples

    def epoch(self):
        return self._cur_epoch

    def iter(self):
        return self._cur_iter

    def num_fields(self):
        return self._num_fields

    def num_samples_finished(self):
        return self._cur_pos

    def reset(self):
        """ Reset the state of the data loader
        E.g., the reader points at the beginning of the dataset again
        :return: None
        """
        self._cur_pos = 0
        self._cur_epoch = 0
        self._cur_iter = 0

    def next_batch(self, batch_size):
        """ fetch the next batch

        :param batch_size: next batch_size
        :return: a tuple includes all data
        """

        start_pos = self._cur_pos
        end_pos = self._cur_pos + batch_size
        this_batch = [None]*self._num_fields
        cur_epoch = self._cur_epoch
        while True:
            out_of_range = end_pos > self._num_samples
            if out_of_range:
                new_start_pos = 0
                new_end_pos = end_pos - self._num_samples
                end_pos = self._num_samples

            for i in range(self._num_fields):
                if this_batch[i] is None:
                    this_batch[i] = self._dataset[i][start_pos:end_pos]
                else:
                    this_batch[i] = np.concatenate(
                        (this_batch[i], self._dataset[i][start_pos:end_pos]), 0)

            if not out_of_range:
                break

            cur_epoch += 1
            start_pos = new_start_pos
            end_pos = new_end_pos

        self._cur_epoch = cur_epoch
        self._cur_pos = end_pos%self._num_samples
        self._cur_iter += 1

        out_type = self.output_types()
        for i in range(len(this_batch)):
            t_dtype = out_type[i]
            if isinstance(t_dtype, str):
                t_dtype = getattr(np, t_dtype)
            this_batch[i] = this_batch[i].astype(t_dtype)

        return this_batch

    @staticmethod
    @abstractmethod
    def output_types():     # only used for net instance
        pass

    @staticmethod
    @abstractmethod
    def output_shapes():
        pass

    @staticmethod
    @abstractmethod
    def output_ranges():
        pass

    @staticmethod
    @abstractmethod
    def output_keys():
        pass

    # extra helper functions -----------------------------------------

    def limit_to_classes(self, class_list):

        if self._cur_pos > 0:
            raise ValueError("Cannot prune the dataset during reading")

        all_output_keys = list(self.output_keys())
        if "class" in all_output_keys:
            class_field_ind = all_output_keys.index("class")
        else:
            raise ValueError("there is no class annotation")
        if not isinstance(class_list, (list, tuple, set)):
            class_list = [class_list]
        class_list = np.array(list(class_list))
        class_field = self._dataset[class_field_ind]
        chosen_idxb = np.in1d(class_field, class_list)
        for i in range(len(self._dataset)):
            self._dataset[i] = self._dataset[i][chosen_idxb]
        self._num_samples = self._dataset[0].shape[0]

        map2new = dict()
        for i in range(len(class_list)):
            map2new[class_list[i]] = i

        class_data = self._dataset[class_field_ind]
        for i in np.arange(self._num_samples):
            class_data[i] = map2new[class_data[i]]

