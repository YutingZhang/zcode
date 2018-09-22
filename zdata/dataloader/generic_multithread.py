import time
from collections import deque, Callable, OrderedDict, Sequence
import threading
import random
# from multiprocessing.dummy import Pool
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
from zdata.dataloader.utils import *


class AbsFieldDataLoader:

    def __init__(self):
        self._af_all_fields = []
        self._af_output_key_mapping = dict()
        self._af_output_type_mapping = dict()
        self._af_output_shape_mapping = dict()

    @property
    def mutable_all_fields(self):
        return self._af_all_fields

    @property
    def mutable_field_key_mapping(self):
        return self._af_output_key_mapping

    @property
    def mutable_field_type_mapping(self):
        return self._af_output_type_mapping

    @property
    def mutable_field_shape_mapping(self):
        return self._af_output_shape_mapping

    def output_keys(self):
        all_fields = []
        for k in self._af_all_fields:
            if k in self._af_output_key_mapping:
                all_fields.append(self._af_output_key_mapping[k])
            else:
                all_fields.append(k)
        return tuple(all_fields)

    def output_types(self):
        return tuple(
            self._af_output_type_mapping[k] for k in self.output_keys()
        )

    def output_shapes(self):
        all_shapes = []
        for k in self.output_keys():
            if k in self._af_output_shape_mapping:
                all_shapes.append(self._af_output_shape_mapping[k])
            else:
                all_shapes.append(None)
        return tuple(all_shapes)


class MultiThreadDataLoader(AbsFieldDataLoader):

    # subclass should implement the following --------------------------------------------

    def read_many_data(self, read_batch_id):
        s_data = self.read_single_data(read_batch_id)
        if isinstance(s_data, tuple):
            tmp_data = tuple([a] for a in s_data)
        else:
            tmp_data = [s_data]
        return tmp_data

    def read_single_data(self, read_batch_id):
        assert False, "read_singe_data or read_many_data must be implemented"
        return [None], [None]

    def reset_data(self):
        pass

    def estimated_num_samples_per_read_batch(self):
        return 1

    # --------------------------------------------------------------------------------------
    # for protected subclass

    def _reset_data(self):
        pass

    # --------------------------------------------------------------------------------------

    def __init__(
            self,
            min_cache_size=200, max_cache_size=None,
            pool_size=None,
            is_random=True,     # sequential otherwise
            rand_seed=None,
            num_read_batch=None,  # must specified if sequential
    ):

        super().__init__()

        self._lock4cache = threading.Lock()
        self._pool_size = cpu_count() if pool_size is None else pool_size
        self._parallel = self._pool_size > 0
        if not self._parallel:      # useful for single thread debugging
            self._pool_size = 1
        self._pool = Pool(self._pool_size)
        self._cache_thread = None
        self._reset_preload_thread()

        self._min_cache_size = min_cache_size
        self._max_cache_size = max_cache_size if max_cache_size is not None else self._min_cache_size * 2
        self._cache = deque()

        self._is_random = is_random
        self._rand_seed_for_cache = rand_seed
        self._rand_stream_for_cache = None
        self._num_read_batch = num_read_batch
        self._sequential_loc = None
        self._reset_loc()

        if hasattr(self, "read_many_data_identifier"):
            self.data_identifier_list = self._data_identifier_list

    def _reset_loc(self):
        if self._is_random:
            self._rand_stream_for_cache = random.Random(self._rand_seed_for_cache)
        else:
            self._sequential_loc = 0

    def _get_read_batch_id(self):
        if self._is_random:
            r = self._rand_stream_for_cache.random()
            if self._num_read_batch is None:    # return a random number in [0,1) if the batch num is unknown
                return r
            else:
                return int(r * self._num_read_batch)
        else:
            cur_loc = self._sequential_loc
            self._sequential_loc = cur_loc + 1
            if self._sequential_loc >= self._num_read_batch:
                self._sequential_loc -= self._num_read_batch
            return cur_loc

    def reset(self):
        self._join_preload_thread()
        self._lock4cache.acquire()

        self._cache.clear()

        self._reset_data()
        self.reset_data()

        self._reset_loc()
        self._reset_preload_thread()
        self._lock4cache.release()

    def _compute_min_cache_size(self, batch_size_needed=None):
        if batch_size_needed is None:
            min_cache_size = self._min_cache_size
        else:
            min_cache_size = max(self._min_cache_size, batch_size_needed * 3)
        return min_cache_size

    def _need_preload(self, batch_size_needed=None):
        return len(self._cache) < self._compute_min_cache_size()

    def _preload_if_needed(self, batch_size_needed=None):
        if self._parallel:
            if self._cache_thread.isAlive():
                return
            if not self._need_preload():
                return
            self._cache_thread = threading.Thread(
                target=(lambda x: lambda: self._preload_dataset(x))(batch_size_needed)
            )
            self._cache_thread.start()
        else:
            # no parallel
            self._preload_dataset(batch_size_needed)

    def _join_preload_thread(self):
        if self._parallel:
            while self._cache_thread.isAlive():
                self._cache_thread.join()

    def _reset_preload_thread(self):
        self._cache_thread = \
            threading.Thread(target=self._preload_dataset) if self._parallel else None

    def _preload_dataset(self, batch_size_needed=None):

        if not self._need_preload():
            return

        max_cache_size = max(self._max_cache_size, self._compute_min_cache_size())
        cache_batch_size = self._pool_size  # * math.ceil(10/self.estimated_num_samples_per_read_batch())
        if batch_size_needed is not None:
            cache_batch_size = max(batch_size_needed, cache_batch_size)
        cache_batch_size = self._pool_size * (math.ceil(cache_batch_size // self._pool_size))
        max_cache_size = max(max_cache_size, cache_batch_size*2)

        while len(self._cache) < max_cache_size:
            batch_ids = [None] * cache_batch_size
            for i in range(cache_batch_size):
                batch_ids[i] = self._get_read_batch_id()

            if self._pool_size > 1:
                # multiple thread
                tmp_data = self._pool.map(self._read_many_data, batch_ids)
            else:
                # single thread
                tmp_data = []
                for b_id in batch_ids:
                    tmp_data.append(self._read_many_data(b_id))
            tmp_data = tuple(zip(*tmp_data))

            self._lock4cache.acquire()
            _num_fileds = len(tmp_data)
            for i in range(cache_batch_size):
                subbatch_data = []
                for j in range(_num_fileds):
                    subbatch_data.append(tmp_data[j][i])
                subbatch_num_data = len(subbatch_data[0])
                # sanity check --------------------------------------
                for j in range(_num_fileds):
                    assert len(subbatch_data[j]) == subbatch_num_data, \
                        "batch size inconsistent across fields"
                # ---------------------------------------------------
                for k in range(subbatch_num_data):
                    a_data = tuple(subbatch_data[j][k] for j in range(_num_fileds))
                    self._cache.append(a_data)
            self._lock4cache.release()

    def _read_many_data(self, read_batch_id):
        return self.read_many_data(read_batch_id)

    def __call__(self, batch_size):
        return self.next_batch(batch_size)

    def next_batch(self, batch_size):

        if batch_size <= 0:
            return None

        self._preload_if_needed(batch_size_needed=batch_size)
        while len(self._cache) < batch_size:
            self._preload_if_needed(batch_size_needed=batch_size)
            time.sleep(0.1)

        self._lock4cache.acquire()
        this_batch = [None] * batch_size
        for i in range(batch_size):
            this_batch[i] = self._cache.popleft()
        self._lock4cache.release()
        eg_data = this_batch[0]
        if isinstance(eg_data, tuple):
            this_batch = tuple(list(a) for a in zip(*this_batch))

        return this_batch

    def _data_identifier_list(self, max_len=None):
        read_many_data_identifier_func = getattr(self, "read_many_data_identifier")
        l = []
        i = 0
        if max_len is None:
            for i in range(self._num_read_batch):
                l.extend(read_many_data_identifier_func(i))
        else:
            while i < self._num_read_batch and len(l) < max_len:
                l.extend(read_many_data_identifier_func(i))
            del l[max_len:]
        return l


class MultiThreadDataLoaderWithCount(MultiThreadDataLoader):

    def __init__(
            self,
            min_cache_size=200, max_cache_size=None,
            pool_size=None,
            is_random=True,     # sequential otherwise
            rand_seed=None,
            num_read_batch=None,  # must specified if sequential
    ):
        super().__init__(
            min_cache_size=min_cache_size, max_cache_size=max_cache_size,
            pool_size=pool_size, is_random=is_random, rand_seed=rand_seed,
            num_read_batch=num_read_batch
        )

        self._is_known_num_samples = \
            hasattr(self, "num_samples") and isinstance(getattr(self, "num_samples"), Callable)
        if self._is_known_num_samples:
            self.epoch = self._epoch
            self.iter = self._iter
            self.num_samples_finished = self._num_samples_finished
            self._reset_data()

    # def num_samples(self):

    def _reset_data(self):
        self._cur_epoch = 0
        self._cur_iter = 0
        self._cur_pos = 0

    def _epoch(self):
        return self._cur_epoch

    def _iter(self):
        return self._cur_iter

    def _num_samples_finished(self):
        return self._cur_pos

    def next_batch(self, batch_size):

        outputs = super().next_batch(batch_size=batch_size)

        if self._is_known_num_samples:
            _num_sample = getattr(self, "num_samples")()
            self._cur_iter += 1
            self._cur_pos += batch_size
            if _num_sample > 0:
                while self._cur_pos >= _num_sample:
                    self._cur_epoch += 1
                    self._cur_pos -= _num_sample

        return outputs

# ------------------------------------------------------


def get_field_dict(dataloader, *args, **kwargs):
    valid_field_names = dataloader.output_keys()
    if isinstance(valid_field_names, str):
        valid_field_names = (valid_field_names,)
    valid_field_names = tuple(valid_field_names)
    if not args and not kwargs:
        field_dict = OrderedDict(zip(valid_field_names, valid_field_names))
    else:
        field_dict = OrderedDict()
        for k in args:
            assert k in valid_field_names, "invalid field name"
            field_dict[k] = k
        for k, v in kwargs:
            assert k in valid_field_names, "invalid field name"
            field_dict[k] = v
    return field_dict


class MultiThreadDataLoaderWithTransformSupport(MultiThreadDataLoaderWithCount):

    def __init__(
            self,
            min_cache_size=200, max_cache_size=None,
            pool_size=None,
            is_random=True,     # sequential otherwise
            rand_seed=None,
            num_read_batch=None,  # must specified if sequential
    ):
        super().__init__(
            min_cache_size=min_cache_size, max_cache_size=max_cache_size,
            pool_size=pool_size, is_random=is_random, rand_seed=rand_seed,
            num_read_batch=num_read_batch
        )

        # set output meta
        if hasattr(self, "output_keys"):
            self._my_output_keys = getattr(self, "output_keys")
        else:
            self._my_output_keys = self._def_output_keys
        if hasattr(self, "output_types"):
            self._my_output_types = getattr(self, "output_types")
        else:
            self._my_output_typess = self._def_output_types
        if hasattr(self, "output_shapes"):
            self._my_output_shapes = getattr(self, "output_shapes")
        else:
            self._my_output_shapes = self._def_output_shapes
        self.output_keys = self._external_output_keys
        self.output_types = self._external_output_types
        self.output_shapes = self._external_output_shapes

        # transform list
        self._base_read_many_data = self.read_many_data
        self._transform_list = []

    # ----------------------------------------------------------------------------------

    def _def_output_keys(self):
        raise ValueError("output_keys not defined")

    def _def_output_types(self):
        raise ValueError("output_types not defined")

    def _def_output_shapes(self):
        return None

    def _external_output_keys(self):
        return self._my_output_keys()

    def _external_output_types(self):
        return self._my_output_types()

    def _external_output_shapes(self):
        return self._my_output_shapes()

    # ----------------------------------------------------------------------------------

    def _read_many_data(self, read_batch_id):
        out = self._base_read_many_data(read_batch_id)
        out = tuple(out)
        for t in self._transform_list:
            new_out = t(*out)
            out = tuple(new_out)
        return out

    # ----------------------------------------------------------------------------------

    def transform(self, transform_cls, *args, **kwargs):
        assert issubclass(transform_cls, DataTransform), "transform_cls must be a subclass of DataTransform"

        # create transform
        t = transform_cls(
            *args, **kwargs,
            input_keys=self.output_keys(),
            input_types=self.output_types(),
            input_shapes=self.output_shapes(),
        )

        # update output meta
        self._my_output_keys = t.output_keys
        self._my_output_types = t.output_types
        self._my_output_shapes = t.output_shapes

        # add transform
        self._transform_list.append(t)

        return self

    def t(self, transform_cls, *args, **kwargs):    # shorthand for transform
        return self.transform(transform_cls, *args, **kwargs)

    def rewrap(self, original_batch_size, *args, **kwargs):
        return DataLoaderWrap(self, original_batch_size, *args, **kwargs)

    def __or__(self, other):    # pipeline style for transform
        if not isinstance(other, Sequence):
            other = (other,)
        other = list(other)
        assert 1 <= len(other) <= 3, "the len of tuple must be 1, 2, or 3"
        if len(other) == 2:
            if isinstance(other[1], dict):
                other = [other[0], tuple(), other[1]]
            else:
                other = [other[0], tuple(), dict()]
        other[1] = tuple(other[1])
        assert isinstance(other[2], dict), "the kwargs must be passed by a dict"

        if isinstance(other[0], DataTransform):
            return self.transform(other[0], *other[1], **other[2])
        elif isinstance(other[0], DataLoaderWrap):
            return other[0](self, *other[1], **other[2])
        else:
            return ValueError("Unrecognized class")

    # for batch concat
    def batch(self, batch_size, *args, **kwargs):
        field_dict = get_field_dict(self, *args, **kwargs)
        return BatchConcatCache((self, batch_size, field_dict))

    # for field concat
    def fields(self, *args, **kwargs):
        field_dict = get_field_dict(self, *args, **kwargs)
        return FieldConcatCache(self, field_dict)


Net = MultiThreadDataLoaderWithTransformSupport
DataLoader = MultiThreadDataLoaderWithTransformSupport


# Wrap a data loader into the transform enabled multi-thread data loader -------------------------

def make_data_loader(obj, soft=False):
    if hasattr(obj, "make_loader"):
        dl = getattr(obj, "make_loader")()
    else:
        if soft:
            dl = obj
        else:
            raise ValueError("obj cannot be used to make data loader")
    return dl


class DataLoaderWrap(DataLoader):

    def __init__(
            self, dataloader, original_batch_size,
            min_cache_size=None, max_cache_size=None, pool_size=None
    ):
        self._dataloader = make_data_loader(dataloader, soft=True)
        self._original_batch_size = original_batch_size

        if hasattr(self._dataloader, "num_samples"):
            self.num_samples = getattr(self._dataloader, "num_samples")
        if hasattr(self._dataloader, "num_samples_finished"):
            self.num_samples_finished = getattr(self._dataloader, "num_samples_finished")
        if hasattr(self._dataloader, "iter"):
            self.iter = getattr(self._dataloader, "iter")
        if hasattr(self._dataloader, "epoch"):
            self.epoch = getattr(self._dataloader, "epoch")
        if hasattr(self._dataloader, "output_keys"):
            self.output_keys = getattr(self._dataloader, "output_keys")
        if hasattr(self._dataloader, "output_types"):
            self.output_types = getattr(self._dataloader, "output_types")
        if hasattr(self._dataloader, "output_shapes"):
            self.output_shapes = getattr(self._dataloader, "output_shapes")

        if min_cache_size is None:
            min_cache_size = self._original_batch_size * 2
        if max_cache_size is None:
            max_cache_size = min_cache_size * 2
        self._num_read_batch_for_original = \
            max(math.ceil(max_cache_size*2/original_batch_size) * 100, 100000)   # use a large enough number

        super().__init__(
            min_cache_size=min_cache_size, max_cache_size=max_cache_size, pool_size=pool_size,
            is_random=False,
            num_read_batch=self._num_read_batch_for_original
        )
        self._pos_read_batch_for_original = 0
        self._original_queue_dict = dict()
        self._lock4original = threading.Lock()

        if hasattr(self._dataloader, "data_identifier_list"):
            self.data_identifier_list = getattr(self._dataloader, "data_identifier_list")

    def reset(self):
        self._lock4original.acquire()
        super().reset()
        self._dataloader.reset()
        self._pos_read_batch_for_original = 0
        self._original_queue_dict = dict()
        self._lock4original.release()

    def _read_a_data_from_original_nonthreadsafe(self):
        d = self._dataloader(self._original_batch_size)
        self._original_queue_dict[self._pos_read_batch_for_original] = d
        self._pos_read_batch_for_original += 1
        while self._pos_read_batch_for_original >= self._num_read_batch_for_original:
            self._pos_read_batch_for_original -= self._num_read_batch_for_original

    def read_many_data(self, read_batch_id):
        self._lock4original.acquire()
        if read_batch_id not in self._original_queue_dict:
            while self._pos_read_batch_for_original != read_batch_id:
                self._read_a_data_from_original_nonthreadsafe()
            self._read_a_data_from_original_nonthreadsafe()

        a = self._original_queue_dict[read_batch_id]
        del self._original_queue_dict[read_batch_id]
        self._lock4original.release()

        return a


wrap = DataLoaderWrap


# Transform --------------------------------------------------


class DataTransform:
    def __init__(self, input_keys, input_types, input_shapes):
        self._input_keys, self._input_types, self._input_shapes = \
            canonicalize_field_name_tuple(input_keys), \
            canonicalize_field_type_tuple(input_types), \
            canonicalize_field_shape_tuple(input_shapes)
        self._output_keys, self._output_types, self._output_shapes = \
            self._input_keys, self._input_types, self._input_shapes

    def input_keys(self):
        return self._input_keys

    def input_types(self):
        return self._input_types

    def input_shapes(self):
        return self._input_shapes

    def output_keys(self):
        return self._output_keys

    def output_types(self):
        return self._output_types

    def output_shapes(self):
        return self._output_shapes

    def transform(self, *args):
        return args

    def index_in_fields(self, field_names):
        if isinstance(field_names, str):
            return self._input_keys.index(field_names)
        else:
            return [self._input_keys.index(k) for k in field_names]

    def __call__(self, *args):
        return self.transform(*args)


# Dataloader combination ---------------------------------------------------


class CountedAbsLoader:
    def __init__(self, num_samples=None):
        self._has_num_samples = num_samples is not None
        if self._has_num_samples:
            self._the_num_samples = num_samples
            self._cur_epoch = 0
            self._cur_iter = 0
            self._cur_pos = 0
            self.epoch = self._epoch
            self.iter = self._iter
            self.num_samples_finished = self._num_samples_finished
            self.num_samples = self._num_samples

    def _epoch(self):
        return self._cur_epoch

    def _iter(self):
        return self._cur_iter

    def _num_samples_finished(self):
        return self._cur_pos

    def _num_samples(self):
        return self._the_num_samples

    def _reset_count(self):
        if self._has_num_samples:
            self._cur_epoch = 0
            self._cur_iter = 0
            self._cur_pos = 0

    def _next_count(self, batch_size):
        if self._has_num_samples:
            _num_sample = self._the_num_samples
            self._cur_iter += 1
            self._cur_pos += batch_size
            if _num_sample > 0:
                while self._cur_pos >= _num_sample:
                    self._cur_epoch += 1
                    self._cur_pos -= _num_sample


class BatchConcatCache:
    def __init__(self, *args):
        self._cache = []
        for dataloader, batch_size, field_dict in args:     # mainly for syntax check
            self._cache.append((dataloader, batch_size, field_dict))

    @property
    def cache(self):
        return self._cache

    def _append_and_clone(self, other):
        assert isinstance(other, BatchConcatCache), "the other object must be BatchConcatCache"
        return BatchConcatCache(*self.cache, *other.cache)

    def __add__(self, other):
        return self._append_and_clone(other)

    def make_loader(self, indicator_field=None):
        return BatchConcatLoader(self, indicator_field=indicator_field)

    def wrap(self, original_batch_size=None, indicator_field=None, *args, **kwargs):
        return self.make_loader(indicator_field=indicator_field).rewrap(original_batch_size, *args, **kwargs)

    rewrap = wrap


class BatchConcatLoader(CountedAbsLoader):
    def __init__(self, obj, indicator_field=None):
        assert isinstance(obj, BatchConcatCache), "obj must be BatchConcatCache"

        self._indicator_field = indicator_field

        self._dataloader_info = []
        _field_names = None
        _num_samples = 0
        for dataloader, subbatch_size, field_dict in obj.cache:
            # num_samples
            if _num_samples is not None and hasattr(dataloader, "num_samples"):
                _num_samples += getattr(dataloader, "num_samples")()
            else:
                _num_samples = None
            # handle fields
            if _field_names is None:
                # the first data loader is the canonical one
                _field_names = tuple(field_dict.values())
            assert len(field_dict) == len(_field_names), "inconsistent number of fields"
            field_mapping = list([None] * len(_field_names))
            _output_keys = canonicalize_field_name_tuple(dataloader.output_keys())
            for k, v in field_dict.items():
                assert k in _output_keys, "cannot find the field"
                assert v in _field_names, "conflicting field_names"
                field_mapping[_field_names.index(v)] = _output_keys.index(k)
            if subbatch_size >= 1:
                self._dataloader_info.append((dataloader, subbatch_size, field_mapping))

        assert self._dataloader_info, "at least one data loader need to be provided"

        _first_dataloader = self._dataloader_info[0][0]
        _first_field_mapping = self._dataloader_info[0][2]
        self._field_names = _field_names
        _output_types_0 = canonicalize_field_type_tuple(_first_dataloader.output_types())
        _output_shapes_0 = canonicalize_field_shape_tuple(_first_dataloader.output_shapes())
        self._field_types = tuple(_output_types_0[i] for i in _first_field_mapping)
        if _output_shapes_0 is None:
            self._field_shapes == None
        else:
            self._field_shapes = tuple(_output_shapes_0[i] for i in _first_field_mapping)

        if self._indicator_field is not None:
            assert self._indicator_field not in self._field_names, \
                "indicator_field conflicts existing field names"
            self._field_names += (self._indicator_field,)
            self._field_types += ("int32",)
            if self._field_shapes is not None:
                self._field_shapes += ([None, 1],)

        self._data_cache = deque()

        super().__init__(_num_samples)

    def output_keys(self):
        return self._field_names

    def output_types(self):
        return self._field_types

    def output_shapes(self):
        return self._field_shapes

    def reset(self):
        self._data_cache.clear()
        for dataloader, _, _ in self._dataloader_info:
            dataloader.reset()
        self._reset_count()

    def _read_a_batch_to_cache(self):
        dataloader_id = 0
        for dataloader, sub_batch_size, field_mapping in self._dataloader_info:
            d = dataloader(sub_batch_size)
            a = tuple(d[i] for i in field_mapping)
            if self._indicator_field is not None:
                a += ([dataloader_id] * sub_batch_size,)
            self._data_cache.extend(zip(*a))
            dataloader_id += 1

    def next_batch(self, batch_size):
        while len(self._data_cache) < batch_size:
            self._read_a_batch_to_cache()
        the_batch = []
        for _ in range(batch_size):
            the_batch.append(self._data_cache.popleft())
        outputs = tuple(list(a) for a in zip(*the_batch))
        self._next_count(batch_size)
        return outputs

    def __call__(self, batch_size):
        return self.next_batch(batch_size)

    def rewrap(self, original_batch_size=None, *args, **kwargs):
        if original_batch_size is None:
            original_batch_size = sum(sub_batch_size for _, sub_batch_size, _ in self._dataloader_info)
        return wrap(self, original_batch_size, *args, **kwargs)


class FieldConcatCache:
    def __init__(self, *args):
        self._cache = []
        for dataloader, field_dict in args:     # mainly for syntax check
            self._cache.append((dataloader, field_dict))

    @property
    def cache(self):
        return self._cache

    def _append_and_clone(self, other):
        assert isinstance(other, FieldConcatCache), "the other object must be FieldConcatCache"
        return BatchConcatCache(*self.cache, *other.cache)

    def __and__(self, other):
        return self._append_and_clone(other)

    def make_loader(self):
        return FieldConcatLoader(self)

    def wrap(self, original_batch_size, *args, **kwargs):
        return self.make_loader().rewrap(original_batch_size, *args, **kwargs)

    rewrap = wrap


class FieldConcatLoader(CountedAbsLoader):

    def __init__(self, obj):
        assert isinstance(obj, FieldConcatCache), "obj must be FieldConcatCache"

        self._dataloader_info = []
        _all_field_names = []
        _all_field_types = []
        _all_field_shapes = []
        _num_samples = None
        for dataloader, field_dict in obj.cache:
            # num_samples
            if hasattr(dataloader, "num_samples"):
                _the_num_samples = getattr(dataloader, "num_samples")()
                if _num_samples is None:
                    _num_samples = _the_num_samples
                else:
                    if _num_samples != _the_num_samples:
                        _num_samples = max(_num_samples, _the_num_samples)
                        print("WARNING: the field-concated data loader have different number of samples")

            # handle fields
            _output_keys = canonicalize_field_name_tuple(dataloader.output_keys())
            _output_types = canonicalize_field_type_tuple(dataloader.output_types())
            _output_shapes = canonicalize_field_shape_tuple(dataloader.output_shapes())

            field_mapping = []
            for k, v in field_dict.items():
                assert k in _output_keys, "cannot find the field"
                assert v not in _all_field_names, "conflicting field name"

                i = _output_keys.index(k)
                field_mapping.append(i)
                _all_field_types.append(_output_types[i])
                _all_field_shapes.append(_output_shapes[i])

                _all_field_names.append(v)

            self._dataloader_info.append((dataloader, field_mapping))

        self._field_names = _all_field_names
        self._field_types = _all_field_types
        self._field_shapes = _all_field_shapes

        super().__init__(_num_samples)

    def output_keys(self):
        return self._field_names

    def output_types(self):
        return self._field_types

    def output_shapes(self):
        return self._field_shapes

    def reset(self):
        for dataloader, _ in self._dataloader_info:
            dataloader.reset()
        self._reset_count()

    def next_batch(self, batch_size):
        the_batch = []
        for dataloader, field_mapping in self._dataloader_info:
            the_subbatch = dataloader(batch_size)
            for i in field_mapping:
                the_batch.append(the_subbatch[i])
        outputs = tuple(the_batch)
        self._next_count(batch_size)
        return outputs

    def __call__(self, batch_size):
        return self.next_batch(batch_size)

    def rewrap(self, original_batch_size, *args, **kwargs):
        return wrap(self, original_batch_size, *args, **kwargs)
