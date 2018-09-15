from zdata.dataloader.generic_multithread import Net as BaseNet
import os

import re

import numpy as np

import threading

import cv2

import pickle, json, scipy.io

from glob import glob

from zutils.vid_seg import VideoSegReader
from zutils.option_struct import OptionDef

import math

from numba import jit


# use mxnet imread if possible ------------
import importlib.util
mxnet_spec = importlib.util.find_spec("mxnet")
if mxnet_spec is None:
    _imread = cv2.imread
else:
    import mxnet as mx
    def _imread(*args, **kwargs):
        return mx.image.imread(*args, **kwargs).asnumpy()
# -----------------------------------------


class NetOptionDef(OptionDef):

    @staticmethod
    def image_loader(p):
        p["shuffle"] = False        # means if to used the same random seed
        p["is_sequential"] = False  # means if shuffle
        p["num_chosen_images"] = None    # when is None, use all
        p["ratio_of_chosen_images"] = None  # similar to num_of_chosen_images, only one of them can be set

        assert p["num_chosen_images"] is None or p["ratio_of_chosen_images"] is None, \
            "Only one of num_of_chosen_images and ratio_of_chosen_images can be specified"
        if p["ratio_of_chosen_images"] is not None:
            assert 0 <= p["ratio_of_chosen_images"] <= 1, \
                "ratio_of_chosen_images must be in [0, 1]"


class Net(BaseNet):

    # user implemented interface -----------------------------

    def default_root_dir(self):
        raise ValueError("No default_root_dir() is defined")
        return None

    def field_folders(self):
        return None

    def data_postprocessing(self, data, field_names):
        return data

    def read_many_auxiliary_data(self, image_ids):
        return None

    MyOptionDef = NetOptionDef

    # --------------------------------------------------------

    def _get_num_fields(self):
        if self._field_folders is None:
            return 1
        else:
            return len(self._field_folders)

    def _all_field_names(self):
        if self._field_folders is None:
            the_field_folders = ["data"]
        else:
            the_field_folders = list(self._field_folders)
        all_field_names = (
            the_field_folders +
            list(self._aux_list)
        )
        return tuple(all_field_names)

    def _get_image_subpath_from_field_subpath(self, field_path, field_id):
        if self._field_folders is None:
            assert field_id == 0, "field_id must be 0"
            return field_path
        else:
            if isinstance(field_id, str):
                field_folder = field_id
            else:
                field_folder = self._field_folders[field_id]
            subfolder_subpath, vid_fn = os.path.split(field_path)
            subfolder_name, _ = os.path.split(subfolder_subpath)
            return os.path.join(subfolder_name, field_folder, vid_fn)

    def image_paths(self, image_id):
        vid_fn = self._image_list[image_id]
        p = []
        for i in range(1, self._get_num_fields()):
            field_vid_fn = self._get_image_subpath_from_field_subpath(vid_fn, i)
            p.append(os.path.join(self._root_dir, field_vid_fn))
        return tuple(p)

    def __init__(
            self, subset_name='train', options=None, aux_list=None, **kwargs
    ):

        # root dir
        if "root_dir" in kwargs and kwargs["root_dir"]:
            self._root_dir = kwargs["root_dir"]
            del kwargs["root_dir"]
        else:
            self._root_dir = self.default_root_dir()
        while self._root_dir[-1] == '/':
            del self._root_dir[-1]

        assert os.path.isdir(self._root_dir), "The root_dir does not exist: %s" % self._root_dir

        self._field_folders = self.field_folders()
        if isinstance(self._field_folders, str):
            self._field_folders = (self._field_folders,)
        if self._field_folders is not None and not self._field_folders:
            self._field_folders = None

        # options
        self.opt = NetOptionDef(options, finalize_check=None)["image_loader"].get_namedtuple()

        # get data list
        self._subset_name = subset_name
        subset_list_fn = os.path.join(self._root_dir, subset_name + ".list")

        assert os.path.exists(subset_list_fn), "the subset name does not exist"
        with open(subset_list_fn, 'r') as f:
            identifier_list = f.readlines()
        identifier_list = [s.strip() for s in identifier_list]
        if self._field_folders is not None:
            first_field_folder = self._field_folders[0]
            file_list = []
            for s in identifier_list:
                subfolder, fn = os.path.split(s)
                file_list.append(
                    os.path.join(subfolder, first_field_folder, fn)
                )
        else:
            file_list = identifier_list

        if self.opt.ratio_of_chosen_images is not None:
            _num_chosen_images = math.ceil(self.opt.ratio_of_chosen_images) * self.opt.ratio_of_chosen_images
        elif self.opt.num_chosen_images is not None:
            _num_chosen_images = self.opt.num_chosen_images
        else:
            _num_chosen_images = len(file_list)

        _total_images = len(file_list)
        if _total_images > _num_chosen_images:
            self._chosen_id_to_image_id = [
                round((i+0.5) * (_total_images / _num_chosen_images)) for i in
                range(0, _num_chosen_images)
            ]
        else:
            self._chosen_id_to_image_id = list(
                range(_total_images)
            )

        self._image_list = file_list

        # handle shuffle/non-shuffled cases
        self._num_samples = len(self._chosen_id_to_image_id)

        # estimated sample num per read
        self._estimated_num_samples_per_read = 1

        # aux data (should be saved in batch)
        if aux_list is None:
            aux_list = []
        if aux_list:
            assert self._field_folders is not None, "aux_list is not supported if field_folders is None"
        for a in aux_list:
            aux_load_func_name = "read_aux_%s" % a
            assert hasattr(self, aux_load_func_name), "%s must exist to load the aux data"
        self._aux_list = aux_list
        self._aux_data = dict()
        self._aux_lock = dict()
        for aux_name in self._aux_list:
            self._aux_lock[aux_name] = threading.Lock()

        # init super
        super_kwargs = {
            "is_random": not self.opt.is_sequential,
            "num_read_batch": self._num_samples,
            **kwargs
        }
        if "rand_seed" not in super_kwargs:
            super_kwargs["rand_seed"] = None if self.opt.shuffle else 12345
        super().__init__(**super_kwargs)

        # set field keys
        self.mutable_all_fields.extend(self._all_field_names())

    def num_samples(self):
        return self._num_samples

    @property
    def subset_name(self):
        return self._subset_name

    _img_ext_list = [
        'png', 'PNG', "jpg", 'JPG', "jpeg", 'JPEG'
    ]
    _general_ext_list = [
        "pkl", 'p', 'json', 'mat'
    ]

    # @jit(nonpython=True)
    @classmethod
    def _complete_image_ext(cls, bare_fn):
        bare_fn = re.sub(r':[0-9]+$',r'', bare_fn)   # remove postfix
        ext_list = cls._img_ext_list + cls._general_ext_list
        for ext in ext_list:
            fn = bare_fn + "." + ext
            if os.path.exists(fn):
                return fn
        raise FileExistsError("No image file is found")

    # @jit(nonpython=True)
    @classmethod
    def _load_file(cls, fn):
        _, ext = os.path.splitext(fn)
        ext = ext[1:]
        if ext in cls._img_ext_list:
            im = _imread(fn)
            if len(im.shape) == 2:
                im = np.reshape(im, im.shape + (1,))
            elif im.shape[2] == 3:
                im = cv2.cvtColor(im, code=cv2.COLOR_BGR2RGB)
            data = im
        elif ext in ('pkl', 'p'):
            with open(fn, 'rb') as f:
                data = pickle.load(f)
        elif ext == "json":
            with open(fn, 'r') as f:
                data = json.load(f)
        elif ext == "mat":
            data = scipy.io.loadmat(fn)
        else:
            raise ValueError("Unrecognized extension")
        return data

    def chosen_id_to_image_id(self, chosen_id):
        return self._chosen_id_to_image_id[chosen_id]

    def read_many_data_identifier(self, read_batch_id):
        return [self._image_list[self.chosen_id_to_image_id(read_batch_id)]]

    @jit(nonpython=True)
    def read_many_data(self, read_batch_id):

        image_id = self.chosen_id_to_image_id(read_batch_id)

        outputs = []

        # read all fields
        _num_fields = self._get_num_fields()
        image_fn = self._image_list[image_id]
        for i in range(_num_fields):
            image_fn_i = (os.path.join(
                self._root_dir,
                self._get_image_subpath_from_field_subpath(image_fn, i)
            ))
            image_fn_i = self._complete_image_ext(image_fn_i)
            data = self._load_file(image_fn_i)
            outputs.append([data])

        # regular aux data, require .read_aux_*(aux_path)
        for a in self._aux_list:
            the_aux = self._read_aux(a, image_id)
            outputs.append([the_aux])

        # custom auxiliary data
        custom_aux = self.read_many_auxiliary_data([image_id])
        if custom_aux is not None:
            outputs.extend(custom_aux)

        outputs = tuple(outputs)

        outputs_pp = self.data_postprocessing(outputs, self._all_field_names())
        if outputs_pp is not None:
            outputs = tuple(outputs_pp)

        return outputs

    def _read_aux(self, aux_name, image_id):
        if aux_name not in self._aux_data:
            the_aux_lock = self._aux_lock[aux_name]
            the_aux_lock.acquire()
            if aux_name not in self._aux_data:
                get_aux_func = getattr(self, "read_aux_%s" % aux_name)
                d = get_aux_func(self.subset_name)
                assert d.shape[0] == len(self._image_list), "inconsistent number of images and aux"
                self._aux_data[aux_name] = d
            the_aux_lock.release()
        return self._aux_data[aux_name][image_id]

    def estimated_num_samples_per_read_batch(self):
        return self._estimated_num_samples_per_read
