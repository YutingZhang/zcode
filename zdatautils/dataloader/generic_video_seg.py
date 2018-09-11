from zdatautils.dataloader.generic_multithread import Net as BaseNet
import os

from glob import glob

from zutils.vid_seg import VideoSegReader
from zutils.option_struct import OptionDef

import threading

import math

from zutils.py_utils import IntervalSearch


class NetOptionDef(OptionDef):

    @staticmethod
    def video_seg_loader(p):
        p["shuffle"] = False        # means if to used the same random seed
        p["is_sequential"] = False  # means if shuffle
        p["num_consecutive_frames"] = 1
        p["consecutive_interval"] = 1

        p["num_videos_per_subfolder"] = None   # use all
        p["num_startable_frames_per_video"] = None   # None means infinity
        p["startable_frame_interval"] = None
        # only one of num_startable_frames_per_video and startable_frame_interval can be not None

        assert p["num_consecutive_frames"] >= 1, "num_consecutive_frames must be >= 1"
        assert p["consecutive_interval"] >= 1, "consecutive_interval must be >=1"
        assert p["num_startable_frames_per_video"] is None or p["num_startable_frames_per_video"] >= 1, \
            "num_startable_frames_per_video should either be None or a number greater than 1"
        assert p["startable_frame_interval"] is None or p["startable_frame_interval"] >= 1, \
            "num_startable_frames_per_video should either be None or a number greater than 1"

        if p["num_consecutive_frames"] == 1:
            # randomly get data
            if p["consecutive_interval"] != 1:
                print("WARNING: consecutive_interval is forced to 1 because num_consecutive_frames == 1")
                p.set("consecutive_interval", 1)

        # sequentially get data
        if p["num_startable_frames_per_video"] is None:
            if p["startable_frame_interval"] is None:
                p.set("startable_frame_interval", 1)
        else:
            assert p["startable_frame_interval"] is None, \
                "num_startable_frames_per_video and startable_frame_interval cannot be set togehter"
        # remark: there are two mode
        # 1) when num_startable_frames_per_video is *not* None, then uniformly get the frames
        # 2) when num_startable_frames_per_video is None, start from the first frame and get frame using
        #       the interval defined by consecutive_interval


class Net(BaseNet):

    # user implemented interface -----------------------------

    def default_root_dir(self):
        raise ValueError("No default_root_dir() is defined")
        return None

    def field_folders(self):
        return None

    def subset_to_subfolders(self, subset_name):
        raise ValueError(".subset(subset_name) need to be defined")
        return []   # should return the subfolder list

    def data_postprocessing(self, data, field_names):
        return data

    def read_many_auxiliary_data(self, video_id, frame_ids):
        return None

    def translate_subfolder_name(self, a):
        return a

    def get_video_reader(self, fn):
        return VideoSegReader(fn)

    def valid_filename_pattern(self):
        return "*"

    MyOptionDef = NetOptionDef

    # --------------------------------------------------------

    def _translate_subfolder_names(self, l):
        return [self.translate_subfolder_name(a) for a in l]

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
            (["has_next_frame"] if self._need_has_next_frame else []) +
            the_field_folders +
            list(self._aux_list)
        )
        return tuple(all_field_names)

    def _get_video_subfolder_subpath(self, subfolder_name, field_id):
        if self._field_folders is None:
            assert field_id == 0, "field_id must be 0"
            return os.path.join(subfolder_name)
        else:
            if isinstance(field_id, str):
                field_folder = field_id
            else:
                field_folder = self._field_folders[field_id]
            return os.path.join(subfolder_name, field_folder)

    def _get_video_subpath_from_field_subpath(self, field_path, field_id):
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

    def video_paths(self, video_id):
        vid_fn = self._video_list[video_id]
        p = []
        for i in range(1, self._get_num_fields()):
            field_vid_fn = self._get_video_subpath_from_field_subpath(vid_fn, i)
            p.append(os.path.join(self._root_dir, field_vid_fn))
        return tuple(p)

    def __init__(
            self, subset_name='train', options=None,
            need_has_next_frame=False, aux_list=None,
            **kwargs
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

        self._need_has_next_frame = need_has_next_frame

        # options
        self.opt = NetOptionDef(options, finalize_check=None)["video_seg_loader"].get_namedtuple()

        # use subset data list if exists
        subset_list_fn = os.path.join(self._root_dir, subset_name + ".list")

        if os.path.exists(subset_list_fn):
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

            self._video_list = file_list

            if self.opt.num_videos_per_subfolder is not None:
                raise NotImplementedError("cannot specify num videos per folder if subset video list is present")

        else:
            # subset
            self._subfolders = self._translate_subfolder_names(self.subset_to_subfolders(subset_name))

            # get video list and meta
            _root_dir_len = len(self._root_dir)
            self._video_list = []
            for sf in self._subfolders:
                first_field_subpath = self._get_video_subfolder_subpath(sf, 0)

                fn_pattern = self.valid_filename_pattern()
                if isinstance(fn_pattern, str):
                    fn_pattern = [fn_pattern]
                subj_video_list = []
                for fp in fn_pattern:
                    sub_subj_video_list = glob(
                        os.path.join(self._root_dir, first_field_subpath, fp)
                    )
                    sub_subj_video_list = [p[_root_dir_len+1:] for p in sub_subj_video_list]
                    subj_video_list.extend(sub_subj_video_list)
                subj_video_list = sorted(list(set(subj_video_list)))

                if self.opt.num_videos_per_subfolder is not None and \
                        self.opt.num_videos_per_subfolder < len(subj_video_list):
                    video_interval = len(subj_video_list) / self.opt.num_videos_per_subfolder
                    subj_video_list = [
                        subj_video_list[round((i+0.5)*video_interval)]
                        for i in range(self.opt.num_videos_per_subfolder)
                    ]

                self._video_list.extend(subj_video_list)

        # get and calculate video meta
        self._frame_counts = []         # total frame count
        self._valid_frame_counts = []   # locations that can start a clip without going out of the frame-count bound
        self._startable_frame_counts = []   # locations that are allowed to start a clip
        self._read_batch_counts = []    # total number of read batch per video
        self._read_batch_sizes = []     # read batch size per video

        num_tailing_frames = self.opt.consecutive_interval * (self.opt.num_consecutive_frames-1)

        for vid_fn in self._video_list:
            # get frame counts and seg_len
            vreader = self.get_video_reader(os.path.join(self._root_dir, vid_fn))
            the_frame_count = vreader.frame_count
            the_seg_len = vreader.seg_len
            del vreader
            self._frame_counts.append(the_frame_count)

            # check the frame counts of the other fields
            for i in range(1, self._get_num_fields()):
                field_vid_fn = self._get_video_subpath_from_field_subpath(vid_fn, i)
                vreader = self.get_video_reader(os.path.join(self._root_dir, field_vid_fn))
                assert vreader.frame_count == the_frame_count, "inconsistent frame counts among fields"
                del vreader

            # valid frame counts
            the_valid_frame_count = (the_frame_count - num_tailing_frames)
            self._valid_frame_counts.append(the_valid_frame_count)

            # startable frame counts
            if self.opt.num_startable_frames_per_video is None:
                the_startable_frame_count = math.ceil(the_valid_frame_count / self.opt.startable_frame_interval)
                the_startable_frame_interval = self.opt.startable_frame_interval
            else:
                if self.opt.num_startable_frames_per_video >= the_valid_frame_count:
                    the_startable_frame_count = the_valid_frame_count
                    the_startable_frame_interval = 1
                else:
                    the_startable_frame_count = self.opt.num_startable_frames_per_video
                    the_startable_frame_interval = the_valid_frame_count // self.opt.num_startable_frames_per_video
            self._startable_frame_counts.append(the_startable_frame_count)

            # read_batch sizes and counts
            if self.opt.is_sequential:
                the_read_batch_size = math.ceil(min(the_seg_len * 2, 500) / the_startable_frame_interval)
            else:
                the_read_batch_size = 1

            the_read_batch_count = math.ceil(the_startable_frame_count / the_read_batch_size)
            self._read_batch_counts.append(the_read_batch_count)
            self._read_batch_sizes.append(the_read_batch_size)

        # build indexer
        self._read_batch_count_cumsum = []
        cur_read_batch_count_sum = 0
        for c in self._read_batch_counts:
            cur_read_batch_count_sum += c
            self._read_batch_count_cumsum.append(cur_read_batch_count_sum)
        self._sampleId_to_vidId = IntervalSearch(self._read_batch_count_cumsum[:-1])

        # handle shuffle/non-shuffled cases
        self._num_samples = sum(self._startable_frame_counts) * self.opt.num_consecutive_frames

        # estimated sample num per read
        self._estimated_num_samples_per_read = \
            sum(self._read_batch_sizes) / len(self._read_batch_sizes) * self.opt.num_consecutive_frames

        # frame-wise aux data
        if aux_list is None:
            aux_list = []
        if aux_list:
            assert self._field_folders is not None, "aux_list is not supported if field_folders is None"
        for a in aux_list:
            aux_load_func_name = "read_aux_%s" % a
            assert hasattr(self, aux_load_func_name), "%s must exist to load the aux data"
        self._aux_list = aux_list
        self._aux_data = []
        self._aux_lock = []
        for i in range(len(self._video_list)):
            self._aux_data.append(dict())
            the_aux_lock_dict = dict()
            for aux_name in self._aux_list:
                the_aux_lock_dict[aux_name] = threading.Lock()
            self._aux_lock.append(the_aux_lock_dict)

        # init super
        super_kwargs = {
            "is_random": not self.opt.is_sequential,
            "num_read_batch": self._read_batch_count_cumsum[-1],
            "pool_size": 4,
            **kwargs
        }
        if "rand_seed" not in super_kwargs:
            super_kwargs["rand_seed"] = None if self.opt.shuffle else 12345
        super().__init__(**super_kwargs)

        # set field keys
        self.mutable_all_fields.extend(self._all_field_names())
        self.mutable_field_shape_mapping["has_next_frame"] = (None, 1)
        self.mutable_field_type_mapping["has_next_frame"] = 'bool'

    def num_samples(self):
        return self._num_samples

    def reset_data(self):
        # self._aux_data.clear()  # not sure if we should reset the meta since it is usually small
        super().reset_data()

    def _read_many_data_identifier(self, read_batch_id):
        vid_id, batch_id_per_vid = self._sampleId_to_vidId[read_batch_id]

        # compute the full list of the startable frames
        the_valid_frame_count = self._valid_frame_counts[vid_id]
        if self.opt.num_startable_frames_per_video is None:
            startable_frames = list(range(0, the_valid_frame_count, self.opt.startable_frame_interval))
        else:
            if self.opt.num_startable_frames_per_video >= the_valid_frame_count:
                startable_frames = list(range(0, the_valid_frame_count))
            else:
                startable_frame_stride = the_valid_frame_count / (self.opt.num_startable_frames_per_video + 1)
                startable_frames = list(
                    round((i+0.5)*startable_frame_stride) for i in
                    range(0, self.opt.num_startable_frames_per_video)
                )

        # get all frame ids
        avg_read_batch_size = self._read_batch_sizes[vid_id]
        frame_ids = []
        start_frame_id_for_the_batch = batch_id_per_vid * avg_read_batch_size
        the_read_batch_size = min(len(startable_frames)-start_frame_id_for_the_batch, avg_read_batch_size)
        for j in range(the_read_batch_size):
            the_startable_frame = startable_frames[start_frame_id_for_the_batch + j]
            frame_ids.extend(
                the_startable_frame + i*self.opt.consecutive_interval
                for i in range(self.opt.num_consecutive_frames)
            )
        return vid_id, frame_ids, the_read_batch_size

    def read_many_data_identifier(self, read_batch_id):
        vid_id, frame_ids, _ = self._read_many_data_identifier(read_batch_id)
        vid_fn = self._video_list[vid_id]
        vid_fn = self._get_video_subpath_from_field_subpath(vid_fn, "")
        return list(vid_fn + ":" + str(i) for i in frame_ids)

    def read_many_data(self, read_batch_id):

        vid_id, frame_ids, the_read_batch_size = self._read_many_data_identifier(read_batch_id)

        # unique frame ids
        sorted_uniq_frame_ids = sorted(list(set(frame_ids)))

        # read all fields
        _num_fields = self._get_num_fields()
        vid_fn = self._video_list[vid_id]
        frame_dict = dict()
        for fr_id in sorted_uniq_frame_ids:
            frame_dict[fr_id] = [None] * _num_fields
        for i in range(_num_fields):
            vreader = self.get_video_reader(os.path.join(
                self._root_dir,
                self._get_video_subpath_from_field_subpath(vid_fn, i)
            ))
            for fr_id in sorted_uniq_frame_ids:
                frame_dict[fr_id][i] = vreader.read_at(fr_id)
            vreader.release()

        # regular aux data, require .read_aux_*(aux_path)
        uniq_aux_outputs = []
        for a in self._aux_list:
            the_aux = self._read_aux(a, vid_id)
            uniq_aux_outputs.append([the_aux[fr_id] for fr_id in sorted_uniq_frame_ids])

        # custom auxiliary data
        uniq_aux_outputs_2 = self.read_many_auxiliary_data(vid_id, sorted_uniq_frame_ids)
        if uniq_aux_outputs_2 is not None:
            uniq_aux_outputs.extend(uniq_aux_outputs_2)

        uniq_aux_outputs = tuple(uniq_aux_outputs)

        # rearrange aux_outputs
        if uniq_aux_outputs is not None and uniq_aux_outputs:
            if isinstance(uniq_aux_outputs, tuple):
                for fr_id, d in zip(sorted_uniq_frame_ids, zip(*uniq_aux_outputs)):
                    frame_dict[fr_id].extend(d)
            else:
                for fr_id, d in zip(sorted_uniq_frame_ids, uniq_aux_outputs):
                    frame_dict[fr_id].append(d)

        # re-arrange data
        frames = []
        for fr_id in frame_ids:
            frames.append(tuple(frame_dict[fr_id]))

        outputs = list(zip(*frames))
        outputs = [list(a) for a in outputs]

        if self._need_has_next_frame:
            has_pre_frame = ([True]*(self.opt.num_consecutive_frames-1) + [False]) * the_read_batch_size
            outputs = [has_pre_frame] + outputs

        outputs = tuple(outputs)

        outputs_pp = self.data_postprocessing(outputs, self._all_field_names())
        if outputs_pp is not None:
            outputs = tuple(outputs_pp)

        return outputs

    def _read_aux(self, aux_name, video_id):
        the_aux_dict = self._aux_data[video_id]
        if aux_name not in the_aux_dict:
            the_aux_lock = self._aux_lock[video_id][aux_name]
            the_aux_lock.acquire()
            if aux_name not in the_aux_dict:
                vid_subpath = self._video_list[video_id]
                aux_path = os.path.join(self._root_dir, self._get_video_subpath_from_field_subpath(vid_subpath, 'Aux'))
                get_aux_func = getattr(self, "read_aux_%s" % aux_name)
                d = get_aux_func(aux_path)
                assert d.shape[0] == self._frame_counts[video_id], "inconsistent number of frames"
                the_aux_dict[aux_name] = d
            the_aux_lock.release()
        return the_aux_dict[aux_name]

    def estimated_num_samples_per_read_batch(self):
        return self._estimated_num_samples_per_read
