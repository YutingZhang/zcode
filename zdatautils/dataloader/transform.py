from zdatautils.dataloader.generic_multithread import DataTransform
from zdatautils.dataloader.utils import *
import dataloader._image_transform as _image_transform
import numpy as np
from copy import copy, deepcopy
from collections import Iterable, Sequence, Callable, deque
import cv2
import math
import random
import threading


class Im2Float(DataTransform):

    def __init__(self, im_fields, *args, **kwargs):

        super().__init__(*args, **kwargs)

        im_fields = canonicalize_field_name_tuple(im_fields)
        self._field_mapping = self.index_in_fields(im_fields)
        _output_types = list(self.input_types())
        for k in self._field_mapping:
            _output_types[k] = "float32"
        self._output_types = tuple(_output_types)

    def transform(self, *args):
        a = list(args)
        for k in self._field_mapping:
            b = a[k]
            if isinstance(b, np.ndarray):
                b = b.astype(np.float32)
                b /= 255
            else:
                b = copy(b)
                for i in range(len(b)):
                    c = np.array(b[i]).astype(np.float32)
                    c /= 255
                    b[i] = c
            a[k] = b
        return tuple(a)


class Slice(DataTransform):

    def __init__(self, im_fields, cropped_shape, *args, **kwargs):

        super().__init__(*args, **kwargs)

        im_fields = canonicalize_field_name_tuple(im_fields)
        self._field_mapping = self.index_in_fields(im_fields)
        _crop_tuple = []
        for s in cropped_shape:
            if s is None:
                _crop_tuple.append(slice(None, None))
            elif isinstance(s, Iterable):
                s = tuple(s)
                _crop_tuple.append(slice(*s))
            else:
                _crop_tuple.append(slice(s))
        self._crop_tuple = tuple(_crop_tuple)

    def transform(self, *args):
        if not self._crop_tuple:
            return tuple(args)

        a = list(args)
        for k in self._field_mapping:
            b = a[k]
            if isinstance(b, np.ndarray):
                b = b[:, self._crop_tuple]
            else:
                b = copy(b)
                for i in range(len(b)):
                    b[i] = np.array(b[i])[self._crop_tuple]
            a[k] = b
        return tuple(a)


def has_next_frame_field_to_idx(has_next_frame_field, input_keys):
    if has_next_frame_field is None:
        if "has_next_frame" in input_keys:
            has_next_frame_field = "has_next_frame"

    if has_next_frame_field is None or has_next_frame_field == "__None__":
        _has_next_frame_field_idx = None
    elif has_next_frame_field == "__ALL__":
        _has_next_frame_field_idx = "__ALL__"
    else:
        _has_next_frame_field_idx = input_keys.index(has_next_frame_field)
    return _has_next_frame_field_idx


def get_has_next_frame_field(has_next_frame_field_idx, inputs):
    n = len(inputs[0])
    if has_next_frame_field_idx is None:
        has_next_frame = [False] * n
    elif has_next_frame_field_idx == "__ALL__":
        has_next_frame = [True] * (n - 1) + [False]
    else:
        has_next_frame_field_idx = int(has_next_frame_field_idx)
        has_next_frame = np.array(inputs[has_next_frame_field_idx]).reshape([-1]).tolist()
        assert len(has_next_frame) == n, "has_next_frame and inconsistent batch size"
    return has_next_frame


class FarnebackFlow(DataTransform):

    def __init__(self, im_field, flow_field=None, has_next_frame_field=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        _input_keys = self.input_keys()

        self._has_next_frame_field_idx = has_next_frame_field_to_idx(has_next_frame_field, _input_keys)

        self._im_field_idx = _input_keys.index(im_field)

        if flow_field is None:
            flow_field = im_field + "/flow"
        assert flow_field not in _input_keys, "flow field name conflicts existing field names"

        self._output_keys = \
            _input_keys[:self._im_field_idx+1] + (flow_field,) + _input_keys[self._im_field_idx+1:]
        self._output_types = \
            self.input_types()[:self._im_field_idx+1] + ("float32",) + self.input_types()[self._im_field_idx+1:]
        if self.input_shapes() is not None:
            flow_shape = self.input_shapes()[self._im_field_idx]
            flow_shape = list(flow_shape)
            flow_shape[3] = 2
            self._output_shapes = \
                self.input_shapes()[:self._im_field_idx+1] + (flow_shape,) + self.input_shapes()[self._im_field_idx+1:]

    def transform(self, *args):

        im = args[self._im_field_idx]

        has_next_frame = get_has_next_frame_field(self._has_next_frame_field_idx, args)

        _im_gray = list([None] * len(im))

        def im_gray(j):
            if _im_gray[j] is None:
                im_j = im[j]
                im_s = im_j.shape
                if len(im_s) == 2:
                    im_gr = im_j
                elif im_s[2] == 3:
                    im_gr = cv2.cvtColor(im_j, cv2.COLOR_RGB2GRAY)
                elif im_s[2] == 4:
                    im_gr = cv2.cvtColor(im_j, cv2.COLOR_RGBA2GRAY)
                elif im_s[2] == 1:
                    im_gr = np.squeeze(im_j, axis=-1)
                else:
                    raise ValueError("cannot convert image to gray")
                if im_gr.dtype in (np.float16, np.float32, np.float64, np.float128):
                    im_gr = (im_gr * 255).astype(np.uint8)
                _im_gray[j] = im_gr
            return _im_gray[j]

        flow = []
        for k in range(len(im)):
            x = np.array(im[k])
            if has_next_frame[k]:
                x1 = im_gray(k)
                x2 = im_gray(k+1)

                pyr_scale = 0.5
                num_scales = math.floor(math.log(math.sqrt(x1.shape[0] * x1.shape[1]) / 16, 1/pyr_scale))
                num_scales = max(num_scales, 0) + 1

                f = cv2.calcOpticalFlowFarneback(
                    x1, x2, None, pyr_scale, num_scales, 15, 3, 5, 1.2, 0
                )
                f = f.astype(np.float32)
            else:
                f = np.zeros(tuple(x.shape[:2]) + (2,), np.float32)
            flow.append(f)

        outputs = tuple(args[:self._im_field_idx+1]) + (flow,) + tuple(args[self._im_field_idx+1:])
        return outputs


class ImAffine(DataTransform):

    def __init__(
            self,
            transform,          # can be a field name or a single specified transform
            output_wh,          # output size in pixel (height, width)
            fields_type_dict,   # mapping for field name to types
            # REMARK on types:
            #   Image: "im";
            #   Point: "xy", "yx"
            #   Box: "box:yxhw", "box:xywh", "box:yxyx", "box:xyxy"
            #   Flow: "flow:xy", "flow:yx"
            transform_type="affine",      # valid values "affine", "box", "topleft-box" (scale)
            output_postfix=None,          # None means to replace original field
            has_next_frame_field=None,  # use to figure out if has valid optical flow
            *args, **kwargs
    ):

        super().__init__(*args, **kwargs)

        if isinstance(output_wh, Sequence):
            assert len(output_wh) == 2, "length of output_wh should be exactly two"
            output_wh = tuple(output_wh)
        else:
            output_wh = (output_wh, output_wh)
        self._output_wh = output_wh

        assert isinstance(fields_type_dict, dict), "fields_type_dict must be a dict (better to be an OrderedDict)"
        self._chosen_fields_idx = self.index_in_fields(fields_type_dict.keys())
        self._chosen_field_type = list(fields_type_dict.values())
        _use_new_output_shapes = []
        for a in self._chosen_field_type:
            assert isinstance(a, str), "field_type must be a str"
            a1 = a[:2] if a.endswith(":n") else a
            assert a1 in (
                "im", "xy", "yx",
                "box", "box:yxhw", "box:xywh", "box:yxyx", "box:xyxy",
                "flow:xy", "flow:yx"
            ), "unrecognized field type"
            assert a != "im:n", "normalized mode cannot apply to image"
            if a1 == "im" or a1.startswith("flow:"):
                _use_new_output_shapes.append(True)
            else:
                _use_new_output_shapes.append(False)

        assert isinstance(transform_type, str), "transform_type must be a str"
        transform_type_bare = transform_type[:-2] if transform_type.endswith(":n") else transform_type
        assert transform_type_bare in (
            "affine",
            "box", "box:yxhw", "box:xywh", "box:yxyx",
            "topleft-box", "topleft-box:wh", "topleft-box:hw"
        ), "unrecognized transform_type"
        self._transform_type = transform_type

        assert output_postfix is None or isinstance(output_postfix, str), \
            "wrong output_postfix"
        self._output_postfix = output_postfix

        if isinstance(transform, str):
            self._transform_field_idx = self.index_in_fields(transform)
        else:
            self._transform_field_idx = None
            self._transform_content = transform

        # handle has_next_frame_field for potential cropping of optical flow
        self._has_next_frame_field_idx = has_next_frame_field_to_idx(has_next_frame_field, self.input_keys())

        # handle field meta
        _input_shapes = self.input_shapes()
        _new_output_shapes = []
        for j in range(len(self._chosen_fields_idx)):
            _in_s = _input_shapes[self._chosen_fields_idx[j]]
            _out_s = list(_in_s)
            if _use_new_output_shapes[j]:
                if _in_s is not None:
                    _out_s[-2] = self._output_wh[0]     # w
                    _out_s[-3] = self._output_wh[1]     # h
            _new_output_shapes.append(_out_s)

        if self._output_postfix is not None:
            self._output_keys = \
                self.input_keys() + \
                tuple(self.input_keys()[i] + self._output_postfix for i in self._chosen_fields_idx)
            self._output_types = \
                self.input_types() + \
                tuple(self.input_types()[i] for i in self._chosen_fields_idx)
            if self.input_shapes() is not None:
                self._output_shapes = _input_shapes + tuple(_new_output_shapes)
        else:
            if self.input_shapes() is not None:
                _output_shapes = list(_input_shapes)
                for j in range(len(self._chosen_fields_idx)):
                    _output_shapes[self._chosen_fields_idx[j]] = _new_output_shapes[j]
                self._output_shapes = tuple(_output_shapes)

    def transform(self, *args):

        outputs = list(args)
        im_wh = None

        a_transform = None if self._transform_field_idx is None else args[self._transform_field_idx]

        def _get_im_wh(_j):
            if im_wh is None:
                return None
            return im_wh[_j]

        def _get_transform(_j):
            if a_transform is None:
                _t = self._transform_content
            else:
                _t = a_transform[_j]
            return self._transform_to_affine(_t, im_wh=_get_im_wh(_j))

        for i, ttype in zip(self._chosen_fields_idx, self._chosen_field_type):
            is_normalized = ttype.endswith(":n")
            if is_normalized:
                ttype = ttype[:2]
            x_array = args[i]
            n = len(x_array)
            y_array = []
            if ttype == "im":
                assert not is_normalized, "normalized mode cannot be used to im"
                im_wh = []
                for j in range(n):
                    x = np.array(x_array[j])
                    im_wh.append((x.shape[1], x.shape[0]))
                    t = _get_transform(j)
                    y = _image_transform.affine_im(x, t, output_wh=self._output_wh)
                    y_array.append(y)
            elif ttype in ("xy", "yx"):
                dim_order = ttype
                for j in range(n):
                    x = x_array[j]
                    t = _get_transform(j)
                    y = _image_transform.affine_point(
                        x, t, output_wh=self._output_wh,
                        dim_order=dim_order,
                        is_normalized=is_normalized,
                        im_wh=_get_im_wh(j)
                    )
                    y_array.append(y)
            elif ttype.startswith("box:"):
                dim_order = ttype[4:]
                for j in range(n):
                    x = x_array[j]
                    t = _get_transform(j)
                    y = _image_transform.affine_box(
                        x, t, output_wh=self._output_wh,
                        dim_order=dim_order,
                        is_normalized=is_normalized,
                        im_wh=_get_im_wh(j)
                    )
                    y_array.append(y)
            elif ttype.startswith("flow:"):

                has_next_frame = get_has_next_frame_field(self._has_next_frame_field_idx, args)

                im_wh = []
                for j in range(n):
                    x = x_array[j]
                    im_wh.append((x.shape[1], x.shape[0]))

                dim_order = ttype[5:]
                for j in range(n):
                    x = x_array[j]
                    if has_next_frame[j]:
                        t = _get_transform(j)
                        t_next = _get_transform(j + 1)
                        y = _image_transform.affine_flow(
                            x, t, t_next,
                            output_wh=self._output_wh,
                            dim_order=dim_order,
                            is_normalized=is_normalized,
                            im_wh=_get_im_wh(j)
                        )
                    else:
                        y = np.zeros_like(x)
                    y_array.append(y)
            else:
                raise ValueError("internal error: unrecognzied field type")

            if self._output_postfix is None:
                outputs[i] = y_array
            else:
                outputs.append(y_array)

        return tuple(outputs)

    def _transform_to_affine(self, t, im_wh=None):

        ttype = self._transform_type
        is_normalized = ttype.endswith(":n")
        if is_normalized:
            ttype = ttype[:-2]
            assert im_wh is not None, "im_wh cannot be None when using normalized mode"
        if im_wh is None:
            im_w = None
            im_h = None
        else:
            im_w, im_h = im_wh

        w0, h0 = self._output_wh

        if ttype == "affine":
            t = np.array(t).astype(np.float32)
            assert len(t.shape)==2 and t.shape[0] == 2 and t.shape[1] == 3, \
                "affine transform should be defined as a 2x3 matrix"
            t = copy(t)
            if is_normalized:
                t = _image_transform.normalized_transform_to_unnormalized_transform(
                    t,
                    dst_w=w0, dst_h=h0,
                    src_w=im_w, src_h=im_h,
                )

        elif ttype.startswith("box"):
            t = np.array(t)
            assert len(t.shape) == 1 and t.shape[0] == 4, \
                "box must be four dimensional"

            if ttype == "box" or ttype == "box:xywh":
                u, v, w, h = t
            elif ttype == "box:yxhw":
                v, u, h, w = t
            elif ttype == "box:xyxy":
                u = t[0]
                v = t[1]
                w = t[2] - t[0]
                h = t[3] - t[1]
            elif ttype == "box:yxyx":
                v = t[0]
                u = t[1]
                h = t[2] - t[0]
                w = t[3] - t[1]
            else:
                raise ValueError("Internal Error: Unrecognized box type")
            if is_normalized:
                u *= im_w
                v *= im_h
                w *= im_w
                h *= im_h
            t = np.zeros([2, 3], dtype=np.float32)
            t[0,0] = w / w0
            t[1,1] = h / h0
            t[0,2] = u
            t[1,2] = v

        elif ttype.startswith("topleft-box"):

            if t is None:
                t = [1, 1]
                is_normalized = True

            t = np.array(t)
            assert len(t.shape) == 1 and t.shape[0] == 2, \
                "box must be two dimensional"

            if ttype == "topleft-box" or ttype == "topleft-box:wh":
                w, h = t
            elif ttype == "topleft-box:hw":
                h, w = t
            else:
                raise ValueError("Internal Error: Unrecognized topleft-box type")
            if is_normalized:
                w *= im_w
                h *= im_h
            t = np.zeros([2, 3], dtype=np.float32)

            t[0,0] = w / w0
            t[1,1] = h / h0

        else:

            raise ValueError("Internal Error: Unrecognized transform type")

        return t


class ImScale(ImAffine):

    def __init__(
            self,
            output_wh,          # output size in pixel (height, width)
            fields_type_dict,   # mapping for field name to types
            # REMARK on types:
            #   Image: "im";
            #   Point: "xy", "yx"
            #   Box: "box:yxhw", "box:xywh", "box:yxyx", "box:xyxy"
            #   Flow: "flow:xy", "flow:yx"
            output_postfix=None,          # None means to replace original field
            has_next_frame_field=None,  # use to figure out if has valid optical flow
            *args, **kwargs
    ):

        super().__init__(
            output_wh=output_wh,
            transform=None,
            fields_type_dict=fields_type_dict,
            output_postfix=output_postfix,
            transform_type="topleft-box",
            has_next_frame_field=has_next_frame_field,
            *args, **kwargs
        )


class ImCropAndScale(ImAffine):

    def __init__(
            self,
            box,                # either the field name for box or a fixed box
            output_wh,          # output size in pixel (height, width)
            fields_type_dict,   # mapping for field name to types
            # REMARK on types:
            #   Image: "im";
            #   Point: "xy", "yx"
            #   Box: "box:yxhw", "box:xywh", "box:yxyx", "box:xyxy"
            #   Flow: "flow:xy", "flow:yx"
            output_postfix=None,          # None means to replace original field
            dim_order="xyhw",
            has_next_frame_field=None,  # use to figure out if has valid optical flow
            *args, **kwargs
    ):

        super().__init__(
            output_wh=output_wh,
            transform=box,
            fields_type_dict=fields_type_dict,
            output_postfix=output_postfix,
            transform_type="box:"+dim_order,
            has_next_frame_field=has_next_frame_field,
            *args, **kwargs
        )


class RandomNormalizedAffine(DataTransform):

    class CenterInsteadOfRandom:
        def __init__(self, value=0.5):
            self._val = value

        def random(self):
            return self._val

    class PixelOrRatio:
        def __init__(self, pixel_val, ratio_val):
            self._pixel = pixel_val
            self._ratio = ratio_val
            assert pixel_val is None or ratio_val is None, "pixel value and ratio cannnot be both set"

        def get_value(self, ref_val, def_val=None):
            if self._pixel is not None:
                return self._pixel
            elif self._ratio is not None:
                return self._ratio * ref_val
            else:
                return def_val

        def is_set(self):
            return self._pixel is not None or self._ratio is not None

    def __init__(
            self,
            ref_im_field, output_transform_field=None,
            input_aspect_ratio=None,  # w/h if not specified, use random aspect ratio based one shorter, width or height
            min_input_aspect_ratio=None,
            max_input_aspect_ratio=None,
            min_shorter=None, min_shorter_ratio=None,
            max_shorter=None, max_shorter_ratio=None,
            min_width=None, min_width_ratio=None,
            max_width=None, max_width_ratio=None,
            min_height=None, min_height_ratio=None,
            max_height=None, max_height_ratio=None,
            padding_x=None, padding_x_ratio=None,
            padding_y=None, padding_y_ratio=None,
            max_rotation_in_angle=None, max_rotation_in_radius=None,
            random_stream=None,     # can be CenterInsteadOfRandom()
            foreground_box_field=None, foreground_box_dim_order="xywh",
            has_next_frame_field=None,
            box_as_image=False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert isinstance(ref_im_field, str), "ref_im_field must be a str"
        self._ref_im_field_idx = self.index_in_fields(ref_im_field)

        if output_transform_field is None:
            output_transform_field = ref_im_field + "/affine-transform"
        assert isinstance(output_transform_field, str), "output_transform_field must be a str"
        assert output_transform_field not in self.input_keys(), "output_transform_field conflicts with existing fields"
        self._output_keys = self.input_keys() + (output_transform_field,)
        self._output_types = self.input_types() + ("float32",)
        if self._output_shapes is not None:
            self._output_shapes = self.input_shapes() + ([None, 2, 3],)

        if foreground_box_field is None:
            self._foreground_box_field_idx = None
        else:
            self._foreground_box_field_idx = self.index_in_fields(foreground_box_field)
        assert foreground_box_dim_order in ("xywh", 'yxhw', 'xyxy', 'yxyx'), \
            "unrecognized foreground_box_dim_order"
        self._foreground_box_dim_order = foreground_box_dim_order
        if box_as_image:
            assert self._foreground_box_field_idx is not None, \
                "box must be given if box_as_image = True"
        self._box_as_image = box_as_image

        if random_stream is None:
            random_stream = random.Random()
        elif isinstance(random_stream, (int, float)):
            random_stream = random.Random(random_stream)
        assert hasattr(random_stream, "random") and isinstance(getattr(random_stream, "random"), Callable), \
            "random_stream must has a member function called random()"
        self._random_stream = random_stream
        self._random_stream_lock = threading.Lock()

        if input_aspect_ratio is None:
            if min_input_aspect_ratio is not None or max_input_aspect_ratio is not None:
                assert min_input_aspect_ratio is not None and max_input_aspect_ratio is not None, \
                    "min_input_aspect_ratio and max_input_aspect_ratio must be set together"
                assert min_input_aspect_ratio <= max_input_aspect_ratio, \
                    "min_input_aspect_ratio must be no greater than max_input_aspect_ratio"
                self._input_aspect_ratio = (min_input_aspect_ratio, max_input_aspect_ratio)
            else:
                self._input_aspect_ratio = None
        else:
            assert min_input_aspect_ratio is None and max_input_aspect_ratio is None, \
                "input_aspect_ratio cannot be set together with min_input_aspect_ratio/max_input_aspect_ratio"
            self._input_aspect_ratio = (input_aspect_ratio, input_aspect_ratio)
        # Remark: if not specified, use random aspect ratio based one shorter, width or height

        self._min_shorter = self.PixelOrRatio(min_shorter, min_shorter_ratio)
        self._max_shorter = self.PixelOrRatio(max_shorter, max_shorter_ratio)
        self._min_width = self.PixelOrRatio(min_width, min_width_ratio)
        self._max_width = self.PixelOrRatio(max_width, max_width_ratio)
        self._min_height = self.PixelOrRatio(min_height, min_height_ratio)
        self._max_height = self.PixelOrRatio(max_height, max_height_ratio)

        if not self._min_shorter.is_set() and not self._min_width.is_set() and not self._min_height.is_set():
            self._min_shorter = self.PixelOrRatio(1, None)
        if self._min_shorter.is_set():
            assert not self._min_width.is_set() and not self._min_height.is_set(), \
                "min_shorter cannot be set together min_width and min_height"
            assert self._input_aspect_ratio is not None, \
                "when specifying the shorter edge, the input_aspect_ratio must be set."

        if not self._max_shorter.is_set() and not self._max_width.is_set() and not self._max_height.is_set():
            self._max_shorter = self.PixelOrRatio(None, 1)
        if self._max_shorter.is_set():
            assert not self._max_width.is_set() and not self._max_height.is_set(), \
                "max_shorter cannot be set together max_width and max_height"
            assert self._input_aspect_ratio is not None, \
                "when specifying the shorter edge, the input_aspect_ratio must be set."

        self._padding_x = self.PixelOrRatio(padding_x, padding_x_ratio)
        self._padding_y = self.PixelOrRatio(padding_y, padding_y_ratio)
        if max_rotation_in_angle is None:
            if max_rotation_in_radius is not None:
                self._max_rotation = abs(max_rotation_in_radius) / math.pi * 180
            else:
                self._max_rotation = 0
        else:
            assert max_rotation_in_radius is None, \
                "should not both set max_rotation_in_angle and max_rotation_in_radius"
            self._max_rotation = abs(max_rotation_in_angle)

        # for sequence
        self._has_next_frame_field_idx = has_next_frame_field_to_idx(has_next_frame_field, self.input_keys())

    def _get_random(self):
        self._random_stream_lock.acquire()
        a = self._random_stream.random()
        self._random_stream_lock.release()
        return a

    @staticmethod
    def _interp_two_ends(a, b, alpha):
        return (b-a) * alpha + a

    @staticmethod
    def _interp_two_scales(a, b, alpha):
        return math.exp((math.log(b)-math.log(a)) * alpha + math.log(a))

    def _get_a_transform(self, im, box):

        # image
        if not isinstance(im, np.ndarray):
            im = np.array(im)

        im_h = im.shape[0]
        im_w = im.shape[1]

        # transforms
        ts = []

        # src normalized <- src unnormalized
        norm_unnorm_t = np.eye(3, 3, dtype=np.float32)
        norm_unnorm_t[0, 0] = 1/im_w
        norm_unnorm_t[1, 1] = 1/im_h
        ts.append(norm_unnorm_t)

        # box
        if box is not None:

            # canonicalize box coordinate
            box = np.array(box)
            foreground_box_dim_order = self._foreground_box_dim_order
            is_box_normalized = self._foreground_box_dim_order.endswith(":n")
            if is_box_normalized:
                foreground_box_dim_order = foreground_box_dim_order[:-2]
            if foreground_box_dim_order == "xywh":
                box[2:4] = box[0:2] + box[2:4]
            elif foreground_box_dim_order == "yxhw":
                box = np.concatenate([box[[1, 0]], box[[1, 0]] + box[[3, 2]]], axis=0)
            elif foreground_box_dim_order == "xyxy":
                pass
            elif foreground_box_dim_order == "yxyx":
                box = box[[1, 0, 3, 2]]
            else:
                raise ValueError("internal error: unrecognized foreground_box_dim_order")
            if is_box_normalized:
                box *= np.array([im_w, im_h, im_w, im_h])

            # trim per image boundary
            box[0] = max(box[0], -0.5)
            box[1] = max(box[1], -0.5)
            box[2] = min(box[2], im_w - 0.5)
            box[3] = min(box[3], im_h - 0.5)

        if self._box_as_image:
            im_w = box[2] - box[0]
            im_h = box[3] - box[1]
            im_left = box[0]
            im_right = box[2]
            im_top = box[1]
            im_bottom = box[3]
        else:
            im_left = -0.5
            im_right = im_w - 0.5
            im_top = -0.5
            im_bottom = im_h - 0.5

        # padding
        pad_x = self._padding_x.get_value(im_w, def_val=0)
        pad_y = self._padding_y.get_value(im_h, def_val=0)
        im_left -= pad_x
        im_right += pad_x
        im_top -= pad_y
        im_bottom += pad_y

        im_corners = np.array(
            [
                [im_left, im_top],
                [im_left, im_bottom],
                [im_right, im_top],
                [im_right, im_bottom],
            ]
        )

        # get min w and h
        if self._min_shorter.is_set():
            if im_h > im_w:
                min_w = self._min_shorter.get_value(im_w)
                min_h = 1
            else:
                min_h = self._min_shorter.get_value(im_h)
                min_w = 1
        else:
            min_w = self._min_width.get_value(im_w, def_val=1)
            min_h = self._min_height.get_value(im_h, def_val=1)

        # get max w and h
        if self._max_shorter.is_set():
            if im_h > im_w:
                max_w = self._max_shorter.get_value(im_right - im_left, def_val=im_right - im_left)
                max_h = im_bottom - im_top
            else:
                max_h = self._max_shorter.get_value(im_bottom - im_top, def_val=im_bottom - im_top)
                max_w = im_right - im_left
        else:
            max_w = self._max_width.get_value(im_right - im_left)
            max_h = self._max_height.get_value(im_bottom - im_top)

        assert max_w >= min_w, "max width must be no less than min width"
        assert max_h >= min_h, "max height must be no less than min height"

        # random rotation
        _rot = self._max_rotation * (self._get_random() * 2 - 1)

        if _rot != 0:
            im_rot_t = _image_transform._affine_to_perspective(
                cv2.getRotationMatrix2D(
                    center=((im_left+im_right)*0.5, (im_top+im_bottom)/2),
                    angle=_rot, scale=1.
                )
            )  # src rotated (unnormalized) <- src original (unnormalized)
            # src original (unnormalized) <- src rotated (unnormalized)
            im_rot_inv_t = np.linalg.inv(im_rot_t)
            ts.append(im_rot_inv_t)
            rotated_im_corners = _image_transform._homo_to_point2d(
                np.dot(_image_transform._point2d_to_homo(im_corners), im_rot_t.T)
            )
            im_left = np.min(rotated_im_corners[:, 0], axis=0)
            im_right = np.max(rotated_im_corners[:, 0], axis=0)
            im_top = np.min(rotated_im_corners[:, 1], axis=0)
            im_bottom = np.max(rotated_im_corners[:, 1], axis=0)

        # handle box
        if box is not None:
            # rotate box if needed
            if _rot != 0:
                # rotate and extend
                box_corners = np.array([
                    [box[0], box[1]],
                    [box[0], box[3]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                ])
                rotated_box_corners = _image_transform._homo_to_point2d(
                    np.dot(_image_transform._point2d_to_homo(box_corners), im_rot_t.T)
                )
                box[0] = np.min(rotated_box_corners[:, 0], axis=0)
                box[2] = np.max(rotated_box_corners[:, 0], axis=0)
                box[1] = np.min(rotated_box_corners[:, 1], axis=0)
                box[3] = np.max(rotated_box_corners[:, 1], axis=0)

            # update min width min height
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]

            min_w = max(min_w, box_w)
            min_h = max(min_h, box_h)
            max_w = max(min_w, max_w)
            max_h = max(min_h, max_h)

        else:
            box_w = 1
            box_h = 1

        # random crop size
        if self._input_aspect_ratio is None:
            crop_w = self._interp_two_ends(min_w, max_w, self._get_random())
            crop_h = self._interp_two_ends(min_h, max_h, self._get_random())
        else:
            # parse aspect ratio
            min_input_aspect_ratio = self._input_aspect_ratio[0]
            max_input_aspect_ratio = self._input_aspect_ratio[1]

            # figure out feasible range (at lease one aspect ratio can be satisfied)
            min_w2 = max(min_w, min_h * min_input_aspect_ratio)
            min_h2 = max(min_h, min_w / max_input_aspect_ratio)
            min_w = min_w2
            min_h = min_h2
            max_w2 = min(max_w, max_h * max_input_aspect_ratio)
            max_h2 = min(max_h, max_w / min_input_aspect_ratio)
            max_w = max_w2
            max_h = max_h2

            # force to solve infeasible cases and fit to the box
            min_w = max(min_w, box_w, box_h * min_input_aspect_ratio)
            min_h = max(min_h, box_h, box_w / max_input_aspect_ratio)
            max_w = max(min_w, max_w)
            max_h = max(min_h, max_h)

            # fit aspect ratio and try to also fit other constraints
            w_first = self._get_random() >= 0.5
            crop_a = self._interp_two_scales(min_input_aspect_ratio, max_input_aspect_ratio, self._get_random())
            if w_first:
                crop_w = self._interp_two_ends(min_w, max_w, self._get_random())
                crop_h = min(max_h, max(min_h, crop_w / crop_a))
            else:
                crop_h = self._interp_two_ends(min_h, max_h, self._get_random())
                crop_w = min(max_w, max(min_w, crop_h * crop_a))

        # feasible location
        min_left = im_left
        max_left = im_right - crop_h
        min_top = im_top
        max_top = im_bottom - crop_w
        if box is not None:
            min_left = max(min_left, box[0]-crop_w+box_w)
            max_left = min(max_left, box[0])
            min_top = max(min_top, box[1]-crop_h+box_h)
            max_top = min(max_top, box[1])
        crop_left = self._interp_two_ends(min_left, max_left, self._get_random())
        crop_top = self._interp_two_ends(min_top, max_top, self._get_random())

        # src unnormalized <- dst normalized
        crop_t = np.eye(3, 3, dtype=np.float32)
        crop_t[0, 0] = crop_h
        crop_t[1, 1] = crop_w
        crop_t[0, 2] = crop_left
        crop_t[1, 2] = crop_top
        ts.append(crop_t)

        t = np.eye(3, 3, dtype=np.float32)
        while ts:
            t1 = ts.pop()
            t = np.dot(t1, t)
        t = t[:2]
        return t.astype(np.float32)

    def transform(self, *args):

        ref_images = args[self._ref_im_field_idx]
        if self._foreground_box_field_idx is not None:
            fg_boxes = np.array(args[self._foreground_box_field_idx])
            assert len(fg_boxes.shape) == 2 and fg_boxes.shape[1] == 4, "the gt box must be Nx4"
            assert len(fg_boxes) == len(ref_images), "ref_images and fg_boxes must have the same length"
        else:
            fg_boxes = list([None] * len(ref_images))

        assert len(ref_images) == len(fg_boxes), \
            "the number of boxes are inconsistent with the number of ref images"

        has_next_frame = get_has_next_frame_field(self._has_next_frame_field_idx, args)

        af = []
        t = None
        for i in range(len(ref_images)):
            if t is None:
                im = ref_images[i]
                box = fg_boxes[i]
                t = self._get_a_transform(im, box)
            af.append(t)
            if not has_next_frame[i]:
                t = None

        return tuple(args) + (af,)


class SwapXY(DataTransform):
    def __init__(self, field_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(field_names, str):
            field_names = [field_names]
        _field_names = list(field_names)
        self._field_mapping = self.index_in_fields(_field_names)

    @staticmethod
    def swap_xy(a):
        a = np.array(a)
        b = np.reshape(a, [-1, a.shape[-1]])
        if b.shape[-1] == 2:
            b = b[:, [1, 0]]
        elif b.shape[-1] == 4:
            b = b[:, [1, 0, 3, 2]]
        else:
            raise ValueError("wrong array shape, cannot swap xy")
        return b

    def transform(self, *args):
        outputs = list(args)
        for i in self._field_mapping:
            a = args[i]
            if isinstance(a, np.ndarray):
                b = self.swap_xy(a)
            else:
                b = []
                for a_j in a:
                    b.append(self.swap_xy(a_j))
            outputs[i] = b
        return tuple(outputs)


class AddConstantField(DataTransform):

    def __init__(self, field_name, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert field_name not in self.input_keys(), "field_name conflicts with existing keys"
        value = np.array(value)
        self._output_keys += (field_name,)
        t = value.dtype.name
        if t == "float":
            t = "float32"
            value = value.astype(np.float32)
        elif t == "int":
            t = "int32"
            value = value.astype(np.int32)
        self._output_types += (t,)
        if self.input_shapes() is not None:
            self._output_shapes += ([None] + list(value.shape),)
        self._value = value

    def transform(self, *args):
        new_col = [self._value] * len(args[0])
        outputs = tuple(args) + (new_col,)
        return outputs


class Cast(DataTransform):

    def __init__(self, type_dict, *args, **kwargs):
        assert isinstance(type_dict, dict), "type_dict must be a dict"
        super().__init__(*args, **kwargs)
        self._field_mapping = self.index_in_fields(type_dict.keys())
        self._new_field_types = tuple(type_dict.values())
        _output_types = list(self.input_types())
        for i, t in zip(self._field_mapping, type_dict.values()):
            _output_types[i] = t
        self._output_types = tuple(_output_types)

    def transform(self, *args):
        outputs = list(args)
        for i, t in zip(self._field_mapping, self._new_field_types):
            a = args[i]
            if isinstance(a, np.ndarray):
                b = a.astype(t)
            else:
                b = []
                for a_j in a:
                    b.append(np.array(a_j).astype(t))
            outputs[i] = b
        return tuple(outputs)


class Copy(DataTransform):

    def __init__(self, copy_dict, *args, **kwargs):
        assert isinstance(copy_dict, dict), "type_dict must be a dict"
        super().__init__(*args, **kwargs)
        self._src_field_mapping = self.index_in_fields(copy_dict.keys())
        self._new_field_types = tuple(copy_dict.values())

        _output_keys = list(self.input_keys())
        _output_types = list(self.input_types())
        _output_shapes = list(self.input_shapes()) if self.input_shapes() is not None else None
        for i, dst_key in zip(self._src_field_mapping, copy_dict.values()):
            assert dst_key not in _output_keys, "dst_key conflicts with existing keys"
            _output_keys.append(dst_key)
            _output_types.append(self.input_types()[i])
            if _output_shapes is not None:
                _output_shapes.append(self.input_shapes()[i])
        self._output_keys = tuple(_output_keys)
        self._output_types = tuple(_output_types)
        if _output_shapes is None:
            self._output_shapes = None
        else:
            self._output_shapes = tuple(_output_shapes)

    def transform(self, *args):
        outputs = list(args)
        for i in self._src_field_mapping:
            outputs.append(copy(args[i]))
        return tuple(outputs)


class Remove(DataTransform):
    def __init__(self, fields_to_remove, *args, **kwargs):
        if isinstance(fields_to_remove, str):
            fields_to_remove = [fields_to_remove]
        fields_to_remove = set(fields_to_remove)
        super().__init__(*args, **kwargs)

        self._field_idx_to_remove = sorted(self.index_in_fields(fields_to_remove), reverse=True)
        _output_keys = list(self.input_keys())
        _output_types = list(self.input_types())
        _output_shapes = list(self.input_shapes())
        for i in self._field_idx_to_remove:
            del _output_keys[i]
            del _output_types[i]
            if _output_shapes is not None:
                del _output_shapes[i]
        self._output_keys = tuple(_output_keys)
        self._output_types = tuple(_output_types)
        self._output_shapes = tuple(_output_shapes)

    def transform(self, *args):
        outputs = list(args)
        for i in self._field_idx_to_remove:
            del outputs[i]
        return tuple(outputs)


class Concat(DataTransform):
    def __init__(self, fields_to_concat, concatenated_field, axis, *args, **kwargs):
        if isinstance(fields_to_concat, str):
            fields_to_concat = [fields_to_concat]
        fields_to_concat = set(fields_to_concat)
        super().__init__(*args, **kwargs)

        self._field_idx_to_concat = sorted(self.index_in_fields(fields_to_concat), reverse=True)
        assert isinstance(concatenated_field, str), \
            "concatenated_field must be a str"
        assert concatenated_field not in self.input_keys(), \
            "concatenated_field conflicts with existing fields"

        assert self._field_idx_to_concat, "must at least one field to concat"
        the_type = self.input_types()[self._field_idx_to_concat[0]]
        for i in self._field_idx_to_concat[1:]:
            assert self.input_types()[i] == the_type, "the fields to concat must have the same type"

        the_shape = list(self.input_shapes()[self._field_idx_to_concat[0]])
        for i in self._field_idx_to_concat[1:]:
            cur_shape = self.input_shapes()[i]
            if the_shape is None:
                the_shape = list(cur_shape)
                if the_shape is not None:
                    the_shape[axis] = None
            else:
                if cur_shape is None:
                    cur_shape = [None] * len(the_shape)
                assert len(cur_shape) == len(the_shape), "inconsistent tensor rank"
                the_axis = axis if axis >= 0 else len(the_shape) + axis
                for j in range(len(the_shape)):
                    if j == the_axis:
                        if the_shape[j] is not None:
                            if cur_shape[j] is not None:
                                the_shape[j] += cur_shape[j]
                            else:
                                the_shape[j] = None
                    else:
                        if the_shape[j] is None:
                            the_shape[j] = cur_shape[j]
                        elif cur_shape is not None:
                            assert the_shape[j] == cur_shape[j], "inconsistent shape"

        self._output_keys = self.input_keys() + (concatenated_field,)
        self._output_types = self.input_types() + (the_type,)
        self._output_shapes = self.input_shapes() + (the_shape,)
        self._axis = axis

    def transform(self, *args):
        outputs = list(args)
        outputs.append(np.concatenate(
            [args[i] for i in self._field_idx_to_concat], axis=self._axis
        ))
        return tuple(outputs)


class Rename(DataTransform):
    def __init__(self, rename_dict, *args, **kwargs):
        assert isinstance(rename_dict, dict), "rename_dict must be a dict"
        super().__init__(*args, **kwargs)
        self._field_mapping = self.index_in_fields(rename_dict.keys())
        self._new_field_names = tuple(rename_dict.values())
        _output_keys = list(self.input_keys())
        for i, k in zip(self._field_mapping, rename_dict.values()):
            _output_keys[i] = k
        self._output_keys = tuple(_output_keys)

    def transform(self, *args):
        return args
