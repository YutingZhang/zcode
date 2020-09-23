from collections import OrderedDict
from typing import Union, Tuple, Callable, Optional, Type, Dict
from copy import deepcopy
import re


__all__ = [
    'NonRecursiveDict',
    'NonRecursiveList',
    'NonRecursiveTuple',
    'disable_recursive',
    'not_recursive',
    'disable_recursive',
    'enable_recursive',
    'recursive_generic_condition_func',
    'flatten_str_dict',
    'recursive_unpack',
    'recursive_apply',
    'recursive_apply_removing_tag',
    'ToRemove',
    'recursive_flatten_to_list',
    'recursive_wrap',
    'recursive_flatten_with_wrap_func',
    'recursive_indicators',
    'recursive_select',
    'recursive_merge_2dicts',
    'recursive_merge_dicts',
    'NoDefaultValue',
    'get_dict_entry',
    'get_entry',
    'get_single_entry',
    'AutoEntryDictWrapper',
]


class NonRecursiveDict(dict):
    pass


class NonRecursiveList(list):
    pass


class NonRecursiveTuple(tuple):
    pass


_non_recursive_type_mapping={
    dict: NonRecursiveDict,
    list: NonRecursiveList,
    tuple: NonRecursiveTuple,
}


_non_recursive_type_rev_mapping={
    NonRecursiveDict: dict,
    NonRecursiveList: list,
    NonRecursiveTuple: tuple,
}


def not_recursive(a):
    if isinstance(a, tuple(_non_recursive_type_mapping.keys())):
        return _non_recursive_type_mapping[type(a)](a)
    else:
        raise ValueError("the type does not support not_recursive")


disable_recursive = not_recursive


def enable_recursive(a):
    if _is_prevented_from_recursive(a):
        return _non_recursive_type_rev_mapping[type(a)](a)
    else:
        return a


def _is_prevented_from_recursive(a):
    return isinstance(a, tuple(_non_recursive_type_rev_mapping.keys()))


is_not_recursive = _is_prevented_from_recursive


def recursive_generic_condition_func(x, *args):
    return (
        not isinstance(x, (list, tuple, dict))
    )


class ToRemove:
    pass


def recursive_apply_removing_tag():
    return ToRemove


class BoundRecursiveApplyScope:
    last_element = None

    def __init__(self, condition_func, func, backup_func):
        self._condition_func = condition_func
        self._func = func
        self._backup_func = backup_func
        self._previous_element = None

    def __enter__(self):
        assert self._previous_element is None, "already in its scope"
        self._previous_element = type(self).last_element
        type(self).last_element = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        type(self).last_element = self._previous_element
        self._previous_element = None

    @classmethod
    def current_bound_recursive_apply(cls, *args):
        assert isinstance(cls.last_element, BoundRecursiveApplyScope), \
            "nothing bound or an internal error"
        return _recursive_apply(
            cls.last_element._condition_func,
            cls.last_element._func,
            *args,
            backup_func=cls.last_element._backup_func
        )


class CustomRecursiveApplyScope:
    last_element = None

    def __init__(self, condition_func, recursive_apply_func):
        self._condition_func = condition_func
        self._recursive_apply_func = recursive_apply_func
        self._previous_element = None

    def __enter__(self):
        assert self._previous_element is None, "already in its scope"
        self._previous_element = type(self).last_element
        type(self).last_element = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        type(self).last_element = self._previous_element
        self._previous_element = None

    @property
    def previous_element(self):
        return self._previous_element

    def test_condition(self, *args):
        return self._condition_func(*args)

    def call_recursive_apply_func(
            self, condition_func, func, *args, backup_func
    ):
        with BoundRecursiveApplyScope(condition_func, func, backup_func):
            return self._recursive_apply_func(*args)

    current_bound_recursive_apply = BoundRecursiveApplyScope.current_bound_recursive_apply


def _recursive_apply(condition_func, func, *args, backup_func=None):

    def generic_backup_func(*my_args):
        if len(my_args) == 1:
            return my_args[0]
        else:
            return my_args

    if condition_func is None:
        condition_func = recursive_generic_condition_func

    if backup_func is None:
        backup_func = generic_backup_func

    if condition_func(*args):
        return func(*args)

    cras = CustomRecursiveApplyScope.last_element
    while cras is not None:
        assert isinstance(cras, CustomRecursiveApplyScope), \
            "internal error: cras must be CustomRecursiveApplyScope"
        if cras.test_condition(*args):
            return cras.call_recursive_apply_func(
                condition_func, func, *args, backup_func=backup_func
            )
        cras = cras.previous_element

    if not _is_prevented_from_recursive(args[0]):
        if isinstance(args[0], (list, tuple)):
            L = list()
            for i in range(len(args[0])):
                sub_args = (t[i] for t in args)
                elt_val = _recursive_apply(
                    condition_func, func, *sub_args, backup_func=backup_func
                )
                if elt_val is not ToRemove:
                    L.append(elt_val)
            if not L and args[0]:
                return ToRemove
            atype = type(args[0])
            if atype is list or atype is tuple:
                return atype(L)
            else:
                try:
                    return atype(*L)
                except ValueError:
                    pass
                return atype(L)
        elif isinstance(args[0], dict):
            D = type(args[0])()
            for k in args[0]:
                sub_args = []
                for t in args:
                    if k in t:
                        sub_args.append(t[k])
                    else:
                        sub_args.clear()
                        break
                if sub_args:
                    elt_val = _recursive_apply(condition_func, func, *sub_args, backup_func=backup_func)
                else:
                    # if key does not exist in every dict, then simply keep the first
                    elt_val = args[0][k]
                if elt_val is not ToRemove:
                    D[k] = elt_val
            if not D and args[0]:
                return ToRemove
            return D

    return backup_func(*args)


def recursive_apply(condition_func, func, *args, unpack_outputs: bool = False, **kwargs):

    if unpack_outputs:
        return _recursive_apply_with_outputs_unpacked(condition_func, func, *args, **kwargs)

    a = _recursive_apply(condition_func, func, *args, **kwargs)
    if a is ToRemove:
        a = type(args[0])()

    if unpack_outputs:
        a = recursive_unpack(condition_func, a)
    return a


class _OutputTuple(tuple):
    pass


def _recursive_apply_with_outputs_unpacked(condition_func, func, *args, num_outputs: int = None, **kwargs):
    a = recursive_apply(condition_func, lambda x: _OutputTuple(func(x)), *args, **kwargs)
    b = recursive_unpack(lambda x: isinstance(x, _OutputTuple), a)
    if num_outputs is not None:
        if len(b) == 0:
            return (a,) * num_outputs
        else:
            assert len(b) == num_outputs, \
                "the specified number of outputs does not match with the actual number of outputs"
    return b


def recursive_flatten_to_list(condition_func, x):

    if condition_func is None:
        condition_func = recursive_generic_condition_func

    if condition_func(x):
        return [x]
    elif not _is_prevented_from_recursive(x):
        if isinstance(x, (list, tuple)):
            return [z for y in x for z in recursive_flatten_to_list(condition_func, y)]
        elif isinstance(x, dict):
            return [z for y in x.values() for z in recursive_flatten_to_list(condition_func, y)]
        else:
            return []
    else:
        return []


def recursive_flatten_with_wrap_func(condition_func, x):

    if condition_func is None:
        condition_func = recursive_generic_condition_func

    return (recursive_flatten_to_list(condition_func, x),
            lambda val: recursive_wrap(condition_func, val, x))


def recursive_unpack(condition_func, x):
    assert condition_func is not None, "no default condition_func for recursive_unpack"
    y, wrap_func = recursive_flatten_with_wrap_func(condition_func, x)
    out = []
    for y_i in zip(*y):
        out.append(wrap_func(list(y_i)))
    return tuple(out)


class _RecursiveWrapIDT:    # index tracker
    def __init__(self):
        self.i = 0

    def inc(self):
        self.i += 1


def recursive_wrap(condition_func, val, ref):
    if condition_func is None:
        condition_func = recursive_generic_condition_func
    return _recursive_wrap(condition_func, val, ref, _RecursiveWrapIDT())


def _recursive_wrap(condition_func, val, ref, idt):

    if condition_func(ref):
        cur_id = idt.i
        idt.inc()
        return val[cur_id]
    elif not _is_prevented_from_recursive(ref):
        if isinstance(ref, (list,tuple)):
            L = list()
            for t in ref:
                L.append(_recursive_wrap(condition_func, val, t, idt))
            atype = type(ref)
            if atype is list or atype is tuple:
                return atype(L)
            else:
                try:
                    return atype(*L)
                except ValueError:
                    pass
                return atype(L)
        elif isinstance(ref, dict):
            D = type(ref)()
            for k, v in ref.items():
                D[k] = _recursive_wrap(condition_func, val, v, idt)
            return D
        else:
            return ref
    else:
        return ref


def first_element_apply(condition_func, func, *args):
    if condition_func is None:
        condition_func = recursive_generic_condition_func
    if condition_func(*args):
        return func(*args)
    elif not _is_prevented_from_recursive(args[0]):
        if isinstance(args[0], (list,tuple)):
            for i in range(len(args[0])):
                sub_args = (t[i] for t in args)
                v = first_element_apply(condition_func, func, *sub_args)
                if v is not None:
                    return v
            return None
        elif isinstance(args[0], dict):
            for k in args[0]:
                sub_args = (t[k] for t in args)
                v = first_element_apply(condition_func, func, *sub_args)
                if v is not None:
                    return v
            return None
        else:
            return None
    else:
        return None


def flatten_str_dict(hierarchical_dict, sep="", flatten_list_and_tuple=False):
    flatten_dict = OrderedDict()
    for k, v in hierarchical_dict.items():
        k = str(k)
        if flatten_list_and_tuple and isinstance(v, (list, tuple)) and not _is_prevented_from_recursive(v):
            v = OrderedDict(enumerate(v))
        if isinstance(v, dict) and not _is_prevented_from_recursive(v):
            flatten_v = flatten_str_dict(v, sep, flatten_list_and_tuple)
            for kk, vv in flatten_v.items():
                fk = k + sep + kk
                assert fk not in flatten_dict, "conflicted keys"
                flatten_dict[k + sep + kk] = vv
        else:
            flatten_dict[k] = v
    return flatten_dict


def recursive_indicators(condition_func, x, default_indicator=False):
    if condition_func is None:
        condition_func = recursive_generic_condition_func
    the_indicators = recursive_apply(
        condition_func, lambda _: default_indicator, x, backup_func=lambda _: default_indicator)
    return the_indicators


def recursive_select(x, the_indicators):
    def selector_func(ind, elt_val):
        if ind:
            return elt_val
        else:
            return recursive_apply_removing_tag()
    selected_struct = _recursive_apply(None, selector_func, the_indicators, x)
    if selected_struct is ToRemove:
        if isinstance(x, (list, tuple, dict)):
            selected_struct = type(x)()
        else:
            selected_struct = None
    return selected_struct


def _merge_replace_func(x1, x2):
    return x2


def recursive_merge_2dicts(d1, d2, merge_func=None):

    if merge_func is None:
        merge_func = _merge_replace_func

    if (
            (not isinstance(d1, dict) or _is_prevented_from_recursive(d1)) or
            (not isinstance(d2, dict) or _is_prevented_from_recursive(d2))
    ):
        return merge_func(d1, d2)

    k1 = set(d1.keys())
    k2 = set(d2.keys())
    k_both = k1.intersection(k2)
    k1_only = k1-k2
    k2_only = k2-k1

    all_keys = list(d1.keys())
    all_keys.extend(k for k in d2.keys() if k in k2_only)

    q = type(d1)()

    for k in all_keys:
        if k in k_both:
            if d2[k] is not ToRemove:
                q[k] = recursive_merge_2dicts(d1[k], d2[k], merge_func=merge_func)
        elif k in k2_only:
            if d2[k] is not ToRemove:
                q[k] = d2[k]
        else:
            q[k] = d1[k]

    return q


def recursive_merge_dicts(*args, merge_func=None):
    if not args:
        return dict()

    q = args[0]
    for p in args[1:]:
        q = recursive_merge_2dicts(q, p, merge_func=merge_func)
    return q


class NoDefaultValue:
    pass


class _GetSingleEntryReturnDefaultValue(BaseException):
    def __init__(self, val):
        self.val = val


def _get_item_advanced(d, item):
    if isinstance(item, str) and item.startswith('@re:'):
        pattern = re.compile(item[len('@re:'):])
        all_keys = list(d)
        matched_keys = [k for k in all_keys if isinstance(k, str) and pattern.match(k)]
        assert len(matched_keys) == 1, 'did not find the key or the key is not unique'
        item = matched_keys[0]
    return d[item]


def _get_single_entry_no_set_item(
        d, item, default_val=NoDefaultValue, default_val_func: Optional[Callable]=None,
        allow_callable_object=True
):

    if default_val_func is None and default_val is NoDefaultValue and not isinstance(item, Callable):
        return _get_item_advanced(d, item)

    try:
        return _get_item_advanced(d, item)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exception:
        key_exception = exception

    if allow_callable_object and isinstance(item, Callable):
        try:
            return item(d)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exception:
            call_exception = exception

    if default_val_func is not None:
        assert default_val is None or default_val is NoDefaultValue, \
            'default_val and default_val_func should not be both specified'
        try:
            raise _GetSingleEntryReturnDefaultValue(default_val_func(item))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass
        raise _GetSingleEntryReturnDefaultValue(default_val_func())

    if default_val is NoDefaultValue:
        raise key_exception

    raise _GetSingleEntryReturnDefaultValue(default_val)


def get_single_entry(
        d, item, default_val=NoDefaultValue, default_val_func: Optional[Callable]=None,
        allow_callable_object=True, set_item_with_default=False
):
    try:
        return _get_single_entry_no_set_item(
            d, item, default_val=default_val, default_val_func=default_val_func,
            allow_callable_object=allow_callable_object
        )
    except _GetSingleEntryReturnDefaultValue as h:
        v = h.val

    if set_item_with_default:
        d[item] = v

    return v


def get_entry(
        d, path: Union[Tuple, str],
        default_val=NoDefaultValue, default_val_func: Optional[Callable]=None,
        path_separator=None, default_dict_type: Optional[Type[Dict]]=None,
        allow_callable_object=True, set_item_with_default=False
):

    if isinstance(path, str):
        if path_separator is None:
            if "/" in path:
                path_separator = "/"
            else:
                path_separator = "."
        path_list = path.split(path_separator)
    else:
        assert path_separator is None, "path_separator should not be specified if path is a tuple"
        path_list = tuple(path)

    if default_dict_type:
        def dict_type(*args, **kwargs):
            return default_dict_type()
    else:
        def dict_type(*args, **kwargs):
            return type(d)()

    a = d
    if default_val is NoDefaultValue and default_val_func is None:
        for p in path_list[:-1]:
            a = get_single_entry(a, p, allow_callable_object=allow_callable_object)
    else:
        for p in path_list[:-1]:
            a = get_single_entry(
                a, p, default_val_func=dict_type,
                allow_callable_object=allow_callable_object,
                set_item_with_default=set_item_with_default
            )

    if path_list:
        a = get_single_entry(
            a, path_list[-1], default_val=default_val, default_val_func=default_val_func,
            allow_callable_object=allow_callable_object,
            set_item_with_default=set_item_with_default
        )

    return a


get_dict_entry = get_entry


class AutoEntryDictWrapper:

    def __init__(self, d: dict, default_val=None, path_separator=None, default_val_func: Callable=None):
        self._d = d
        if default_val_func is None:
            def default_val_func(_):
                return deepcopy(default_val)
        else:
            assert default_val is None, "default_value should not be specified if default_val_func is specified"
        self._default_val_func = default_val_func
        self._path_separator = path_separator

        for m in dir(d):
            if not hasattr(self, m):
                setattr(self, m, getattr(d, m))

    def __getitem__(self, item):
        return get_dict_entry(
            self._d, item, self._default_val_func(item), self._path_separator,
            allow_callable_object=False, set_item_with_default=True
        )

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def items(self):
        return self._d.items()


def into_structured_ndarray():
    from .recursive_utils4np import into_structured_ndarray
    return into_structured_ndarray()

