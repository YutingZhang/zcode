from collections import Iterable, OrderedDict
from copy import copy
from typing import Dict, Union, Any, Type, List, Tuple
from z_python_utils.classes import value_class_for_with, dummy_class_for_with
import sys


__all__ = [
    'OptionDef',
    'OptionDefaultValue',
    'OptionStruct',
    'SubOption',
]


global_incore_function = value_class_for_with(False)


def os_incore(func):

    def wrap(*args, **kwargs):
        if len(args) >= 1:
            arg_self = args[0]
        else:
            arg_self = None

        if isinstance(arg_self, OptionStructCore):
            if '_incore_function' in dir(arg_self):
                c = getattr(arg_self, '_incore_function')(True)
            else:
                c = dummy_class_for_with()
        else:
            c = global_incore_function(True)

        with c:
            return func(*args, **kwargs)

    return wrap


class OptionStructUnsetCacheNone:
    pass


class OptionDefaultValue:
    def __init__(self, value, type_mapping: Union[Type, List[Tuple[Type, Type]]] = None):
        self._value = value
        if type_mapping is None:
            type_mapping = []
        elif isinstance(type_mapping, type):
            type_mapping = [(Any, type_mapping)]
        else:
            type_mapping = []
            for input_type, target_type in type_mapping:
                type_mapping.append((input_type, target_type))
        self._type_mapping = type_mapping

    @property
    def default_value(self):
        return self._value

    def custom_value(self, value):
        for input_type, target_type in self._type_mapping:
            if input_type is None or input_type is Any or isinstance(value, input_type):
                value = target_type(value)
                break
        return value


class OptionStructCore:
    """
    OptionStructCore: implementated the core functions to set default values and merge user specified values
    """

    @os_incore
    def __init__(self, option_dict_or_struct=None):
        self.user_dict = {}
        self.enabled_dict = OrderedDict()
        self.unset_set = set()
        self.option_def = None
        self.option_name = None
        if option_dict_or_struct is not None:
            self.add_user_dict(option_dict_or_struct)
        self.called_function_names = set()

    @os_incore
    def add_user_dict(self, option_dict_or_struct):
        if isinstance(option_dict_or_struct, dict):
            app_user_dict = option_dict_or_struct
        elif isinstance(option_dict_or_struct, OptionStructCore):
            app_user_dict = option_dict_or_struct.enabled_dict
        elif isinstance(option_dict_or_struct, Iterable):
            for d in option_dict_or_struct:
                self.add_user_dict(d)
            return
        else:
            raise ValueError("Invalid option dict")
        self.user_dict = {**self.user_dict, **app_user_dict}

    @os_incore
    def declare(self, key):
        self.set_default(key, _SubOptionPlaceHolder(self, key))

    @os_incore
    def set(self, key, value):
        if isinstance(key, str) and key.startswith("_"):
            raise ValueError("keys starts with _ are reserved")
        assert not isinstance(value, SubOption), \
            "Cannot set with SubOption"
        self.enabled_dict[key] = value
        self.user_dict[key] = value

    @os_incore
    def unset(self, key):
        self.user_dict.pop(key, None)
        self.enabled_dict.pop(key, OptionStructUnsetCacheNone())
        self.unset_set.add(key)

    @os_incore
    def set_default(self, key, default_value):
        if isinstance(key, str) and key.startswith("_"):
            raise ValueError("keys starts with _ are reserved")
        if isinstance(default_value, SubOption):
            # predefined sub options
            assert key not in self.enabled_dict, "should not enable an entry twice"
            if key in self.user_dict:
                self.enabled_dict[key] = default_value.option_struct(self.user_dict[key])
            else:
                self.enabled_dict[key] = default_value.option_struct(dict())
        elif isinstance(default_value, OptionStructCore):
            # inline option struct
            if key in self.user_dict:
                sub_user_dict = self.user_dict[key]
                if isinstance(sub_user_dict, OptionStructCore):
                    sub_user_dict = sub_user_dict.get_dict()
                else:
                    assert isinstance(sub_user_dict, Dict), \
                        "user must specify a dict or nothing if inline OptionStruct is used"
                default_value.add_user_dict(sub_user_dict)
            self.enabled_dict[key] = default_value
        elif isinstance(default_value, _SubOptionPlaceHolder):
            self.enabled_dict[key] = default_value
        else:
            if isinstance(default_value, OptionDefaultValue):
                if key in self.user_dict:
                    the_value = default_value.custom_value(self.user_dict[key])
                else:
                    the_value = default_value.default_value
            else:
                if key in self.user_dict:
                    the_value = self.user_dict[key]
                else:
                    the_value = default_value
            self.enabled_dict[key] = the_value

    @os_incore
    def get_enabled(self, key):
        return self.enabled_dict[key]

    @os_incore
    def __contains__(self, item):
        return item in self.enabled_dict

    @os_incore
    def __getitem__(self, item: str):
        return self.get_enabled(item)

    @os_incore
    def __setitem__(self, key: str, value):
        self.set_default(key, value)

    @os_incore
    def __delitem__(self, key):
        self.unset(key)

    @os_incore
    def __len__(self):
        return len(self.enabled_dict)

    @os_incore
    def __iter__(self):
        return self.enabled_dict.keys()

    @staticmethod
    def _get_namedtuple(d, tuple_type_name):
        from collections import namedtuple
        d = copy(d)
        for k, v in d.items():
            if isinstance(v, OptionStructCore):
                d[k] = v.get_namedtuple(tuple_type_name=(tuple_type_name + "_" + k))
        return namedtuple(tuple_type_name, d.keys())(**d)

    _get_namedtuple_deprecation_warning_printed = False

    @os_incore
    def get_namedtuple(self, tuple_type_name=None):

        if not type(self)._get_namedtuple_deprecation_warning_printed:
            type(self)._get_namedtuple_deprecation_warning_printed = True
            print(
                "WARNING: OptionStruct: get_namedtuple is deprecated. "
                "You are actually getting an EasyDict",
                file=sys.stderr
            )

        return self.get_edict()

        # DISABLE named tuple

        if tuple_type_name is None:
            assert self.option_name is not None, "tuple_type_name must be specified"
            tuple_type_name = self.option_name
        return self._get_namedtuple(self.enabled_dict, tuple_type_name)

    @staticmethod
    def _get_dict(d):
        d = copy(d)
        keys_to_delete = []
        for k, v in d.items():
            if isinstance(v, OptionStructCore):
                d[k] = v.get_dict()
            elif isinstance(v, _SubOptionPlaceHolder):
                keys_to_delete.append(k)
        for k in keys_to_delete:
            d.pop(k)
        return d

    @os_incore
    def get_dict(self):
        return self._get_dict(self.enabled_dict)

    @os_incore
    def get_edict(self):
        from easydict import EasyDict
        return EasyDict(self.get_dict())

    @os_incore
    def _require(self, option_name, is_include):
        assert isinstance(self.option_def, OptionDef), "invalid option_def"
        p = self.option_def.get_optstruct(option_name)
        self.enabled_dict = {**self.enabled_dict, **p.enabled_dict}
        if is_include:
            self.user_dict = {**self.user_dict, **p.user_dict}      # only inherent actual user-specified.
        else:
            self.user_dict = {**self.user_dict, **p.enabled_dict}   # take any required value as user specified
        self.unset_set = self.unset_set.union(p.unset_set)

    @os_incore
    def include(self, option_name):
        self._require(option_name, is_include=True)

    @os_incore
    def require(self, option_name):
        self._require(option_name, is_include=False)

    # @os_incore
    def c_init_call(self, option_name):
        self.called_function_names.clear()
        self.c_include(option_name)

    # @os_incore
    def c_include(self, option_name):
        if option_name in self.called_function_names:
            return
        self.option_def: OptionDef
        self.option_def.run_to_get_optstruct(option_name, self)

    @os_incore
    def finalize(self, error_uneaten=True, display_name: str = None):
        uneaten_keys = set(self.user_dict.keys()) - set(self.enabled_dict.keys()) - self.unset_set
        if len(uneaten_keys) > 0:
            print("WARNING: uneaten options")
            for k in uneaten_keys:
                print("  %s%s: " % ((display_name + '.' if display_name is not None else '') , k), end="")
                print(self.user_dict[k])
            if error_uneaten:
                raise ValueError("uneaten options")

        for k, v in self.enabled_dict.items():
            if isinstance(v, OptionStructCore):
                v.finalize(
                    error_uneaten=error_uneaten,
                    display_name="%s.%s" % (display_name, k) if display_name is not None else display_name
                )


class OptionStruct(OptionStructCore):

    @os_incore
    def __init__(self, option_dict_or_struct=None):
        super().__init__(option_dict_or_struct=option_dict_or_struct)
        self._incore_function = value_class_for_with(False)
        self._core_initialized = True

    @property
    def _incore(self):
        return (
                '_core_initialized' not in dir(self) or
                not self._core_initialized or
                self._incore_function.current_value or
                global_incore_function.current_value
        )

    def __setattr__(self, key, value):
        if self._incore or key in dir(self):
            return super().__setattr__(key, value)
        self[key] = value

    def __getattr__(self, item):
        # if self._incore or item not in self:
        #     raise AttributeError("Attribute does not exist: %s" % item)
        if self._incore:
            raise AttributeError("Attribute does not exist: %s" % item)
        if item not in self:
            self.declare(item)
        return self[item]


class _SubOptionPlaceHolder:
    def __init__(self, option_struct: OptionStructCore, key: str):
        self._option_struct = option_struct
        self._key = key
        self._suboption_struct = None
        self._incore_function = value_class_for_with(False)
        self._initialized = True

    def convert_to_suboption_struct(self):
        with self._incore_function(True):
            if self._suboption_struct is None:
                self._suboption_struct = OptionStruct()
                self._option_struct[self._key] = self._suboption_struct

    @property
    def _incore(self):
        return "_initialized" not in dir(self) or not self._initialized or self._incore_function.current_value

    @property
    def suboption_struct(self) -> OptionStruct:
        self.convert_to_suboption_struct()
        return self._suboption_struct

    def __getitem__(self, item):
        return self.suboption_struct[item]

    def __setitem__(self, key, value):
        self.suboption_struct[key] = value

    def __setattr__(self, key, value):
        if self._incore:
            return super().__setattr__(key, value)
        setattr(self.suboption_struct, key, value)

    def __getattr__(self, item):
        if self._incore:
            raise AttributeError("Attribute does not exist: %s" % item)
        return getattr(self.suboption_struct, item)


class OptionDef:

    finalize_check_env = value_class_for_with(False)

    def __init__(self, user_dict=None, def_cls_or_obj=None, finalize_check=None):
        if user_dict is None:
            user_dict = dict()
        elif isinstance(user_dict, OptionStructCore):
            if finalize_check is None:
                # None means auto become True or False at __init__,
                # it can also be "auto" which means to be decided while calling get_optstruct
                finalize_check = False
            user_dict = user_dict.get_dict()
        else:
            user_dict = copy(user_dict)
        if finalize_check is None:
            finalize_check = True
        if isinstance(finalize_check, str):
            assert finalize_check in ("auto",), "unsupported value for finalize_check"
        self._finalize_check = finalize_check

        for k in list(user_dict.keys()):
            if k.startswith('~') in user_dict:
                del user_dict[k]   # automatically ignore the ~VERSION_CONTROL and other keywords starting with ~

        self._user_dict = user_dict
        self._opts = {}
        if def_cls_or_obj is None:
            self._def_obj = self
        elif isinstance(def_cls_or_obj, type):
            self._def_obj = def_cls_or_obj()
        else:
            self._def_obj = def_cls_or_obj

        # use object level function
        setattr(self, "struct_def", self._struct_def_objlevel)

    def get_optstruct(self, item):
        if item in self._opts:
            return self._opts[item]
        else:
            assert hasattr(self._def_obj, item), "no such method for option definition"
            p = OptionStruct(self._user_dict)
            p.option_def = self
            p.option_name = item + "_options"

            p.c_init_call(item)

            self._opts[item] = p
            return p

    def run_to_get_optstruct(self, item, p: OptionStructCore):
        # opt_def_func = getattr(self._def_obj, item)
        # pt_def_func(p)
        eval("self._def_obj.%s(p)" % item)

    @staticmethod
    def assert_option_struct(p):
        assert isinstance(p, OptionStructCore), "p must be an OptionStruct(Core)"

    def _struct_def_objlevel(self, mem_func_name, finalize_check=True):
        return SubOption(self._def_obj, mem_func_name=mem_func_name, finalize_check=finalize_check)

    @classmethod
    def struct_def(cls, mem_func_name, finalize_check=True):
        return SubOption(cls, mem_func_name=mem_func_name, finalize_check=finalize_check)

    def __getitem__(self, item):
        finalize_check = self._finalize_check
        if finalize_check == "auto":
            finalize_check = self.finalize_check_env.current_value
            set_finalize_env = dummy_class_for_with()
        else:
            set_finalize_env = self.finalize_check_env(finalize_check)
        with set_finalize_env:
            p = self.get_optstruct(item)
            if finalize_check:
                p.finalize(error_uneaten=True)
        return p


class SubOption:
    # hold an OptionDef in the parent OptionDef field

    def __init__(self, def_cls_or_obj, mem_func_name, finalize_check="auto"):
        self._def_cls_or_obj = def_cls_or_obj
        self._mem_func_name = mem_func_name
        self._finalize_check = finalize_check

    def option_struct(self, user_dict):
        # generate option struct from this option definition
        opt_def = OptionDef(
            user_dict=user_dict,
            def_cls_or_obj=self._def_cls_or_obj,
            finalize_check=self._finalize_check
        )
        opt_struct = opt_def[self._mem_func_name]
        return opt_struct


# deprecated ------------------------------------------------------

class OptionStructDef(SubOption):

    _deprecated_warning_printed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not type(self)._deprecated_warning_printed:
            print(
                "DEPRECATION: OptionStructDef is a deprecated naming of the class, please use SubOption",
                file=sys.stderr
            )
            type(self)._deprecated_warning_printed = True
