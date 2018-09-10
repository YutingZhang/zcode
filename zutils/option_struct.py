from collections import namedtuple, Iterable, Sequence
from copy import copy
from easydict import EasyDict as edict
from py_utils import value_class_for_with, dummy_class_for_with


class OptionStruct_UnsetCacheNone:
    pass


class OptionStruct:

    def __init__(self, option_dict_or_struct=None):
        self.user_dict = {}
        self.enabled_dict = {}
        self.unset_set = set()
        self.option_def = None
        self.option_name = None
        if option_dict_or_struct is not None:
            self.add_user_dict(option_dict_or_struct)

    def add_user_dict(self, option_dict_or_struct):
        if isinstance(option_dict_or_struct, dict):
            app_user_dict = option_dict_or_struct
        elif isinstance(option_dict_or_struct, OptionStruct):
            app_user_dict = option_dict_or_struct.enabled_dict
        elif isinstance(option_dict_or_struct, Iterable):
            for d in option_dict_or_struct:
                self.add_user_dict(d)
            return
        else:
            raise ValueError("Invalid option dict")
        self.user_dict = {**self.user_dict, **app_user_dict}

    def set(self, key, value):
        if isinstance(key, str) and key.startswith("_"):
            raise ValueError("keys starts with _ are reserved")
        assert not isinstance(value, OptionStructDef), \
            "Cannot set with OptionStructDef"
        self.enabled_dict[key] = value
        self.user_dict[key] = value

    def unset(self, key):
        self.user_dict.pop(key, None)
        self.enabled_dict.pop(key, OptionStruct_UnsetCacheNone())
        self.unset_set.add(key)

    def set_default(self, key, default_value):
        if isinstance(key, str) and key.startswith("_"):
            raise ValueError("keys starts with _ are reserved")
        if isinstance(default_value, OptionStructDef):
            assert key not in self.enabled_dict, "should not enable an entry twice"
            if key in self.user_dict:
                self.enabled_dict[key] = default_value.option_struct(self.user_dict[key])
            else:
                self.enabled_dict[key] = default_value.option_struct(dict())
            return
        if key in self.user_dict:
            self.enabled_dict[key] = self.user_dict[key]
        else:
            self.enabled_dict[key] = default_value

    def get_enabled(self, key):
        return self.enabled_dict[key]

    def __getitem__(self, item):
        return self.get_enabled(item)

    def __setitem__(self, key, value):
        self.set_default(key, value)

    @staticmethod
    def _get_namedtuple(d, tuple_type_name):
        d = copy(d)
        for k, v in d.items():
            if isinstance(v, OptionStruct):
                d[k] = v.get_namedtuple(tuple_type_name=(tuple_type_name + "_" + k))
        return namedtuple(tuple_type_name, d.keys())(**d)

    def get_namedtuple(self, tuple_type_name=None):
        if tuple_type_name is None:
            assert self.option_name is not None, "tuple_type_name must be specified"
            tuple_type_name = self.option_name
        return self._get_namedtuple(self.enabled_dict, tuple_type_name)

    @staticmethod
    def _get_dict(d):
        d = copy(d)
        for k, v in d.items():
            if isinstance(v, OptionStruct):
                d[k] = v.get_dict()
        return d

    def get_dict(self):
        return self._get_dict(self.enabled_dict)

    def get_edict(self):
        return edict(self.get_dict())

    def _require(self, option_name, is_include):
        assert isinstance(self.option_def, OptionDef), "invalid option_def"
        p = self.option_def.get_optstruct(option_name)
        self.enabled_dict = {**self.enabled_dict, **p.enabled_dict}
        if is_include:
            self.user_dict = {**self.user_dict, **p.user_dict}      # only inherent actual user-specified.
        else:
            self.user_dict = {**self.user_dict, **p.enabled_dict}   # take any required value as user specified
        self.unset_set = self.unset_set.union(p.unset_set)

    def include(self, option_name):
        self._require(option_name, is_include=True)

    def require(self, option_name):
        self._require(option_name, is_include=False)

    def finalize(self, error_uneaten=True):
        uneaten_keys = set(self.user_dict.keys()) - set(self.enabled_dict.keys()) - self.unset_set
        if len(uneaten_keys) > 0:
            print("WARNING: uneaten options")
            for k in uneaten_keys:
                print("  %s: " % k, end="")
                print(self.user_dict[k])
            if error_uneaten:
                raise ValueError("uneaten options")


class OptionDef:

    finalize_check_env = value_class_for_with(False)

    def __init__(self, user_dict=None, def_cls_or_obj=None, finalize_check=None):
        if user_dict is None:
            user_dict = dict()
        elif isinstance(user_dict, OptionStruct):
            if finalize_check is None:
                # None means auto become True or False at __init__,
                # it can also be "auto" which means the be decided while calling get_optstruct
                finalize_check = False
            user_dict = user_dict.get_dict()
        else:
            user_dict = copy(user_dict)
        if finalize_check is None:
            finalize_check = True
        if isinstance(finalize_check, str):
            assert finalize_check in ("auto",), "unsupported value for finalize_check"
        self._finalize_check = finalize_check

        if "_VERSION_CONTROL" in user_dict:
            del user_dict["_VERSION_CONTROL"]   # automatically ignore the _VERSION_CONTROL keyword

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
            # opt_def_func = getattr(self._def_obj, item)
            # pt_def_func(p)
            eval("self._def_obj.%s(p)" % item)
            self._opts[item] = p
            return p

    @staticmethod
    def assert_option_struct(p):
        assert isinstance(p, OptionStruct), "p must be an OptionStruct"

    def _struct_def_objlevel(self, mem_func_name, finalize_check=True):
        return OptionStructDef(self._def_obj, mem_func_name=mem_func_name, finalize_check=finalize_check)

    @classmethod
    def struct_def(cls, mem_func_name, finalize_check=True):
        return OptionStructDef(cls, mem_func_name=mem_func_name, finalize_check=finalize_check)

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


class OptionStructDef:

    def __init__(self, def_cls_or_obj, mem_func_name, finalize_check="auto"):
        self._def_cls_or_obj = def_cls_or_obj
        self._mem_func_name = mem_func_name
        self._finalize_check = finalize_check

    def option_struct(self, user_dict):
        opt_def = OptionDef(
            user_dict=user_dict,
            def_cls_or_obj=self._def_cls_or_obj,
            finalize_check=self._finalize_check
        )
        opt_struct = opt_def[self._mem_func_name]
        return opt_struct
