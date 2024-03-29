import contextlib
import inspect
from inspect import isfunction, ismethod
from typing import List, Iterable, Optional, Callable, Any, Sized
import threading
import uuid
import random
import os
import sys
import importlib


__all__ = [
    'ClassPropertyDescriptor',
    'classproperty',
    'get_new_members',
    'update_class_def_per_ref',
    'link_with_instance',
    'ValuedContext',
    'ValueForWithContext',
    'value_class_for_with',
    'dummy_class_for_with',
    'ClsWithCustomInit',
    'NonSelfAttrDoesNotExist',
    'get_nonself_attr_for_type',
    'TagClass',
    'CallableObjectWrapper',
    'ObjectWrapper',
    'wrap_obj',
    'SizedWrapperOfIterable',
    'get_class_fullname',
    'load_obj_from_file',
    'ObjectPool',
    'DummyEverything',
    'global_registry',
    'set_path_for_orphan_object_if_needed',
]


class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def get_new_members(inherent_class, base_class):
    assert issubclass(inherent_class, base_class), "must be inherent class and base class"
    target_mem = dict(inspect.getmembers(inherent_class))
    base_mem = dict(inspect.getmembers(base_class))
    new_mem = list(filter(lambda a: a[0] not in base_mem or base_mem[a[0]] is not a[1], target_mem.items()))
    return new_mem


def update_class_def_per_ref(target_class, ref_class, target_base_class=None, ref_base_class=None):
    # this is for merge two class definition branch into one
    if target_base_class is None:
        target_base_class = object

    reserved_mem_names = set(dict(get_new_members(target_class, target_base_class)).keys())

    if ref_base_class is None:
        ref_base_class = object

    new_mem = dict(get_new_members(ref_class, ref_base_class))
    override_mem_names = set(new_mem.keys()) - set(reserved_mem_names)

    for k in override_mem_names:
        setattr(target_class, k, new_mem[k])


def link_with_instance(self, another):
    # this is for merges the definition of another instance

    my_attr_dict = list(filter(
        lambda kk: not (kk.startswith('__') and kk.endswith('__')),
        dir(self)))

    for k in dir(another):
        if k.startswith('__') and k.endswith('__'):
            continue
        if k in my_attr_dict:
            continue
        v = getattr(another, k)
        if not (isfunction(v) or ismethod(v) or callable(v)):
            continue
        setattr(
            self, k, (lambda vv: lambda *arg, **kwargs: vv(*arg, **kwargs))(v))


class ValuedContext:
    def __init__(self, *args, **kwargs):
        pass


class _ValueForWith:

    _context_pool = dict()

    def __init__(self, init_value=None, value_for_with_context_class=None):
        super().__init__()
        class_id = random.random()
        while class_id in type(self)._context_pool:
            class_id = random.random()
        self._class_id = class_id
        self._value_stack = [init_value]
        type(self)._context_pool[class_id] = self._value_stack
        if value_for_with_context_class is None:
            value_for_with_context_class = ValueForWithContext
        self._value_for_with_context_class = value_for_with_context_class

    def __call__(self, value):
        return self._value_for_with_context_class(
            value_stack=self._value_stack,
            value=value
        )

    @property
    def current_value(self):
        return self.value_stack[-1]

    @current_value.setter
    def current_value(self, val):
        self.value_stack[-1] = val

    @property
    def value_stack(self) -> List:
        return self._value_stack


class ValueForWithContext(ValuedContext):

    def __init__(self, value_stack, value):
        super().__init__()
        self._value_stack = value_stack
        self._value = value

    def __enter__(self):
        self.value_stack.append(self._value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.value_stack.pop()
        return False

    @property
    def current_value(self):
        return self.value_stack[-1]

    @current_value.setter
    def current_value(self, val):
        self.value_stack[-1] = val

    @property
    def value_stack(self) -> List:
        return self._value_stack


value_class_for_with = _ValueForWith


class dummy_class_for_with(ValuedContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class ClsWithCustomInit:

    init_scope = value_class_for_with()

    def __init__(self):
        self.args = self._args_dummy
        if getattr(type(self), "init") is not ClsWithCustomInit.init:
            custom_init = getattr(self, 'init')
            custom_init_need_no_args = not inspect.signature(custom_init).parameters
            if custom_init_need_no_args:
                self.__call_custom_init_func(custom_init)
            else:
                self._custom_init = custom_init
                self.args = self._args

    def __call_custom_init_func(self, func, *args, **kwargs):
        with self.init_scope(self):
            return self._call_custom_init_func(func, *args, **kwargs)

    def _call_custom_init_func(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _args(self, *args, **kwargs):
        delattr(self, "args")
        self.__call_custom_init_func(self._custom_init, *args, **kwargs)
        delattr(self, "_custom_init")
        return self

    def _args_dummy(self):
        delattr(self, "args")
        return self

    def init(self, *args, **kwargs):
        pass

    @property
    def _is_custom_initialized(self):
        return not hasattr(self, "_custom_init")

    def _assert_custom_initialized(self):
        assert self._is_custom_initialized, "need to use args() to specify extra arguments"

    @property
    def _is_in_custom_init(self):
        return self in self.init_scope.value_stack


class NonSelfAttrDoesNotExist:
    pass


def get_nonself_attr_for_type(cls: type, name, target_type=None):
    assert hasattr(cls, name), "no such attr"
    a0 = getattr(cls, name)
    mro = inspect.getmro(cls)
    for t in mro:
        if target_type is not None and not issubclass(t, target_type):
            continue
        if issubclass(cls, t):
            continue
        if hasattr(t, name):
            a = getattr(t, name)
            if a is not a0:
                return a
    raise NonSelfAttrDoesNotExist


class TagClass:

    name = None

    def __init__(self):
        assert type(self) is not TagClass, "cannot instantiate TagClass directly, must use an inherent class"

    def __eq__(self, other):
        return isinstance(other, type(self)) or (other is type(self))

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        if self.name is None:
            return "[Tag:%s]" % type(self).__name__
        else:
            return self.name

    def __str__(self):
        return self.__repr__()


class ObjectWrapper:
    def __init__(self, obj):
        self._obj = obj
        # my_members = dir(self)
        # for member_name in dir(self._obj):
        #     if member_name in my_members or not member_name.startswith('__'):
        #         continue
        #     the_member = object.__getattribute__(self._obj, member_name)
        #     if not callable(the_member):
        #         continue
        #     if member_name == '__getattribute__':
        #         continue
        #     setattr(self, member_name, the_member)
        self._initialized = True

    def get_wrapped_object(self):
        return self._obj

    def __getattr__(self, item):
        if hasattr(self._obj, item):
            return getattr(self._obj, item)
        raise AttributeError('No such attribute: %s' % item)

    # def __getattribute__(self, item):
    #     try:
    #         return object.__getattribute__(self, item)
    #     except AttributeError:
    #         pass
    #     obj = object.__getattribute__(self, '_obj')
    #     if hasattr(obj, item):
    #         return getattr(self._obj, item)
    #     raise AttributeError('No such attribute: %s' % item)

    def __setattr__(self, key, value):
        if '_initialized' not in dir(self) or not self._initialized:
            super().__setattr__(key, value)
            return
        setattr(self._obj, key, value)


class CallableObjectWrapper(ObjectWrapper):

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)


class SubscriptableObjectWrapper(ObjectWrapper):

    def __getitem__(self, item):
        return self._obj[item]


class CallableSubscriptableObjectWrapper(ObjectWrapper):

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getitem__(self, item):
        return self._obj[item]


class SubscriptableSizedObjectWrapper(ObjectWrapper):

    def __getitem__(self, item):
        return self._obj[item]

    def __len__(self, item):
        return self._obj[item]


class CallableSubscriptableSizedObjectWrapper(ObjectWrapper):

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getitem__(self, item):
        return self._obj[item]

    def __len__(self, item):
        return self._obj[item]


def wrap_obj(obj):
    is_callable = callable(obj)
    is_subscriptable = hasattr(obj, '__getitem__')
    is_sized = isinstance(obj, Sized)
    if is_callable:
        if is_subscriptable:
            if is_sized:
                return CallableSubscriptableSizedObjectWrapper(obj)
            else:
                return CallableSubscriptableObjectWrapper(obj)
        else:
            return CallableObjectWrapper(obj)
    else:
        if is_subscriptable:
            if is_sized:
                return SubscriptableSizedObjectWrapper(obj)
            else:
                return SubscriptableObjectWrapper(obj)
        else:
            return ObjectWrapper(obj)


class DummyEverything:

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return DummyEverything()

    def __setattr__(self, key, value):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getitem__(self, item):
        return DummyEverything()

    def __setitem__(self, key, value):
        pass

    def keys(self):
        yield from iter(self)

    def values(self):
        for k in iter(self):
            yield self[k]

    def items(self):
        yield from zip(self.keys(), self.values())

    def __enter__(self):
        return DummyEverything()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __len__(self):
        return 1

    def __iter__(self):
        yield DummyEverything()


class SizedWrapperOfIterable:
    def __init__(self, obj: Iterable, n: int):
        self._obj = obj
        self._n = n

    def __len__(self):
        return self._n

    def __iter__(self):
        return iter(self._obj)


class _ObjectPoolDefaultValueNotSet:
    pass


class ObjectPool:

    def __init__(self):
        self._d = None
        self._d_lock = threading.Lock()
        self._pid = -1
        self._lock = threading.Lock()

    @property
    def d(self):
        with self._d_lock:
            my_pid = os.getpid()
            if self._pid != my_pid:
                self._pid = my_pid
                self._d = dict()
            return self._d

    @staticmethod
    def generate_uuid():
        return str(os.getpid()) + "-" + str(uuid.uuid4())

    def add(self, obj):
        with self._lock:
            object_id = self.generate_uuid()
            while object_id in self.d:
                object_id = self.generate_uuid()
            self.d[object_id] = (obj, os.getpid())
            return object_id

    def pop(self, object_id, default_value: Any = _ObjectPoolDefaultValueNotSet()):
        with self._lock:
            if not isinstance(default_value, _ObjectPoolDefaultValueNotSet) and object_id not in self.d:
                return default_value
            obj, the_pid = self.d[object_id]
            if the_pid == os.getpid():
                self.d.pop(object_id)
            return obj

    def get(self, object_id):
        with self._lock:
            return self.d[object_id][0]


def get_class_fullname(a, use_filename_for_main: bool = False):
    if not inspect.isclass(a):
        a = type(a)
    mod = inspect.getmodule(a)
    module_name = mod.__name__
    if use_filename_for_main:
        need_to_use_filename = False
        if module_name == "__main__":
            need_to_use_filename = True
        elif hasattr(a, "____LOADED_AS_ORPHAN____") and getattr(a, "____LOADED_AS_ORPHAN____"):
            # backdoor to force using filename
            need_to_use_filename = True
        if need_to_use_filename:
            module_name = os.path.abspath(mod.__file__) + "#"
    class_name = a.__name__
    return module_name + "." + class_name


@contextlib.contextmanager
def set_path_for_orphan_object_if_needed(a):
    if hasattr(a, "____LOADED_AS_ORPHAN____") and getattr(a, "____LOADED_AS_ORPHAN____"):
        mod = inspect.getmodule(a)
        mod_dir = os.path.dirname(mod.__file__)
    else:
        mod_dir = None

    need_add_dir = False
    if mod_dir:
        need_add_dir = mod_dir not in sys.path
        sys.path.insert(0, mod_dir)

    yield

    if need_add_dir:
        sys.path.remove(mod_dir)


def load_obj_from_file(obj_spec: str, package_dirs: Optional[List[str]] = None):
    if not package_dirs:
        package_dirs = []

    if ":" in obj_spec:
        obj_spec_fn_obj = obj_spec.split(':')
        obj_name = obj_spec_fn_obj[-1]
        py_filename = os.path.abspath(":".join(obj_spec_fn_obj[:-1]))
        if os.path.isfile(py_filename):
            module_name = os.path.splitext(os.path.basename(py_filename))[0]
            module_parent_dir = os.path.dirname(py_filename)
            obj_spec = module_name + "." + obj_name
        elif os.path.isdir(py_filename):
            module_parent_dir = py_filename
            obj_spec = obj_name
        else:
            raise AssertionError("cannot find file or dir: " + py_filename)

        package_dirs.append(module_parent_dir)

    for pp in package_dirs:
        pp = os.path.abspath(pp)
        if pp not in sys.path:
            sys.path.insert(0, pp)

    obj_spec_chain = obj_spec.split('.')
    obj_name = obj_spec_chain[-1]
    module_name = ".".join(obj_spec_chain[:-1])
    mod = importlib.import_module(module_name)
    obj = getattr(mod, obj_name)
    return obj


class _NotGiven:
    pass


class GlobalRegistry:

    _registry = dict()

    @classmethod
    def registry(cls):
        return cls._registry

    @classmethod
    def get_registry(cls, identifier, prefix=_NotGiven):
        if prefix is _NotGiven and isinstance(identifier, tuple):
            prefix, identifier = identifier
        else:
            if prefix is _NotGiven:
                prefix = None
        r = cls._registry[prefix, identifier]
        return r[0]

    @staticmethod
    def canonicalize_prefix_and_identifier(identifier, prefix=_NotGiven):
        if prefix is _NotGiven and isinstance(identifier, tuple):
            prefix, identifier = identifier
        else:
            if prefix is _NotGiven:
                prefix = None
        return prefix, identifier

    @classmethod
    def set_registry(cls, identifier, obj, prefix=_NotGiven):
        prefix, identifier = cls.canonicalize_prefix_and_identifier(identifier=identifier, prefix=prefix)
        cls.register(obj, identifier=identifier, prefix=prefix, change_count=(prefix, identifier) not in cls._registry)
        return cls.get_registry(identifier=identifier, prefix=prefix)

    @classmethod
    def register(cls, obj, identifier=None, prefix=None, change_count: bool = True):
        prefix, identifier = cls.canonicalize_prefix_and_identifier(identifier=identifier, prefix=prefix)
        if (prefix, identifier) not in cls._registry:
            cls._registry[prefix, identifier] = [None, 0]
        if obj is not None:
            cls._registry[prefix, identifier][0] = obj
        if change_count:
            cls._registry[prefix, identifier][1] += 1
        return prefix, identifier

    @classmethod
    def deregister(cls, prefixed_identifier, change_count: bool = True):
        prefix, identifier = prefixed_identifier
        if (prefix, identifier) not in cls._registry:
            return
        if change_count:
            cls._registry[prefix, identifier][1] -= 1
        if cls._registry[prefix, identifier][1] <= 0:
            cls._registry.pop((prefix, identifier))

    @classmethod
    def fully_deregister(cls, prefixed_identifier):
        if prefixed_identifier in cls._registry:
            cls._registry.pop(prefixed_identifier)

    def __init__(
            self, obj, identifier=None, prefix=None,
            change_count_at_enter: bool = True, change_count_at_exit: bool = True
    ):
        self.obj = obj
        self.identifier = identifier
        self.prefix = prefix
        self.change_count_at_enter = change_count_at_enter
        self.change_count_at_exit = change_count_at_exit

    def __enter__(self):
        self.prefix, self.identifier = type(self).register(
            self.obj, self.identifier, self.prefix, change_count=self.change_count_at_enter
        )
        return self.prefix, self.identifier

    def __exit__(self, exc_type, exc_val, exc_tb):
        type(self).deregister((self.prefix, self.identifier), change_count=self.change_count_at_exit)


class _GlobalRegistryInterface:
    def __init__(self):
        self.register = GlobalRegistry.register
        self.deregister = GlobalRegistry.deregister

    def __call__(self, *args, **kwargs):
        return GlobalRegistry(*args, **kwargs)

    def __getitem__(self, item):
        return GlobalRegistry.get_registry(item)

    def __setitem__(self, key, value):
        return GlobalRegistry.set_registry(identifier=key, obj=value)

    def __delitem__(self, key):
        return GlobalRegistry.fully_deregister(key)

    @staticmethod
    def new_registry(obj=None):
        return GlobalRegistry.register(obj=obj)

    @property
    def all(self) -> dict:
        return GlobalRegistry.registry()

    def has(self, *args) -> bool:
        return GlobalRegistry.canonicalize_prefix_and_identifier(*args) in self.all

    def keys(self):
        return self.all.keys()

    def values(self):
        for k in self.keys():
            yield self[k]

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def __len__(self):
        return len(self.all)

    @staticmethod
    def clear():
        GlobalRegistry.registry().clear()


global_registry = _GlobalRegistryInterface()

