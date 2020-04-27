import inspect
from inspect import isfunction, ismethod
from typing import List
import random


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
    'wrap_obj'
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
    # this is for merges the defintion of another instance

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


def wrap_obj(obj):
    if callable(obj):
        return CallableObjectWrapper(obj)
    else:
        return ObjectWrapper(obj)
