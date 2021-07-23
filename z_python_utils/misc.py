import random
import sys
import os
import io
from typing import Iterable, Optional
from recursive_utils.recursive_utils import *
_ = recursive_apply


def rbool(a):
    try:
        c = bool(a)
    except OSError:
        raise
    except:
        c = True
    return c


def float_to_rational(a, max_denominator=10):
    if a == int(a):
        return int(a), 1

    if a < 0:
        the_sign = -1
        a = -a
    else:
        the_sign = 1

    d = int(a)
    f = a - d

    min_diff = 1
    the_frac = None
    for i in range(2, max_denominator+1):
        for j in range(1, i):
            cur_diff = abs(f - j/i)
            if min_diff > cur_diff:
                min_diff = cur_diff
                the_frac = (j, i)
    output = (int(the_sign * (the_frac[0] + d * the_frac[1])), int(the_frac[1]))
    return output


class _IdAsHash:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return id(self.val)

    def __eq__(self, other):
        return id(self.val) == id(other.val)


def unique_per_id(a: Iterable):
    b = set(_IdAsHash(x) for x in a)
    u = [x.val for x in b]
    return u


# -----------------------------------------------------------------------------
# Based on: https://stackoverflow.com/users/2069807/mrwonderful


def structured_dump(obj, nested_level=0, file=sys.stdout):

    if file is None or file == "str" or file is str:
        str_io = io.StringIO()
        structured_dump(obj, nested_level=nested_level, file=str_io)
        return str_io.getvalue()

    spacing = '   '
    if isinstance(obj, dict):
        print('%s{' % (nested_level * spacing), file=file)
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print('%s%s:' % ((nested_level + 1) * spacing, k), file=file)
                structured_dump(v, nested_level + 1, file=file)
            else:
                print('%s%s: %s' % ((nested_level + 1) * spacing, k, v), file=file)
        print('%s}' % (nested_level * spacing), file=file)
    elif isinstance(obj, list):
        print('%s[' % ((nested_level) * spacing), file=file)
        for v in obj:
            if isinstance(v, Iterable) and not isinstance(v, str):
                structured_dump(v, nested_level + 1, file=file)
            else:
                print('%s%s' % ((nested_level + 1) * spacing, v), file=file)
        print('%s]' % ((nested_level) * spacing), file=file)
    else:
        print('%s%s' % (nested_level * spacing, obj), file=file)


def z_python_utils_dir():
    return os.path.dirname(__file__)


def zcode_dir():
    return os.path.dirname(z_python_utils_dir())


pyu_dir = z_python_utils_dir


def order_preserving_unique(a):
    visited = set()
    b = []
    for x in a:
        if x not in visited:
            b.append(x)
            visited.add(x)
    return b


def online_shuffle(a: Iterable, window_size: int, random_generator: Optional[random.Random] = None):

    if window_size == 0:
        return a

    shuffle = (
        random_generator.shuffle if random_generator is not None
        else random.shuffle
    )

    if window_size < 0:     # shuffle all together
        a = list(a)
        shuffle(a)
        return a

    window_randint = (
        random_generator.randint if random_generator is not None
        else random.randint
    )
    b = dict()
    s = set(range(window_size))
    for v in a:
        i = s.pop()
        b[i] = v
        if not s:
            j = window_randint(0, window_size)
            s.add(j)
            yield b.pop(j)

    rest_values = list(b.values())
    shuffle(rest_values)
    for v in rest_values:
        yield v
