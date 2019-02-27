from collections import namedtuple, Iterable


def first_in_dict(d):
    for k in d:
        return d[k]


def convert2set(a):
    if isinstance(a, set):
        return a
    if a is None:
        return set()
    if isinstance(a, list) or isinstance(a, tuple):
        return set(a)
    else:
        return {a}


def dict2namedtuple(d, tuple_name=None):
    if tuple_name is None:
        tuple_name = "lambda_namedtuple"
    return namedtuple(tuple_name, d.keys())(**d)


def canonicalize_slice(s, end=None):
    return slice(
        0 if s.start is None else s.start,
        end if s.end is None else s.end,
        1 if s.step is None else s.step,
    )


def robust_index(a, i):
    if a is None:
        return None
    else:
        return a[i]


def ordered_unique(seq: Iterable):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def even_partition(total_num, partition_num):
    smaller_size = total_num//partition_num
    larger_num = total_num - smaller_size * partition_num
    smaller_num = partition_num - larger_num
    p = [smaller_size+1] * larger_num + [smaller_size] * smaller_num
    return p


def even_partition_indexes(total_num, partition_num):
    subset_num = even_partition(total_num, partition_num)
    epi = list()
    for i in range(len(subset_num)):
        epi.extend([i] * subset_num[i])
    return epi


class FlagEater:

    def __init__(self, flags=None, default_value=False, default_pos_value=True):
        if isinstance(flags, FlagEater):
            self._flags = flags._flags
            self._default_value = flags._default_value
            self._default_pos_value = flags._default_pos_value
        else:
            self._default_value = default_value
            self._default_pos_value = default_pos_value
            self._flags = dict()
            self.add_flags(flags)

    def add_flags(self, flags):
        if flags is None:
            return
        if isinstance(flags, dict):
            self._flags = {**self._flags, **flags}
        elif isinstance(flags, (list, tuple, set)):
            for k in flags:
                self._flags[k] = self._default_pos_value
        else:
            self._flags[flags] = self._default_pos_value

    def pop(self, key):
        return self._flags.pop(key, self._default_value)

    def finalize(self):
        assert not self._flags, "Not all flags are eaten"


class IntervalSearch:
    def __init__(self, split_points, leftmost_val=0):
        self._sp = sorted(split_points)
        self._leftmost_val = leftmost_val

    def __getitem__(self, item):
        a = self._sp

        if not a:
            return 0

        n = len(a)

        left = 0
        right = n + 1
        left_val = self._leftmost_val

        mid = (left+right) // 2
        while mid != left:
            b = a[mid-1]
            if b <= item:
                left = mid
                left_val = b
            else:
                right = mid
            mid = (left + right) // 2

        return mid, (item-left_val)  # (interval_id, loc in interval)

    def __len__(self):
        return len(self._sp)

    @property
    def splitting_points(self):
        return self._sp

    @property
    def leftmost_val(self):
        return self._leftmost_val
