import os
import sys
import site
from .misc import order_preserving_unique
from .io import call_until_success


def self_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss
    return mem_usage


def get_env_variable(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    return ""


def cleanup_env_path_variable(path_var: str):
    return ':'.join(order_preserving_unique(path_var.split(':')))


def system_env_dict() -> dict:
    env_dict = dict()
    env_dict['PATH'] = cleanup_env_path_variable(get_env_variable('PATH'))
    env_dict['LD_LIBRARY_PATH'] = cleanup_env_path_variable(get_env_variable('LD_LIBRARY_PATH'))
    python_path = order_preserving_unique(sys.path)
    python_path = filter(lambda x: "/.pycharm_helpers/" not in x, python_path)
    for site_package_path in site.getsitepackages():
        if not site_package_path:
            continue
        if site_package_path[-1] != '/':
            site_package_path += '/'
        python_path = filter(
            lambda x: not x.startswith(site_package_path) and x != site_package_path[:-1], python_path
        )
    env_dict['PYTHONPATH'] = ":".join(python_path)
    return env_dict
