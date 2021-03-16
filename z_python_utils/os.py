import os
import sys
from .misc import order_preserving_unique
from .io import call_until_success
import subprocess
import site


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

    # system paths
    env_dict['PATH'] = cleanup_env_path_variable(get_env_variable('PATH'))
    env_dict['LD_LIBRARY_PATH'] = cleanup_env_path_variable(get_env_variable('LD_LIBRARY_PATH'))

    # python paths
    python_path = order_preserving_unique(sys.path)
    python_path = [os.path.abspath(x) for x in python_path]
    python_path = order_preserving_unique(python_path)
    python_path = list(filter(lambda x: "/.pycharm_helpers/" not in x, python_path))

    # filter out python standard paths
    cmd4base_python_path = 'PYTHONPATH="" python3 -c "import sys ; [print(a) for a in sys.path];"'
    proc = call_until_success(
        OSError, subprocess.Popen,
        cmd4base_python_path,
        shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
        executable="bash",
    )
    base_python_path = proc.stdout.read().decode("utf-8")
    proc.communicate()
    proc.wait()

    base_python_path = [os.path.abspath(x) for x in filter(bool, base_python_path.split("\n"))]
    preserved_python_path = set(python_path) - set(base_python_path)
    python_path = filter(lambda x: x in preserved_python_path, python_path)

    # filter out more python standard paths
    python_path = filter(lambda x: "/.pycharm_helpers/" not in x, python_path)
    for site_package_path in site.getsitepackages():
        if not site_package_path:
            continue
        site_package_path = os.path.abspath(site_package_path)
        if site_package_path[-1] != '/':
            site_package_path += '/'
        python_path = filter(
            lambda x: not x.startswith(site_package_path) and x != site_package_path[:-1], python_path
        )

    env_dict['PYTHONPATH'] = ":".join(python_path)

    env_dict['PYTHON'] = os.path.abspath(sys.executable)

    return env_dict


def run_and_get_stdout(cmd: str) -> (str, str):
    proc = call_until_success(
        OSError, subprocess.Popen, cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        executable="bash",
        shell=True
    )
    out, err = proc.communicate()
    rc = proc.wait()
    out = out.decode()
    err = err.decode()
    return out, err, rc


def run_system(cmd: str) -> (str, str):
    return call_until_success(
        OSError, os.system, cmd
    )
