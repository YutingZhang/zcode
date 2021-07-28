import os
import sys
from .misc import order_preserving_unique
from .io import call_until_success
import subprocess
import site
from typing import Callable, Dict, Optional


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


def robustPopen(*args, **kwargs):
    proc = call_until_success(
        OSError, subprocess.Popen, *args, **kwargs
    )
    return proc


def run_and_get_stdout(cmd: str, input_str: Optional[str] = None) -> (str, str, int):
    proc = robustPopen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE if input_str else subprocess.DEVNULL,
        executable="bash",
        shell=True
    )
    if input_str:
        out, err = proc.communicate(input=input_str.encode())
    else:
        out, err = proc.communicate()
    rc = proc.wait()
    out = out.decode()
    err = err.decode()
    return out, err, rc


def run_system(cmd: str) -> (str, str):
    return call_until_success(
        OSError, os.system, cmd
    )


def _screen_session_name_with_index(session_name: str, index: int = None):
    if index is None:
        return session_name
    return session_name + "-" + str(index)


def screen_session_exists(session_name: str, index: int = None):
    session_name = _screen_session_name_with_index(session_name, index)

    out, _, _ = run_and_get_stdout(f"screen -ls | grep '[0-9]*\\.{session_name}\t'")
    out = out.strip()
    return bool(out)


def screen_create_session(session_name: str, cmd: str, index: int = None, verbose: bool = True):
    cmd = f'export WORKER_SESSION_GROUP_NAME={session_name}; ' + cmd
    session_name = _screen_session_name_with_index(session_name, index)
    cmd = f'export WORKER_SESSION_NAME={session_name}; ' + cmd
    if index is not None:
        cmd = f'export WORKER_SESSION_ID={index}; ' + cmd
    full_cmd = f'screen -S "{session_name}" -d -m bash -c "{cmd}"'
    if verbose:
        print("   Create Screen Session: %s" % session_name)
        print("     +", full_cmd, flush=True)

    run_system(full_cmd)


def screen_quit_session(session_name: str, index: int = None):
    session_name = _screen_session_name_with_index(session_name, index)
    full_cmd = f"screen -S {session_name} -X quit"
    run_system(full_cmd)


def screen_create_session_group(
        session_name: str, num_sessions: int, cmd_gen: Callable[[int], str], verbose: bool = True
):
    if verbose:
        print("Create Screen Session Group: %s" % session_name, flush=True)
    for i in range(num_sessions):
        print(f" - Session {i} / {num_sessions}", flush=True)
        if screen_session_exists(session_name, index=i):
            print("   Existed")
            continue
        cmd = cmd_gen(i)
        screen_create_session(session_name, cmd, index=i, verbose=verbose)


def get_num_cuda_gpus():
    num_gpus, _, _ = run_and_get_stdout("nvidia-smi -L | wc -l")
    num_gpus = num_gpus.strip()
    num_gpus = int(num_gpus)
    return num_gpus


def jobc_watch_create_session_group(
        session_name: str, num_sessions: int,
        jobc_var_dir: str,
        working_dir: Optional[str] = None,
        env_var_gen: Optional[Callable[[int], Dict[str, str]]] = None,
        verbose: bool = True
):
    def _cmd_gen(session_id: int):
        _cmd = []
        if working_dir:
            _cmd.append(f"cd '{working_dir}'")
        if env_var_gen is not None:
            env_vars = env_var_gen(session_id)
            if env_vars:
                for k, v in env_vars.items():
                    _cmd.append(f"export {k}='{v}'")
        if session_id >= 0:
            _session_name = session_name + '-' + str(session_id)
        else:
            _session_name = session_name
        _cmd.append(f"jobc-watch '{jobc_var_dir}' '{_session_name}'")
        return "; ".join(_cmd)
    if num_sessions > 0:
        return screen_create_session_group(
            session_name, num_sessions=num_sessions, cmd_gen=_cmd_gen, verbose=verbose
        )
    else:
        return screen_create_session(session_name, cmd=_cmd_gen(-1))


def jobc_watch_create_session(
        session_name: str, jobc_var_dir: str,
        working_dir: Optional[str] = None,
        env_vars: Dict[str, str] = None,
        verbose: bool = True
):
    return jobc_watch_create_session_group(
        session_name, num_sessions=0, jobc_var_dir=jobc_var_dir, working_dir=working_dir,
        env_var_gen=lambda i: env_vars, verbose=verbose
    )


def _get_jobc_add_cmd(jobc_var_dir: str, name: Optional[str] = None):
    if name:
        return f"jobc-add '{jobc_var_dir}' '{name}'"
    else:
        return f"jobc-add '{jobc_var_dir}'"


def jobc_add_from_script(jobc_var_dir: str, script: str, name: Optional[str] = None, verbose: bool = True):
    cmd = _get_jobc_add_cmd(jobc_var_dir, name)
    if verbose:
        print(" +", cmd, flush=True)
        print("   +", script)
    _, _, ec = run_and_get_stdout(cmd, input_str=script)
    return not bool(ec)


def jobc_add_from_file(jobc_var_dir: str, filename: str, name: Optional[str] = None, verbose: bool = True):
    cmd = f"cat '{filename}' | " + _get_jobc_add_cmd(jobc_var_dir, name)
    if verbose:
        print(" +", cmd, flush=True)
    _, _, ec = run_and_get_stdout(cmd)
    return not bool(ec)
