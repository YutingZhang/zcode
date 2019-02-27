from copy import copy
import os
import sys
import subprocess
import traceback
from .time import timestamp_for_filename
from .functions import call_until_success


# get git versioning info
def git_version_dict(dir_path=None, record_traceback=None):
    """
    This function returns the status of

    :param dir_path: the directory to get git status information. It can be any subfolder inside a git repository.
                    If not specified, use current working directory.
    :param record_traceback: bool. If True, the function records changes that affect files that in the call stack,
                    when calling this function.
                    When not specified, record_traceback = True  if dir_path == current working directory;
                                        record_traceback = False, otherwise,
    :return: {
        "DATE_TIME": date time string,
        "ARGV": arguments to run the top-level script,
        "HEAD_CHECKSUM": the commit checksum for the current branch HEAD,
        "DIFF_WITH_HEAD": any diff with HEAD for tracked files,
        "TRACEBACK_FILE_CACHE": (only available when record_traceback is True)
            record of the full files in the call stack if a file is different from the git commit or not tracked by git
    }
    """
    if dir_path is None or not dir_path:
        dir_path = os.getcwd()
        prefixed_command = ""
    else:
        dir_path = os.path.abspath(dir_path)
        prefixed_command = "cd '%s'; " % dir_path

    if record_traceback is None:
        record_traceback = os.getcwd() == dir_path

    gv_dict = dict()

    gv_dict["DATE_TIME"] = timestamp_for_filename()

    gv_dict["ARGV"] = copy(sys.argv)

    is_git = call_until_success(
        OSError, subprocess.call,
        "%sgit branch" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        executable="bash",
    ) == 0
    if not is_git:
        return gv_dict

    # git checksum of HEAD
    proc = call_until_success(
        OSError, subprocess.Popen,
        "%sgit rev-parse HEAD" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
        executable="bash",
    )
    head_checksum = proc.stdout.read().decode("utf-8")
    proc.communicate()
    proc.wait()
    gv_dict["HEAD_CHECKSUM"] = head_checksum.strip()

    # git diff with HEAD
    proc = call_until_success(
        OSError, subprocess.Popen,
        "%sgit diff" % prefixed_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
        executable="bash",
    )
    diff_with_head = proc.stdout.read().decode("utf-8")
    proc.communicate()
    proc.wait()
    if not diff_with_head.strip():
        diff_with_head = None
    gv_dict["DIFF_WITH_HEAD"] = diff_with_head

    # user files in trackback
    if record_traceback:
        file_cache = []
        for a in traceback.extract_stack():
            fn = a.filename
            if not os.path.exists(fn):
                # not a file
                continue
            if not fn.startswith(dir_path):
                # not a file in the specified dir
                continue

            is_tracked = call_until_success(
                OSError, subprocess.call,
                "%sgit ls-files --error-unmatch '%s'" % (prefixed_command, fn),
                shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                executable="bash",
            ) == 0
            is_different = True

            if is_tracked:
                proc = call_until_success(
                    OSError, subprocess.Popen,
                    "%sgit diff '%s'" % (prefixed_command, fn),
                    shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    executable="bash",
                )
                diff_for_this_file = proc.stdout.read().decode("utf-8")
                proc.communicate()
                proc.wait()
                diff_for_this_file = diff_for_this_file.strip()
                is_different = bool(diff_for_this_file)

            if is_different:
                # if different in git, save the file content
                with open(fn, 'r') as f:
                    c = f.read()
                    file_cache.append((fn, c))
            else:
                # if clean in git
                file_cache.append((fn, None))

        gv_dict["TRACEBACK_FILE_CACHE"] = file_cache

    return gv_dict
