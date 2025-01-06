import itertools
import os
import time
from typing import Any, Dict, List, Tuple, Union
import shutil
import traceback
import sys
from contextlib import contextmanager
import signal
import numpy as np
import resource

from utils import (
    # create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


class TimeoutException(Exception):
    pass

class TimeLimit:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        # Set the signal handler and the alarm
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        # Disable the alarm
        signal.alarm(0)
        # If an exception was raised, we need to propagate it
        if exc_type is TimeoutException:
            return False  # Propagate the exception
        return True  # Suppress other exceptions

    def handle_timeout(self, signum, frame):
        raise TimeoutException("Time limit exceeded")

# Context manager to redirect stdout and stderr
@contextmanager
def swallow_io():
    class DevNull:
        def write(self, _): pass
        def flush(self): pass

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = DevNull(), DevNull()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

# Example implementation of a context manager to temporarily change and restore directory
@contextmanager
def temp_directory():
    original_dir = os.getcwd()
    try:
        temp_dir = os.path.join(original_dir, "tempdir")
        os.makedirs(temp_dir, exist_ok=True)
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

def unsafe_execute(
    entry_point: str,
    code: str,
    inputs: List,
    expected: List,
    time_limits: List,
    atol: float,
    stat,  # Value
    details,  # Array
    progress,  # Value
    debug=False
    
):
    if debug:
        print("Starting execution")

    exec_globals = {}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_chdir = os.chdir
    og_putenv = os.putenv
    og_resource = sys.modules["resource"]


    maximum_memory_bytes = 1 * 1024 * 1024 * 1024
    reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
    try:
        with swallow_io():
            exec(code, exec_globals)
            
            fn = exec_globals.get(entry_point, None)
            if fn is None:
                if debug:
                    print(f"Function {entry_point} not found.")
                stat = 0
                return stat, details

        for i, inp in enumerate(inputs):
            try:
                with TimeLimit(time_limits[i]):
                    with swallow_io():
                        out = fn(*inp)
                exp = expected[i]
                exact_match = out == exp
                if atol == 0 and is_float(exp):
                    atol = 1e-6  # enforce atol for float comparison
                if not exact_match and atol != 0:
                    assert type(out) == type(exp)
                    if isinstance(exp, (list, tuple)):
                        assert len(out) == len(exp)
                    assert np.allclose(out, exp, rtol=1e-07, atol=atol)
                else:
                    assert exact_match
            except Exception as e:
                if debug:
                    print(f"Exception during test {i}: {e}")
                # traceback.print_exc()
                details[i] = False
                progress += 1
                continue

            details[i] = True
            progress += 1

        stat = 1

    except Exception as e:
        if debug:
            print("Error during execution")
        traceback.print_exc()
        stat = 1

    finally:
        # Restore original state if necessary
        sys.stdout, sys.stderr = original_stdout, original_stderr
        os.chdir = original_chdir
        os.putenv = og_putenv
        sys.modules["resource"] = og_resource

        # Any additional cleanup can go here

    return stat, details



if __name__ == "__main__":
    code = '''def fib4(n: int) -> int:
    if n == 0: return 0
    elif n == 1: return 0
    elif n == 2: return 2
    elif n == 3: return 0

    a, b, c, d = 0, 0, 2, 0
    for _ in range(4, n + 1):
        a, b, c, d = b, c, d, a + b + c + d

    return d
    '''

    _SUCCESS = 1
    _FAILED = 0
    
    inputs = [(0,), (1,), (5,)]
    progress = 0#Value("i", 0)
    stat = 0#Value("i", _UNKNOWN)
    details = [False for _ in range(len(inputs))]

    
    _, y = unsafe_execute(entry_point="fib4", code=code, inputs=inputs, expected=[0, 0, 4], time_limits=[60, 60, 60], atol=1e-6, stat=stat, details=details, progress=progress)
    print(y)
    