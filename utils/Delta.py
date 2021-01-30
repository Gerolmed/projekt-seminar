import time
from functools import wraps


def delta(fn):
    @wraps(fn)
    def calculate_delta(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        return ret, elapsed_time

    return calculate_delta
