import time
import logging
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"[TIMED] Ran {func.__name__} in {elapsed_time:.2f} seconds")
        return result
    return wrapper_timer