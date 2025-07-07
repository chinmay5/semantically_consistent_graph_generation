import time

import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def time_logging(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds, {execution_time/60:.4f} minutes to execute.")
        return result
    return wrapper


def func_info(func):
    def wrapper(*args, **kwargs):
        print(f"Invoking {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} execution complete")
        return result
    return wrapper
