import time
import neptune

from typing import Callable


def obtrusive_call(func: Callable):
    MAX_CALL = 10
    MAX_DELAY = 10

    def wrapper(*args, **kwargs):
        exception = None

        for idx in range(MAX_CALL):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Try run {func.__name__}. Attempt {idx}/{MAX_CALL} failed.")
                if idx < MAX_CALL - 1:
                    print("Let's try again...")
                    print()
                    time.sleep(min(idx, MAX_DELAY))
                else:
                    print("Limit of attempt has been exceed. Call failed.")
                    print()
                    exception = e

        raise exception

    return wrapper


@obtrusive_call
def init(*args, **kwargs):
    return neptune.init(*args, **kwargs)


@obtrusive_call
def create_experiment(*args, **kwargs):
    return neptune.create_experiment(*args, **kwargs)


@obtrusive_call
def append_tags(*args, **kwargs):
    return neptune.append_tags(*args, **kwargs)


@obtrusive_call
def log_text(*args, **kwargs):
    return neptune.log_text(*args, **kwargs)


@obtrusive_call
def log_metric(*args, **kwargs):
    return neptune.log_metric(*args, **kwargs)


@obtrusive_call
def log_artifact(*args, **kwargs):
    return neptune.log_artifact(*args, **kwargs)

