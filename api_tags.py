import functools
import warnings


class UnfinishedAPIWarning(Warning):
    pass


class UntestedAPIWarning(Warning):
    pass


# A decorator to mark a function as untested
def untested(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', UntestedAPIWarning)  # ensure the warning is always shown
        warnings.warn(
            f"Call to untested function {func.__name__} with args: {args}, kwargs: {kwargs}.",
            category=UntestedAPIWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)

    return new_func


# A decorator for stable APIs
def stable_api(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # Optionally log or document stable API usage here
        return func(*args, **kwargs)

    return new_func


# A decorator to mark a function as unfinished
def unfinished_api(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', UnfinishedAPIWarning)  # ensure the warning is always shown
        warnings.warn(
            f"Call to unfinished API {func.__name__} with args: {args}, kwargs: {kwargs}.",
            category=UnfinishedAPIWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)

    return new_func


# A decorator to mark a function as buggy (raise an error when called)
def bug_api(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        raise NotImplementedError(
            f"The function {func.__name__} is known to have a bug and has not been implemented properly.")

    return new_func
