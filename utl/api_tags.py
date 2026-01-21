import functools
import warnings


class UnfinishedAPIWarning(Warning):
    pass


class UntestedAPIWarning(Warning):
    pass


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    >>> @deprecated
    >>> def my_func(*args, **kwargs):
    >>>     return args, kwargs
    >>> my_func(1, 2, a=3, b=4)
    warning: Call to deprecated function my_func.
    (1, 2), {'a': 3, 'b': 4}
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func

# A decorator to mark a function as untested


def untested(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # ensure the warning is always shown
        warnings.simplefilter('always', UntestedAPIWarning)
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
        # ensure the warning is always shown
        warnings.simplefilter('always', UnfinishedAPIWarning)
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
