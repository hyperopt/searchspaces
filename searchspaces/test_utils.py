from functools import wraps
from nose import SkipTest


def skip_if_no_module(name):
    """
    W

    Parameters
    ----------
    name : str
        The name of the import to depend on.
    Raises
    ------
    SkipTest
        If the module named in `name` can't be imported.
    """
    def wrapper_maker(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                __import__(name)
            except ImportError:
                raise SkipTest()
            return f(*args, **kwargs)
        return wrapped
    return wrapper_maker
