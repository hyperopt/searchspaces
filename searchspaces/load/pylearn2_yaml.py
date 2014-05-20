"""
Support for sucking in pylearn2's `Proxy` IR and converting to `PartialPlus`.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/searchspaces"

from pylearn2.config import yaml_parse
from pylearn2.utils.string_utils import preprocess
from ..partialplus import partial, as_partialplus, Literal
from functools import partial as _partial


def append_yaml_src(obj, yaml_src):
    """
    Append a `.yaml_src` attribute to objects that permit.

    Parameters
    ----------
    obj : object
        The object on which to set the attribute.
    yaml_src : basestring
        The value to set for the `yaml_src` attribute.

    Returns
    -------
    obj : object
        The object, with or without the attribute set, depending
        on whether `AttributeError` was raised on the attempt.
    """
    try:
        obj.yaml_src = yaml_src
    except AttributeError:
        pass
    return obj


def append_yaml_callback(proxy, pp):
    """
    A callback that returns a `PartialPlus` node that appends a
    `yaml_src` to the result of another `PartialPlus` node.

    Parameters
    ----------
    proxy : pylearn2.config.yaml_parse.Proxy
        The Proxy object corresponding to `pp`.
    pp : PartialPlus
        A `PartialPlus` object derived from `proxy`.

    Returns
    -------
    pp_append : PartialPlus
        A `PartialPlus` node that calls `append_yaml-src` on
        `pp` with the `yaml_src` from `Proxy`.

    Notes
    -----
    This is a particular instance of a general callback that may
    serve more general purposes.
    """
    return partial(append_yaml_src, pp, proxy.yaml_src)


def proxy_to_partialplus(proxy, literal_callback=None,
                         proxy_callback=append_yaml_callback,
                         preprocess_strings=True,
                         environ=None, bindings=None):
    """
    Convert a `Proxy` hierarchy read in from a Pylearn2 YAML
    file into a `PartialPlus` graph.

    Parameters
    ----------
    proxy : object
        An object in a `Proxy` hierarchy derived from parsing a
        Pylearn2 YAML file.
    literal_callback : callable, optional
        One-argument callable to be run on the value before
        instantiating a `Literal` node with that value (and
        before running string substitution preprocessing).
        Should return a replacement value.
    proxy_callback : callable, optional
        Two-argument callable to be run with the `Proxy` node
        as the first argument and the created `PartialPlus`
        node as the second argument. Should return a replacement
        `PartialPlus` node. By default, `append_yaml_callback`.
    preprocess_strings : bool, optional
        If `True` (the default), strings will be preprocessed
        with `pylearn2.utils.string_utils.preprocess`, and the
        supplied `environ` argument.
    environ : dict, optional
        If supplied, preferentially accept values for string
        substitution from this dictionary as well as `os.environ`.
        That is, if a key appears in both, this dictionary takes
        precedence.
    bindings : dict, optional
        A dictionary of previously converted `Proxy` objects to
        their equivalent `PartialPlus` representations.

    Returns
    -------
    node : object
        A `PartialPlus` or `Literal` object corresponding to the
        object represented by `proxy`.

    Raises
    ------
    ValueError
        If `environ` is specified but `preprocess_strings` is
        `False`.

    Notes
    -----
    If you implement a custom `proxy_callback`, you might want to call
    `append_yaml_src` from within it.
    """
    literal_callback = literal_callback if literal_callback else (lambda x: x)
    if not preprocess_strings and environ:
        raise ValueError('environ specified but preprocess_strings is False')
    # So we don't re-convert already converted objects.
    if bindings is None:
        bindings = {}
    # Convenience wrapper for recursive calls.
    recurse = _partial(proxy_to_partialplus,
                       literal_callback=literal_callback,
                       proxy_callback=proxy_callback,
                       preprocess_strings=preprocess_strings,
                       environ=environ, bindings=bindings)
    if isinstance(proxy, yaml_parse.Proxy):
        if proxy in bindings:
            return bindings[proxy]
        # Positional arguments.
        if proxy.callable == yaml_parse.do_not_recurse:
            p = Literal(append_yaml_src(proxy.keywords['value'],
                                        proxy.yaml_src))
        else:
            args = ([recurse(v) for v in proxy.positionals]
                    if proxy.positionals else [])
            kwargs = (dict((k, recurse(v))
                           for k, v in proxy.keywords.iteritems())
                      if proxy.keywords else {})
            callback = proxy_callback if proxy_callback else lambda _, x: x
            p = callback(proxy, partial(proxy.callable, *args, **kwargs))
            # Don't put a do_not_recurse Literal in the bindings.
            bindings[proxy] = p
    # If it's a list, recurse on the elements.
    elif isinstance(proxy, list):
        p = as_partialplus([recurse(v) for v in proxy])
    # If it's a dict, recurse on the values.
    elif isinstance(proxy, dict):
        p = as_partialplus(dict((k, recurse(v)) for k, v in proxy.iteritems()))
    else:
        # If it's not a Proxy, list or a dict.
        o = literal_callback(proxy)
        # Preprocess strings if necessary.
        p = (partial(preprocess, o, environ=Literal(environ))
             if preprocess_strings and isinstance(o, basestring)
             else as_partialplus(o))
    return p


def load(stream, environ=None, **kwargs):
    """
    Loads a YAML configuration from a string or file-like object
    into a `PartialPlus` graph.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.

    Returns
    -------
    graph : Node

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    return proxy_to_partialplus(yaml_parse.load(stream, instantiate=False,
                                                **kwargs), environ=environ)


def load_path(path, environ=None, **kwargs):
    """
    Convenience function for loading a YAML configuration from a file
    into a `PartialPlus` graph.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    environ : dict, optional
        A dictionary used for ${FOO} substitutions in addition to
        environment variables. If a key appears both in `os.environ`
        and this dictionary, the value in this dictionary is used.

    Returns
    -------
    graph : Node
        A `PartialPlus` or `Literal` node representing the root
        node of the YAML hierarchy.

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    return proxy_to_partialplus(yaml_parse.load_path(path, instantiate=False,
                                                     **kwargs),
                                environ=environ)
