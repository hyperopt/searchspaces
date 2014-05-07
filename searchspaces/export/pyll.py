"""
hyperopt pyll backend for partialplus graphs.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"


__all__ = ["as_pyll"]


try:
    from hyperopt import pyll
except ImportError:
    raise ImportError("need hyperopt bro TODO better message")

from ..partialplus import is_tuple_node, is_list_node, is_pos_args_node
from ..partialplus import topological_sort, Literal

# TODO: arithmetic. Is it even necessary?


def _convert_literal(pp_literal):
    """
    Convert a searchspaces Literal to a hyperopt Literal."""
    return pyll.as_apply(pp_literal.value)


def _convert_sequence(pp_seq, bindings):
    """
    Convert a tuple or list node into the equivalent Apply node.

    Parameters
    ----------
    pp_seq : PartialPlus
        Must be a tuple node or a list node.
    bindings : dict
        A dictionary mapping `PartialPlus`/`Literal` objects to Apply
        nodes already converted, for converting the elements of the sequence.

    Returns
    -------
    apply_seq : Apply
        The equivalent `Apply` node representation.
    """
    return pyll.as_apply([bindings[p] for p in pp_seq.args])


def _func_to_apply(f, args, kwargs, bindings):
    """
    Convert a function call with `args` and `kwargs` into  an Apply node.

    Parameters
    ----------
    f : callable
        A function/callable object. Must be callable and also have a
        `__name__` attribute.
    args : sequence
        A sequence (list, tuple) of arguments as `PartialPlus`es or
        `Literal`s.
    args : sequence
        A dictionary mapping keyword arguments to `PartialPlus`es or
        `Literal`s.
    bindings : dict
        A dictionary mapping `PartialPlus`/`Literal` objects to Apply
        nodes already converted, for converting the elements/values
        in `args` and `kwargs`.

    Returns
    -------
    apply_seq : Apply
        The equivalent `Apply` node representation.
    """
    # TODO: support o_len when partialplus supports it
    try:
        f = pyll.scope.define_if_new(f)
    except ValueError:
        raise ValueError("conflicting definition in hyperopt.pyll.scope for " +
                         f.__name__)
    args = [bindings[p] for p in args]
    kwargs = dict((k, bindings[v]) for k, v in kwargs.items())
    return getattr(pyll.scope, f.__name__)(*args, **kwargs)


def _convert_partialplus(node, bindings):
    """
    Convert a `PartialPlus` node into  an Apply node.

    Parameters
    ----------
    node : PartialPlus
        A `PartialPlus` object to be converted into an `Apply`.
    bindings : dict
        A dictionary mapping `PartialPlus`/`Literal` objects to Apply
        nodes already converted, for converting the elements/values
        in `node.args` and `node.keywords`.

    Returns
    -------
    apply_seq : Apply
        The equivalent `Apply` node representation.

    Notes
    -----
    Special-cases the `PartialPlus` "pos-args" node used for constructing
    dictionaries and dictionary subclasses. For these, creates an `Apply`
    with `node.args[0]` as the function and `node.args[1]` as the
    positionals.
    """
    args = node.args
    kwargs = node.keywords
    if is_pos_args_node(node):
        assert isinstance(node.args[0], Literal)
        assert hasattr(node.args[0].value, '__call__')
        assert len(kwargs) == 0
        f = args[0].value
        args = [pyll.as_apply([bindings[p] for p in args[1:]])]
    else:
        f = node.func
        args = [bindings[p] for p in args]
        kwargs = dict((k, bindings[v]) for k, v in kwargs.items())
    f = pyll.scope.define_if_new(f)
    return getattr(pyll.scope, f.__name__)(*args, **kwargs)


def as_pyll(pp):
    """
    """
    bindings = {}
    for node in reversed(list(topological_sort(pp))):
        if isinstance(node, Literal):
            bindings[node] = _convert_literal(node)
        elif is_tuple_node(node) or is_list_node(node):
            bindings[node] = _convert_sequence(node, bindings)
        else:
            bindings[node] = _convert_partialplus(node, bindings)
    return bindings[pp]
