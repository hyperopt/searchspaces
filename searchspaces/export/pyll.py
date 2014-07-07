"""
hyperopt pyll backend for partialplus graphs.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"


__all__ = ["as_pyll"]

import inspect
import operator
import types


try:
    from hyperopt import pyll, hp

except ImportError:
    raise ImportError("This functionality requires hyperopt "
                      "<http://hyperopt.github.io/hyperopt/>")

from ..partialplus import (is_sequence_of_literals, is_sequence_node,
                           is_pos_args_node, is_variable_node, is_choice_node,
                           is_literal, is_uniform_categorical,
                           is_weighted_categorical)
from ..partialplus import topological_sort, Literal


def _convert_uniform_categorical(pp_var, bindings):
    # Already know it's a variable node.
    val_type = pp_var.keywords['value_type']
    assert is_sequence_of_literals(val_type), \
        "val_type for categorical must be sequence of literals"
    n_choices = len(val_type.args)
    assert is_literal(pp_var.keywords['name'])
    randint = hp.randint(pp_var.keywords['name'].value, n_choices)
    return bindings[val_type][randint]


def _convert_weighted_categorical(pp_var, bindings):
    raise NotImplementedError()


def _convert_variable(pp_variable, bindings):
    """Convert a PartialPlus variable node into a hyperopt stochastic."""
    keywords = dict(pp_variable.keywords)
    distribution = bindings[keywords['distribution']]
    # This literal constraint is necessary so we can create the right kind of
    # node at conversion time.
    assert isinstance(distribution, pyll.base.Literal)
    assert isinstance(distribution.obj, (basestring, types.NoneType))
    # Special handling for categoricals.
    if is_weighted_categorical(pp_variable):
        return _convert_weighted_categorical(pp_variable, bindings)
    elif is_uniform_categorical(pp_variable):
        return _convert_uniform_categorical(pp_variable, bindings)
    else:
        inspectable_func = getattr(pyll.stochastic, distribution.obj, None)
        hp_func = getattr(hp, distribution.obj, None)
        if inspectable_func is None:
            raise ImportError("Couldn't find hyperopt.pyll.stochastic.%s" %
                              str(distribution))
        if hp_func is None:
            raise ImportError("Couldn't find hyperopt.hp.%s" %
                              str(distribution))
    # Use the name as the label.
    assert is_literal(pp_variable.keywords['name'])
    # Hyperopt's convention for distributions
    keywords['label'] = keywords['name']
    keywords['low'] = keywords['minimum']
    keywords['high'] = keywords['maximum']
    if is_literal(keywords['maximum']):
        # TODO: Literal should have arithmetic capabilities.
        maximum = keywords['maximum']
        keywords['upper'] = (Literal(maximum.value + 1)
                             if maximum.value is not None
                             else maximum)
        bindings[keywords['upper']] = _convert_literal(keywords['upper'])
    else:
        keywords['upper'] = keywords['maximum'] + 1
        bindings[keywords['upper']] = _convert_partialplus(keywords['upper'],
                                                           bindings)
    del keywords['minimum'], keywords['maximum']
    arg_names = inspect.getargspec(inspectable_func).args
    if distribution.obj == 'randint':
        assert 'low' not in keywords or (
            is_literal(keywords['low']) and keywords['low'].value is None
        ), (
            "nonzero minimum not supported by hyperopt randint"
        )
        # TODO: check for integer upper value?
    dist_args = dict((k, bindings[keywords[k]]) for k in arg_names
                     if k in keywords)
    dist_args['label'] = bindings[keywords['label']]
    return hp_func(**dist_args)


def _convert_literal(pp_literal):
    """Convert a searchspaces Literal to a hyperopt Literal."""
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
    if is_variable_node(node):
        # TODO: currrently variables can't have hyper(hyper)parameters
        # that are partialpluses. Fix this.
        return _convert_variable(node, bindings)
    elif is_sequence_node(node):
        return _convert_sequence(node, bindings)
    # Convert the pos_args node for, e.g. dictionaries.
    elif is_pos_args_node(node):
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
    apply_node = getattr(pyll.scope, f.__name__)(*args, **kwargs)
    apply_node.define_params = {'f': f}
    return apply_node


def as_pyll(root):
    """
    Converts a `partialplus` (sub)graph into a `hyperopt.pyll` graph,
    making the appropriate representational substitutions.

    Parameters
    ----------
    root : Node

    Returns
    -------
    pyll_root : Apply
        A (graph of) `hyperopt.pyll.Apply` node(s).
    """
    bindings = {}
    for node in reversed(list(topological_sort(root))):
        if isinstance(node, Literal):
            bindings[node] = _convert_literal(node)
        else:
            bindings[node] = _convert_partialplus(node, bindings)
    return bindings[root]
