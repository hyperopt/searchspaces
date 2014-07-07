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
                           is_literal, is_categorical)
from ..partialplus import topological_sort, Literal


def _convert_categorical(pp_var, bindings):
    val_type = pp_var.keywords['value_type']
    assert is_sequence_of_literals(val_type), \
        "val_type for categorical must be sequence of literals"
    n_choices = len(val_type.args)
    assert is_literal(pp_var.keywords['name'])
    if 'p' in pp_var.keywords:
        p = pp_var.keywords['p']
        randint_stoch = pyll.scope.categorical(bindings[p], upper=n_choices)
        name = bindings[pp_var.keywords['name']]
        randint = pyll.scope.hyperopt_param(name, randint_stoch)
    else:
        randint = hp.randint(bindings[pp_var.keywords['name']], n_choices)
    return bindings[val_type][randint]


def _convert_variable(pp_variable, bindings):
    """Convert a PartialPlus variable node into a hyperopt stochastic."""
    keywords = dict(pp_variable.keywords)
    distribution = bindings[keywords['distribution']]
    # This literal constraint is necessary so we can create the right kind of
    # node at conversion time.
    assert isinstance(distribution, pyll.base.Literal)
    assert isinstance(distribution.obj, (basestring, types.NoneType))
    # Special handling for categoricals.
    if is_categorical(pp_variable):
        return _convert_categorical(pp_variable, bindings)
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
        one = Literal(1)
        bindings[one] = _convert_literal(one)
        keywords['upper'] = keywords['maximum'] + one
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
    dist_args['label'] = bindings[keywords['name']]
    return hp_func(**dist_args)


def _convert_choice(pp_choice, bindings):
    """
    Convert a PartialPlus `choice` node into a hyperopt `switch`.
    """
    getitem_node = pp_choice.args[0]
    assert getitem_node.func == operator.getitem
    assert len(getitem_node.args) == 2
    choices, index = getitem_node.args
    assert is_variable_node(index), "non-variable node encountered in choice"
    assert is_categorical(index), ("choice() conversion to pyll currently "
                                   "requires a categorical index")
    assert is_pos_args_node(choices)
    assert all(is_sequence_node(p) and
               len(p.args) == 2 for p in choices.args[1:]), \
        "Expected a sequence of pairs in choice list"
    if not all(is_literal(c.args[0]) for c in choices.args[1:]):
        raise ValueError("Need all keys in choice list to be literals "
                         "(not computed)")
    assert bindings[index].name == 'getitem'
    # Pull out the randint from the previously converted categorical.
    randint_node = bindings[index].pos_args[1]
    # Assumes value type is list of literals; _convert_uniform_categorical
    # would have already failed if not. Can probably remove this assertion.
    assert is_sequence_of_literals(index.keywords['value_type'])
    val_type_keys = [k.value for k in index.keywords['value_type'].args]
    c_keys, c_values = [list(x) for x in zip(*((c.args[0].value, c.args[1])
                                                for c in choices.args[1:]))]
    assert (len(val_type_keys) == len(c_keys) and  # small optimization
            len(set(val_type_keys).symmetric_difference(c_keys))) == 0, \
        "values %s taken on by categorical not same as choices %s" % (
            str(val_type_keys), str(c_keys)
        )
    value_idx_lookup = [c_values[c_keys.index(v)] for v in val_type_keys]
    hopt_values = [bindings[v] for v in value_idx_lookup]
    return pyll.scope.switch(randint_node, *hopt_values)


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
    # Convert substitutable variable nodes.
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
    elif is_choice_node(node):
        return _convert_choice(node, bindings)
    else:
        f = node.func
        args = [bindings[p] for p in args]
        kwargs = dict((k, bindings[v]) for k, v in kwargs.iteritems())
    # In any case, add the function to the scope object if need be and create
    # an equivalent Apply node. define_params tells us what setup we need to
    # do when this node if and when this node is deserialized.
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
