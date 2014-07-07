import operator
import numpy as np

from searchspaces.partialplus import (
    partial, as_partialplus, evaluate, choice, variable
)
from searchspaces.test_utils import skip_if_no_module
try:
    from searchspaces.export.pyll import as_pyll
    from hyperopt.pyll import rec_eval, scope
    from hyperopt.pyll.stochastic import recursive_set_rng_kwarg
except ImportError:
    pass


@skip_if_no_module('hyperopt.pyll')
def test_repeated_node():
    q = partial(float, 5)
    p = as_pyll(as_partialplus([q, q, [q]]))
    assert p.pos_args[0] is p.pos_args[1]
    assert p.pos_args[0] is p.pos_args[2].pos_args[0]


def test_randint():
    v = variable('some_random_int', value_type=int, distribution='randint',
                 maximum=5)
    p = as_pyll(v)
    assert p.name == 'hyperopt_param'
    assert p.pos_args[0].obj == 'some_random_int'
    assert p.pos_args[1].name == 'randint'
    assert p.pos_args[1].arg['upper'].obj == 6
    # upper is not a constant
    v2 = variable('some_other_int', value_type=int, distribution='randint',
                  maximum=partial(operator.add, 2, 3))
    p2 = as_pyll(v2)
    assert p2.name == 'hyperopt_param'
    assert p2.pos_args[0].obj == 'some_other_int'
    assert p2.pos_args[1].name == 'randint'
    assert p2.pos_args[1].arg['upper'].name == 'add'


def check_continuous_variable(label, dist_name, **params):
    v = variable(label, value_type=float, distribution=dist_name,
                 **params)
    p = as_pyll(v)
    assert p.name == 'float'  # Implementation detail
    assert p.pos_args[0].name == 'hyperopt_param'
    assert p.pos_args[0].pos_args[0].obj == label
    assert p.pos_args[0].pos_args[1].name == dist_name
    if 'minimum' in params:
        params['low'] = params['minimum']
        del params['minimum']
    if 'maximum' in params:
        params['high'] = params['maximum']
        del params['maximum']
    for key in params:
        assert p.pos_args[0].pos_args[1].arg[key].obj == params[key]


@skip_if_no_module('hyperopt.pyll')
def test_normal_variable():
    check_continuous_variable('some_normal_thing', 'normal',
                              mu=5, sigma=9)


@skip_if_no_module('hyperopt.pyll')
def test_lognormal_variable():
    check_continuous_variable('some_lognormal_thing', 'lognormal',
                              mu=3, sigma=14)


@skip_if_no_module('hyperopt.pyll')
def test_qnormal_variable():
    check_continuous_variable('some_quantized_normal_thing', 'qnormal',
                              mu=2.7, sigma=3.5, q=2)


@skip_if_no_module('hyperopt.pyll')
def test_qlognormal_variable():
    check_continuous_variable('some_quantized_lognormal_thing', 'qlognormal',
                              mu=4.444, sigma=5.0, q=2)


@skip_if_no_module('hyperopt.pyll')
def test_uniform_variable():
    check_continuous_variable('some_uniform_thing', 'uniform',
                              minimum=0, maximum=5)


@skip_if_no_module('hyperopt.pyll')
def test_loguniform_variable():
    check_continuous_variable('some_loguniform_thing', 'loguniform',
                              minimum=7, maximum=9)


@skip_if_no_module('hyperopt.pyll')
def test_quniform_variable():
    check_continuous_variable('some_quantized_uniform_thing', 'quniform',
                              minimum=44, maximum=55)


@skip_if_no_module('hyperopt.pyll')
def test_qloguniform_variable():
    check_continuous_variable('some_quantized_loguniform_thing', 'qloguniform',
                              minimum=-532, maximum=66)


@skip_if_no_module('hyperopt.pyll')
def test_nonuniform_categorical():
    p = as_pyll(variable('baz', value_type=[3, 5, 9],
                         distribution='categorical',
                         p=[0.1, 0.4, 0.5]))
    assert p.name == 'getitem'
    assert p.pos_args[0].name == 'pos_args'
    assert p.pos_args[1].name == 'hyperopt_param'
    assert p.pos_args[1].pos_args[0].name == 'literal'
    assert p.pos_args[1].pos_args[0].obj == 'baz'
    assert p.pos_args[1].pos_args[1].name == 'categorical'
    assert p.pos_args[1].pos_args[1].arg['p'].name == 'pos_args'
    assert p.pos_args[1].pos_args[1].arg['p'].pos_args[0].obj == 0.1
    assert p.pos_args[1].pos_args[1].arg['p'].pos_args[1].obj == 0.4
    assert p.pos_args[1].pos_args[1].arg['p'].pos_args[2].obj == 0.5
    # Make sure this executes and yields a value in the right domain.
    recursive_set_rng_kwarg(p, np.random)
    try:
        values = [rec_eval(p) for _ in xrange(10)]
    except Exception:
        assert False
    assert all(v in [3, 5, 9] for v in values)


@skip_if_no_module('hyperopt.pyll')
def test_uniform_categorical():
    p = as_pyll(variable('foo', value_type=[-1, 1, 4]))
    assert p.name == 'getitem'
    assert p.pos_args[0].name == 'pos_args'
    assert p.pos_args[1].name == 'hyperopt_param'
    assert p.pos_args[1].pos_args[0].name == 'literal'
    assert p.pos_args[1].pos_args[0].obj == 'foo'
    assert p.pos_args[1].pos_args[1].name == 'randint'
    # Make sure this executes and yields a value in the right domain.
    recursive_set_rng_kwarg(p, np.random)
    try:
        values = [rec_eval(p) for _ in xrange(10)]
    except Exception:
        assert False
    assert all(v in [-1, 1, 4] for v in values)


@skip_if_no_module('hyperopt.pyll')
def test_nonuniform_choice():
    var = variable('blu', value_type=[2, 4, 8], distribution='categorical',
                   p=[0.2, 0.7, 0.1])
    p = as_pyll(choice(var,
                       (2, 'abc'),
                       (4, 'def'),
                       (8, 'ghi')))
    assert p.name == 'switch'
    assert p.pos_args[0].name == 'hyperopt_param'
    assert p.pos_args[0].pos_args[0].obj == 'blu'
    assert p.pos_args[0].pos_args[1].name == 'categorical'
    assert p.pos_args[0].pos_args[1].arg['p'].name == 'pos_args'
    assert p.pos_args[0].pos_args[1].arg['p'].pos_args[0].obj == 0.2
    assert p.pos_args[0].pos_args[1].arg['p'].pos_args[1].obj == 0.7
    assert p.pos_args[0].pos_args[1].arg['p'].pos_args[2].obj == 0.1
    # Make sure this executes and yields a value in the right domain.
    recursive_set_rng_kwarg(p, np.random)
    try:
        values = [rec_eval(p) for _ in xrange(10)]
    except Exception:
        assert False
    assert all(v in ['abc', 'def', 'ghi'] for v in values)


@skip_if_no_module('hyperopt.pyll')
def test_uniform_choice():
    p = as_pyll(choice(variable('foo', value_type=[7, 9, 11]),
                       (7, 'rst'),
                       (9, 'uvw'),
                       (11, 'xyz')))
    assert p.name == 'switch'
    assert p.pos_args[0].name == 'hyperopt_param'
    assert p.pos_args[0].pos_args[0].obj == 'foo'
    assert p.pos_args[0].pos_args[1].name == 'randint'
    assert p.pos_args[0].pos_args[1].arg['upper'].obj == 3
    # Make sure this executes and yields a value in the right domain.
    recursive_set_rng_kwarg(p, np.random)
    try:
        values = [rec_eval(p) for _ in xrange(10)]
    except Exception:
        assert False
    assert all(v in ['rst', 'uvw', 'xyz'] for v in values)


@skip_if_no_module('hyperopt.pyll')
def test_pyll_tuple():
    x = as_partialplus((6, 9, 4))
    y = as_pyll(x)
    assert evaluate(x) == rec_eval(y)


@skip_if_no_module('hyperopt.pyll')
def test_pyll_list():
    x = as_partialplus([5, 3, 9])
    y = as_pyll(x)
    # rec_eval always uses tuple
    assert evaluate(x) == list(rec_eval(y))


@skip_if_no_module('hyperopt.pyll')
def test_pyll_list_tuple_nested():
    x = as_partialplus([[5, 3, (5, 3)], (4, 5)])
    y = as_pyll(x)
    # rec_eval always uses tuple
    val_y = rec_eval(y)
    # Correct for tuple-only in rec_eval.
    assert evaluate(x) == [list(val_y[0]), val_y[1]]


@skip_if_no_module('hyperopt.pyll')
def test_pyll_func():
    # N.B. Only uses stuff that's already in the SymbolTable.
    x = partial(float, 5)
    y = as_pyll(x)
    assert evaluate(x) == rec_eval(y)


@skip_if_no_module('hyperopt.pyll')
def test_pyll_nested_func():
    x = partial(float, partial(int, 5.5))
    y = as_pyll(x)
    assert evaluate(x) == rec_eval(y)


@skip_if_no_module('hyperopt.pyll')
def test_pyll_deeply_nested_func():
    # N.B. uses stuff that isn't in the SymbolTable yet, must remove.
    try:
        def my_add(x, y):
            return x + y

        x = as_partialplus(
            (partial(float, partial(my_add, 0, partial(int, 3.3))) / 2,
             partial(float, 3))
        )
        y = as_pyll(x)
        evaluate(x) == rec_eval(y)
    finally:
        scope.undefine(my_add)


@skip_if_no_module('hyperopt.pyll')
def test_dict():
        x = as_partialplus({'x': partial(float,
                                         partial(float,
                                                 partial(int, 3.3))) / 2,
                            'y': partial(float, 3)
                            })
        y = as_pyll(x)
        assert evaluate(x) == rec_eval(y)


@skip_if_no_module('hyperopt.pyll')
def test_pyll_scope_doesnt_overwrite():
    raised = False
    try:
        def float(x):
            return x + 1
        as_pyll(partial(float, 3))
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    test_pyll_func()
