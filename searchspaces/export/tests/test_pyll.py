from searchspaces.partialplus import partial, as_partialplus, evaluate
from searchspaces.export.pyll import as_pyll
from hyperopt.pyll import rec_eval, scope


def test_pyll_tuple():
    x = as_partialplus((6, 9, 4))
    y = as_pyll(x)
    assert evaluate(x) == rec_eval(y)


def test_pyll_list():
    x = as_partialplus([5, 3, 9])
    y = as_pyll(x)
    # rec_eval always uses tuple
    assert evaluate(x) == list(rec_eval(y))


def test_pyll_list_tuple_nested():
    x = as_partialplus([[5, 3, (5, 3)], (4, 5)])
    y = as_pyll(x)
    # rec_eval always uses tuple
    val_y = rec_eval(y)
    # Correct for tuple-only in rec_eval.
    assert evaluate(x) == [list(val_y[0]), val_y[1]]


def test_pyll_func():
    # N.B. Only uses stuff that's already in the SymbolTable.
    x = partial(float, 5)
    y = as_pyll(x)
    assert evaluate(x) == rec_eval(y)


def test_pyll_nested_func():
    x = partial(float, partial(int, 5.5))
    y = as_pyll(x)
    assert evaluate(x) == rec_eval(y)


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


def test_dict():
        x = as_partialplus({'x': partial(float,
                                         partial(float,
                                                 partial(int, 3.3))) / 2,
                            'y': partial(float, 3)
                            })
        y = as_pyll(x)
        print y
        assert evaluate(x) == rec_eval(y)


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
