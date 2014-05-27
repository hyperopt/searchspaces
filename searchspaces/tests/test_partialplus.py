from collections import OrderedDict
import operator
from searchspaces.partialplus import partial, Literal
from searchspaces.partialplus import evaluate, variable, is_indexable
from searchspaces.partialplus import depth_first_traversal, topological_sort
from searchspaces.partialplus import as_partialplus as as_pp


def test_is_indexable():
    """Tests is_indexable works as expected."""
    # We only test cases where the node function is getitem, since that's
    # assumed as a precondition.
    node = partial(operator.getitem, [4, 2], 1)
    assert is_indexable(node)
    node = partial(operator.getitem, {4: 2}, 1)
    assert is_indexable(node)
    # Malformed getitem nodes.
    node = partial(operator.getitem, {4: 2}, 1, 3)
    assert not is_indexable(node)
    node = partial(operator.getitem, [5, 3, 9], 1, 3)
    assert not is_indexable(node)
    node = partial(operator.getitem, [5, 3, 9], 1, k=4)
    assert not is_indexable(node)
    # Not a sequence or a dict-like.
    node = partial(operator.getitem, 5, 3)
    assert not is_indexable(node)


def test_arithmetic():
    """Test that arithmetic works properly on PartialPlus objects."""
    def check(a, b):
        assert evaluate(as_pp(partial(int, a)) +
                        as_pp(partial(int, b))) == a + b
        assert evaluate(as_pp(partial(int, a)) -
                        as_pp(partial(int, b))) == a - b
        assert evaluate(as_pp(partial(int, a)) *
                        as_pp(partial(int, b))) == a * b
        assert evaluate(as_pp(partial(int, a)) /
                        as_pp(partial(int, b))) == a / b
        assert evaluate(as_pp(partial(int, a)) %
                        as_pp(partial(int, b))) == a % b
        assert evaluate(as_pp(partial(int, a)) |
                        as_pp(partial(int, b))) == a | b
        assert evaluate(as_pp(partial(int, a)) ^
                        as_pp(partial(int, b))) == a ^ b
        assert evaluate(as_pp(partial(int, a)) &
                        as_pp(partial(int, b))) == a & b
    yield check, 6, 5
    yield check, 4, 2
    yield check, 9, 11


def test_lazy_index_list():
    """
    Test that lazy evaluation of indexing works with element lookups on
    lists.
    """
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'

    # TODO: James: I opted for this behaviour rather than list(f, el1, el2...)
    # is there a compelling reason to do that? It kind of breaks with the
    # model.
    plist = as_pp([-1, partial(dont_eval)])
    assert -1 == evaluate(plist[0])


def test_lazy_index_tuple():
    """
    Test that lazy evaluation of indexing works with element lookups on
    lists.
    """
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'

    # TODO: James: I opted for this behaviour rather than list(f, el1, el2...)
    # is there a compelling reason to do that? It kind of breaks with the
    # model.
    plist = as_pp((-1, partial(dont_eval)))
    assert -1 == evaluate(plist[0])


def test_lazy_index_range():
    """
    Test that lazy evaluation of indexing works with range lookups.
    """
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'
    plist = as_pp([-1, 0, 1, partial(dont_eval)])
    assert [-1, 0, 1] == evaluate(plist[:3])

    plist = as_pp((-1, 0, 1, partial(dont_eval)))
    assert (-1, 0, 1) == evaluate(plist[:3])


def test_getitem_dict():
    """Test that indexing works on a PartialPlus'd dict."""
    v = evaluate(as_pp({'a': partial(float, '3'), 'b': 3})['a'])
    assert v == 3.0


def test_lazy_index_dict():
    """
    Test that lazy evaluation of indexing works with dict lookups.
    """
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'

    r = evaluate(as_pp({'a': partial(dont_eval), 'b': 3})['b'])
    assert r == 3


def test_switch_ordered_dict():
    """Test that lazy evaluation works with OrderedDicts."""
    def dont_eval():
        # -- This function body should never be evaluated
        #    because we only need the 0'th element of `plist`
        assert 0, 'Evaluate does not need this, should not eval'
    r = evaluate(as_pp(OrderedDict({'a': partial(dont_eval), 'b': 3}))['b'])
    assert r == 3


def test_arg():
    """Test basic partial.arg lookups."""
    def f(a, b=None):
        return -1

    assert partial(f, 0, 1).arg['a'] == Literal(0)
    assert partial(f, 0, 1).arg['b'] == Literal(1)

    assert partial(f, 0).arg['a'] == Literal(0)
    assert partial(f, 0).arg['b'] == Literal(None)

    assert partial(f, a=3).arg['a'] == Literal(3)
    assert partial(f, a=3).arg['b'] == Literal(None)

    assert partial(f, 2, b=5).arg['a'] == Literal(2)
    assert partial(f, 2, b=5).arg['b'] == Literal(5)

    assert partial(f, a=2, b=5).arg['a'] == Literal(2)
    assert partial(f, a=2, b=5).arg['b'] == Literal(5)


def test_star_args():
    """Test partial.arg lookups on *args."""
    def f(a, *b):
        return -1

    assert partial(f, 0, 1).arg['a'] == Literal(0)
    assert partial(f, 0, 1).arg['b'] == (Literal(1),)
    assert partial(f, 0, 1, 2, 3).arg['b'] == (Literal(1), Literal(2),
                                               Literal(3))


def test_kwargs():
    """Test partial.arg lookups on **kwargs."""
    def f(a, **b):
        return -1

    assert partial(f, 0, b=1).arg['a'] == Literal(0)
    assert partial(f, 0, b=1).arg['b'] == {'b': Literal(1)}
    assert partial(f, 0, foo=1, bar=2, baz=3).arg['b'] == {
        'foo': Literal(1),
        'bar': Literal(2),
        'baz': Literal(3),
    }


def test_star_kwargs():
    """Test partial.arg lookups on *args and **kwargs."""
    def f(a, *u, **b):
        return -1

    assert partial(f, 0, b=1).arg['a'] == Literal(0)
    assert partial(f, 0, b=1).arg['b'] == {'b': Literal(1)}

    assert partial(f, 0, 'q', 'uas', foo=1, bar=2).arg['a'] == Literal(0)
    assert partial(f, 0, 'q', 'uas', foo=1, bar=2).arg['u'] == (Literal('q'),
                                                                Literal('uas'))
    assert partial(f, 0, 'q', 'uas', foo=1, bar=2).arg['b'] == {
        'foo': Literal(1),
        'bar': Literal(2),
    }


def test_tuple():
    """Test that tuples are correctly traversed/converted."""
    def add(x, y):
        return x + y

    x = as_pp(((3, partial(add, 2, 3)), partial(add, 5, 7), partial(float, 9)))
    y = evaluate(x)
    assert y == ((3, 5), 12, 9.0)
    assert isinstance(y[2], float)


def test_list():
    """Test that lists are correctly traversed/converted."""
    def sub(x, y):
        return x - y

    x = as_pp([[3, partial(sub, 2, 3)], partial(sub, 5, 7), partial(float, 9)])
    y = evaluate(x)
    assert y == [[3, -1], -2, 9.0]
    assert isinstance(y[2], float)


def test_dict():
    """Test that dicts are correctly traversed/converted."""
    def mod(x, y):
        return x % y
    x = as_pp({5: partial(mod, 5, 3), 3: (7, 9), 4: [partial(mod, 9, 4)]})
    y = evaluate(x)
    assert y == {5: 2, 3: (7, 9), 4: [1]}


def test_ordered_dict():
    """Test that OrderedDicts are correctly traversed/converted."""
    def mod(x, y):
        return x % y
    x = as_pp(OrderedDict({5: partial(mod, 5, 3), 3: (7, 9),
                           4: [partial(mod, 9, 4)]}))
    y = evaluate(x)
    assert y == {5: 2, 3: (7, 9), 4: [1]}


def test_depth_first_traversal():
    """Test that depth-first traversal works."""
    # p1 must appear after either p2 or p3, but not necessarily after both.
    p1 = partial(float, 5.0)
    # p2 must appear after either p3 or p4, but not necessarily after both.
    p2 = p1 + 0.5
    p3 = p1 / p2
    p4 = p2 * p3
    p5 = partial(int, p4)
    traversal = list(depth_first_traversal(p5))
    assert traversal.index(p5) == 0
    assert traversal.index(p4) == 1
    assert traversal.index(p3) > traversal.index(p4)
    assert (traversal.index(p2) > traversal.index(p3) or
            traversal.index(p2) > traversal.index(p4))
    assert (traversal.index(p1) > traversal.index(p2) or
            traversal.index(p1) > traversal.index(p3))


def test_topological_sort():
    """Test that topological sort produces a correct partial ordering."""
    # p1 must appear before BOTH p2 and p3.
    p1 = partial(float, 5)
    # p2 must appear before BOTH p3 and p4.
    p2 = p1 + 0.5
    p3 = p1 / p2
    p4 = p2 * p3
    p5 = partial(int, p4)
    toposort = list(topological_sort(p5))
    assert toposort.index(p5) == 0
    assert toposort.index(p4) == 1
    assert (toposort.index(p1) > toposort.index(p2)
            and toposort.index(p1) > toposort.index(p3))
    assert (toposort.index(p2) > toposort.index(p3)
            and toposort.index(p2) > toposort.index(p4))
    assert toposort.index(Literal(5)) > toposort.index(p1)
    assert toposort.index(Literal(0.5)) > toposort.index(p2)


def test_cycle_detection():
    """
    Test that depth_first_traversal and topological_sort correctly
    find cycles.
    """
    def assert_raised(graph, fn):
        raised = False
        try:
            list(fn(graph))
        except ValueError:
            raised = True
        assert raised

    p1 = partial(float, 5)
    p2 = partial(int, p1)
    p3 = partial(float, p2)
    p4 = partial(int, p3)
    p1.keywords['not_a_real_keyword'] = p1
    # Simple cycle on a single node.
    assert_raised(p1, depth_first_traversal)
    assert_raised(p1, topological_sort)
    # Detect a single node cycle when it isn't the root.
    assert_raised(p2, depth_first_traversal)
    assert_raised(p2, topological_sort)
    # Larger cycle.
    del p1.keywords['not_a_real_keyword']
    p1.keywords['not_a_real_keyword_either'] = p2
    assert_raised(p4, depth_first_traversal)
    assert_raised(p4, topological_sort)
    p1.keywords['not_a_real_keyword_either'] = p3
    assert_raised(p4, depth_first_traversal)
    assert_raised(p4, topological_sort)
    p1.keywords['not_a_real_keyword_either'] = p4
    assert_raised(p4, depth_first_traversal)
    assert_raised(p4, topological_sort)
    del p1.keywords['not_a_real_keyword_either']
    # Test with a positional argument.
    p1.append_arg(p4)
    assert_raised(p4, depth_first_traversal)
    assert_raised(p4, topological_sort)


def test_two_objects():
    """
    Test that identical expression in different parts of graph evaluates
    to an identical object.
    """
    class Foo(object):
        pass
    p = partial(Foo)
    q = as_pp([p, [0, 1, p], [(p,)]])
    r = evaluate(q)
    assert r[0] is r[1][-1]
    assert r[0] is r[2][0][0]


def test_variable_substitution():
    """Test that variable substitution works correctly."""
    x = variable(name='x', value_type=int)
    y = variable(name='y', value_type=float)
    p = as_pp({3: x, x: [y, [y]], y: 4})
    # Currently no type-checking. This will fail when we add it and need to be
    # updated.
    e = evaluate(p, x='hey', y=5)
    assert e[3] == 'hey'
    assert e['hey'] == [5, [5]]
    assert e[5] == 4
