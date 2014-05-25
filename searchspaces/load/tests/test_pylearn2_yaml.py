import os
import tempfile
from searchspaces.test_utils import skip_if_no_module
from searchspaces import evaluate
try:
    from searchspaces.load.pylearn2_yaml import (
        append_yaml_src, append_yaml_callback, proxy_to_partialplus,
        load, load_path
    )
    from pylearn2.config.yaml_parse import Proxy, do_not_recurse
except ImportError:
    pass

class Foo(object):
    def __init__(self, x=None):
        self.x = x


@skip_if_no_module('pylearn2')
def test_proxy_to_partialplus():
    # Use proxy_callback=None to test without YAML-appending.
    pp = proxy_to_partialplus(Proxy(callable=dict, positionals=(),
                                    keywords={'x': [5, 3, 2]}, yaml_src=None),
                              proxy_callback=None)
    assert evaluate(pp) == {'x': [5, 3, 2]}


@skip_if_no_module('pylearn2')
def test_proxy_to_partialplus_literal_callback():
    def baz(x):
        if isinstance(x, int):
            x *= 2
        return x

    # Use proxy_callback=None to test without YAML-appending.
    pp = proxy_to_partialplus(Proxy(callable=dict, positionals=(),
                                    keywords={'x': 5}, yaml_src=None),
                              proxy_callback=None, literal_callback=baz)
    assert evaluate(pp)['x'] == 10


@skip_if_no_module('pylearn2')
def test_append_yaml_src():
    # Test that the append doesn't trigger an AttributeError on builtins.
    raised = False
    try:
        append_yaml_src({'x': 5}, "blah blah blah")
    except AttributeError:
        raised = True
    assert not raised

    result = append_yaml_src(Foo(), "bloop bloop bloop")
    assert hasattr(result, "yaml_src")
    assert result.yaml_src == "bloop bloop bloop"


@skip_if_no_module('pylearn2')
def test_append_yaml_callback():
    pfail = Proxy(callable=dict, positionals=(), keywords={'value': 5},
                  yaml_src="test_value_1")
    pobj = Proxy(callable=Foo, positionals=(), keywords={'x': 3},
                 yaml_src="test_value_2")
    raised = False
    try:
        test = append_yaml_callback(pfail, proxy_to_partialplus(pfail))
    except AttributeError:
        raised = True
    assert not raised
    test = evaluate(append_yaml_callback(pobj, proxy_to_partialplus(pobj)))
    assert hasattr(test, 'yaml_src')
    assert test.yaml_src == "test_value_2"


@skip_if_no_module('pylearn2')
def test_preprocessing():
    try:
        os.environ['FOO'] = 'abcdef'
        pp = proxy_to_partialplus(Proxy(callable=dict, positionals=(),
                                        keywords={'x': '${FOO}'},
                                        yaml_src=None))
        p = evaluate(pp)
        assert p['x'] == 'abcdef'
        pp = proxy_to_partialplus(Proxy(callable=dict, positionals=(),
                                        keywords={'x': '${BAR}'},
                                        yaml_src=None),
                                  environ={'BAR': 'fedcba'})
        p = evaluate(pp)
        assert p['x'] == 'fedcba'
        pp = proxy_to_partialplus(Proxy(callable=dict, positionals=(),
                                        keywords={'x': '${FOO}'},
                                        yaml_src=None),
                                  environ={'FOO': 'ghijkl'})
        p = evaluate(pp)
        assert p['x'] == 'ghijkl'
    finally:
        del os.environ['FOO']
    # Test turning it off.
    pp = proxy_to_partialplus(Proxy(callable=dict, positionals=(),
                                    keywords={'x': '${BAZ}'},
                                    yaml_src=None), preprocess_strings=False)
    p = evaluate(pp)
    assert p['x'] == '${BAZ}'


@skip_if_no_module('pylearn2')
def test_do_not_recurse():
    proxy = Proxy(callable=do_not_recurse, positionals=(),
                  keywords={'value': Proxy(None, None, None, None)},
                  yaml_src=None)
    assert isinstance(evaluate(proxy_to_partialplus(proxy)), Proxy)


@skip_if_no_module('pylearn2')
def test_identical_proxy_identical_partialplus():
    proxy = Proxy(lambda: None, None, None, None)
    pp = proxy_to_partialplus([{'a': proxy}, proxy], proxy_callback=None)
    assert pp.args[0].args[1].args[1] is pp.args[1]


@skip_if_no_module('pylearn2')
def test_load():
    src = '!obj:searchspaces.load.tests.test_pylearn2_yaml.Foo {x: 5}\n'
    pp = load(src)
    p = evaluate(pp)
    assert p.yaml_src == src
    assert p.x == 5
    assert isinstance(p, Foo)
    src = "!obj:searchspaces.load.tests.test_pylearn2_yaml.Foo {x: '${FOO}'}\n"
    pp = load(src, environ={'FOO': 'abcdef'})
    p = evaluate(pp)
    assert p.x == 'abcdef'


@skip_if_no_module('pylearn2')
def test_load_path():
    src = '!obj:searchspaces.load.tests.test_pylearn2_yaml.Foo {x: 5}\n'
    try:
        fd, fn = tempfile.mkstemp()
        os.close(fd)
        with open(fn, 'w') as f:
            f.write(src)
        pp = load_path(fn)
        p = evaluate(pp)
        print p
        assert p.yaml_src == src
        assert p.x == 5
        assert isinstance(p, Foo)
    finally:
        os.remove(fn)
