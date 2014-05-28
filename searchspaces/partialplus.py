"""
Support code for `functools.partial` based deferred-evaluation
mechanism.
"""
__authors__ = "David Warde-Farley"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

# Keep this to only standard library imports so that this is droppable in
# another code-base for re-use.
from collections import deque
import compiler
from functools import partial as _partial
import operator
import warnings
from itertools import izip, repeat

# TODO: support o_len functionality from old Apply nodes


def is_choice_node(node):
    return hasattr(node, 'func') and node.func is choice_node


def is_literal(node):
    return isinstance(node, Literal)


def is_variable_node(node):
    return hasattr(node, 'func') and node.func is variable_node


def is_tuple_node(node):
    return hasattr(node, 'func') and node.func is make_tuple


def is_list_node(node):
    return hasattr(node, 'func') and node.func is make_list


def is_sequence_node(node):
    return is_tuple_node(node) or is_list_node(node)


def is_pos_args_node(node):
    return (hasattr(node, 'func') and node.func is call_with_list_of_pos_args)


def is_dict_like_node(node):
    return (is_pos_args_node(node) and is_literal(node.args[0]) and
            issubclass(node.args[0].value, dict))


def is_indexable(node):
    if len(node.args) != 2 or len(node.keywords) > 0:
        return False
    obj, index = node.args
    if is_sequence_node(obj) or is_dict_like_node(obj):
        return True
    else:
        return False


def make_list(*args):
    """
    Wrapper for the builtin `list()` that calls it on *args.

    Handles tuples encountered by as_partialplus, so that we don't
    have to have special logic for recursing on lists.
    """
    return list(args)


def make_tuple(*args):
    """
    Wrapper for the builtin `tuple()` that calls it on *args.

    Handles tuples encountered by as_partialplus, so that we don't
    have to have special logic for recursing on tuples.
    """
    return tuple(args)


def variable_node(*args, **kwargs):
    """
    Marker function for variable nodes created by `variable()`.

    Notes
    -----
    By convention we store everything in kwargs.
    """


def choice_node(v):
    return v


def choice(choice_var, *args):
    new_args = [as_partialplus(v) for v in args]
    # TODO: support len(2) lists as well as len(2) tuples?
    if any(not is_tuple_node(n) or len(n.args) != 2 for n in new_args):
        raise ValueError("arguments to choice() must be length-2 tuples")
    v = partial(call_with_list_of_pos_args, dict, *args)[choice_var]
    return partial(choice_node, v)


def call_with_list_of_pos_args(f, *args):
    return f(args)


def partial(f, *args, **kwargs):
    """
    A workalike for `functools.partial` that actually (recursively)
    creates `PartialPlus` objects via `as_partialplus`.

    Parameters
    ----------
    f : callable
        Function whose evaluation to defer.

    Notes
    -----
    Remaining positional and keyword arguments are passed along to
    `f`, as `functools.partial`.
    """
    return as_partialplus(_partial(f, *args, **kwargs))


def as_partialplus(p):
    """
    Convert a (possibly nested) `partial` to the

    Parameters
    ----------
    p : object
        If `p` is a `functools.partial`, a list, or a tuple, it is
        given special treatment, and its arguments/elements are recursed
        upon. Otherwise, it is wrapped in a `Literal`.

    Returns
    -------
    node : object
        A `PartialPlus`, or a `Literal`.
    """
    if isinstance(p, (PartialPlus, Literal)):
        return p
    elif isinstance(p, _partial):
        args = [as_partialplus(a) for a in p.args]
        if p.keywords:
            kwargs = dict((k, as_partialplus(v))
                          for k, v in p.keywords.iteritems())
            return PartialPlus(p.func, *args, **kwargs)
        else:
            return PartialPlus(p.func, *args)
    # Not using isinstance, on purpose. Want literal lists and tuples,
    # not subclasses.
    elif type(p) in (list, tuple):
        if type(p) == list:
            func = make_list
        else:
            func = make_tuple
        return PartialPlus(func, *(as_partialplus(e) for e in p))
    # Definitely want this to work for OrderedDicts.
    elif isinstance(p, dict):
        # Special-case dictionaries to recurse on values.
        actual_args = [as_partialplus((k, v)) for k, v in p.iteritems()]
        # Sort by is_literal in reverse order, so we hit the literal
        # keys first.
        actual_args.sort(key=lambda x: is_literal(x[0]), reverse=True)
        args = [Literal(p.__class__)] + actual_args

        return PartialPlus(call_with_list_of_pos_args, *args)
    else:
        return Literal(p)


class UniqueStack(object):
    """
    Implementation of a stack (using a deque) that also checks pushed elements
    for uniqueness.
    """
    def __init__(self):
        self._deque = deque()
        self._members = set()

    def push(self, elem):
        """
        Push an element onto the stack.

        Parameters
        ----------
        elem : object

        Raises
        ------
        KeyError
            If `elem` already exists in this stack.
        """

        if elem in self._members:
            raise KeyError(str(elem))
        else:
            self._deque.append(elem)
            self._members.add(elem)

    def pop(self):
        """
        Pop the topmost element off the stack.

        Returns
        -------
        elem : object
            The top-most element from the stack (removed).

        Raises
        ------
        IndexError
            If the stack is empty.
        """
        try:
            elem = self._deque.pop()
            self._members.remove(elem)
        except IndexError:
            raise IndexError("pop from an empty %s" % self.__class__.__name__)
        return elem

    def pop_until(self, elem):
        """
        Pop and discard items until the head of the stack is the
        object `elem`.

        Parameters
        ----------
        elem : object
            The desired element at the head of the stack.

        Raises
        ------
        ValueError
            If the stack is emptied before finding `elem`.
        """
        while len(self._deque) > 0 and self._deque[-1] is not elem:
            self.pop()
        if len(self._deque) == 0:
            raise ValueError("never found sentinel element")


def _traversal_helper(root, build_inverted=False):
    """
    Helper function for `depth_first_traversal` and `topological_sort`.

    Parameters
    ----------
    root : Node
    build_inverted : boolean, optional
        If `True`, after all nodes have been yielded, yield a dictionary
        containing an inverted index.

    Returns
    -------
    gen : generator object
        A generator producing nodes from the graph, in a depth-first order.
        If `build_inverted` is True then the last item it yields is a
        dictionary containing an inverted index (mapping nodes to their
        parents).

    Raises
    ------
    ValueError
        If the graph contains a directed cycle.
    """
    assert isinstance(root, Node)
    visited = {}
    to_visit = deque()
    # TODO: optimize out UniqueStack class by just doing what it would do
    # in the function body.
    path = UniqueStack()
    # None = our sentinel value for "no parent".
    path.push(None)
    to_visit.append((None, root))
    while len(to_visit) > 0:
        parent, node = to_visit.pop()
        path.pop_until(parent)
        try:
            path.push(node)
        except KeyError:
            raise ValueError("call graph contains a directed cycle")
        if node not in visited:
            if build_inverted:
                visited.setdefault(node, set()).add(parent)
            else:
                visited[node] = True
            yield node
            if isinstance(node, PartialPlus):
                children = node.args + (tuple(node.keywords.values())
                                        if node.keywords is not None else ())
                to_visit.extend((node, c) for c in children)
        elif build_inverted:
            visited[node].add(parent)
    if build_inverted:
        visited[root].remove(None)
        yield visited


def depth_first_traversal(root):
    """
    Perform a depth-first traversal of a graph of PartialPlus objects.

    Parameters
    ----------
    root : Node

    Returns
    -------
    gen : generator object
        A generator producing nodes from the graph, in a depth-first order.

    Raises
    ------
    ValueError
        If the graph contains a directed cycle.
    """
    return _traversal_helper(root)


def topological_sort(root):
    """
    Perform a topological sort of a graph of PartialPlus objects.

    Parameters
    ----------
    root : Node

    Returns
    -------
    gen : generator object
        A generator producing nodes from the graph, in a topological order.

    Raises
    ------
    ValueError
        If the graph contains a directed cycle.
    """
    # TODO: make this more efficient and natively support reverse sort
    # (probably by getting two dictionaries).
    candidates = deque(_traversal_helper(root, build_inverted=True))
    dependencies = candidates.pop()
    visited = set()
    while candidates:
        proposed = candidates.popleft()
        if dependencies[proposed].difference(visited):
            candidates.append(proposed)
        else:
            visited.add(proposed)
            yield proposed


class MissingArgument(object):
    """Object to represent a missing argument to a function application
    """
    def __init__(self):
        assert 0, "Singleton class not meant to be instantiated"


def _extract_param_names(fn):
    """
    Grab the names of positional arguments, as well as the varargs
    and kwargs parameter, if they exist.

    Parameters
    ----------
    fn : function object
        The function to be inspected.

    Returns
    -------
    param_names : list
        A list of all the function's argument names.

    pos_args : list
        A list of names of the non-special arguments to `fn`.

    args_param : str or None
        The name of the variable-length positional args parameter,
        or `None` if `fn` does not accept a variable number of
        positional arguments.

    kwargs_param : str or None
        The name of the variable-length keyword args parameter,
        or `None` if `fn` does not accept a variable number of
        keyword arguments.
    """
    code = fn.__code__

    extra_args_ok = bool(code.co_flags & compiler.consts.CO_VARARGS)
    extra_kwargs_ok = bool(code.co_flags & compiler.consts.CO_VARKEYWORDS)
    expected_num_args = (code.co_argcount + int(extra_args_ok) +
                         int(extra_kwargs_ok))
    assert len(code.co_varnames) >= expected_num_args
    param_names = code.co_varnames[:expected_num_args]
    args_param = (param_names[code.co_argcount]
                  if extra_args_ok else None)
    kwargs_param = (param_names[code.co_argcount + int(extra_args_ok)]
                    if extra_kwargs_ok else None)
    pos_params = param_names[:code.co_argcount]
    return pos_params, args_param, kwargs_param


def _bind_parameters(params, named_args, kwargs_param, binding=None):
    """
    Resolve bindings for arguments from a list of parameter
    names.

    Parameters
    ----------
    params : list
        A list of names of positional parameters.

    named_args : dict
        A dictionary mapping names of keyword parameters to
        values to bind to them.

    kwargs_param : str or None
        The name of the extended/optional keywords parameter
        to use for keys in `named_args` that do not appear in
        `params`. If this is None, excess keyword arguments
        not listed in `params` will raise an error.

    binding : dict, optional
        A dictionary of existing name to value bindings, i.e.
        from processing positional arguments.

    Returns
    -------
    binding : dict
        A dictionary of argument names to bound values, including
        any passed in via the `binding` argument.
    """
    binding = {} if binding is None else dict(binding)
    if kwargs_param:
        binding[kwargs_param] = {}
    params_set = set(params)
    for aname, aval in named_args.iteritems():
        if aname in params_set and not aname in binding:
            binding[aname] = aval
        elif aname in binding and aname != kwargs_param:
            raise TypeError('Duplicate argument for parameter: %s' % aname)
        elif kwargs_param:
            binding[kwargs_param][aname] = aval
        else:
            raise TypeError('Unrecognized keyword argument: %s' % aname)
    return binding


def _param_assignment(pp):
    """
    Calculate parameter assignment of partial
    """
    binding = {}

    fn = pp.func
    code = fn.__code__
    pos_args = pp.args
    named_args = {} if pp.keywords is None else pp.keywords
    params, args_param, kwargs_param = _extract_param_names(fn)

    if len(pos_args) > code.co_argcount and not args_param:
        raise TypeError('Argument count exceeds number of positional params')
    elif args_param:
        binding[args_param] = pos_args[code.co_argcount:]

    # -- bind positional arguments
    for param_i, arg_i in izip(params, pos_args):
        binding[param_i] = arg_i

    # -- bind keyword arguments
    binding.update(_bind_parameters(params, named_args, kwargs_param, binding))
    expected_length = (len(params) + int(kwargs_param is not None) +
                       int(args_param is not None))
    assert len(binding) <= expected_length

    # Right-aligned default values for params. Default to empty tuple
    # so that iteration below simply terminates in this case.
    defaults = fn.__defaults__ if fn.__defaults__ else ()

    # -- fill in default parameter values
    for param_i, default_i in izip(params[-len(defaults):], defaults):
        binding.setdefault(param_i, Literal(default_i))

    # -- mark any outstanding parameters as missing
    missing_names = set(params) - set(binding)
    binding.update(izip(missing_names, repeat(MissingArgument)))
    return binding


class Node(object):
    def clone(self):
        bindings = {}
        nodes = reversed(list(topological_sort(self)))
        for node in nodes:
            if isinstance(node, Literal):
                bindings[node] = Literal(node.value)
            else:  # PartialPlus
                func = node.func
                args = [bindings[a] for a in node.args]
                keywords = dict((k, bindings[v])
                                for k, v in node.keywords.iteritems())
                bindings[node] = PartialPlus(func, *args, **keywords)
        return bindings[nodes[-1]]

    def inputs(self):
        return ()


class Literal(Node):
    func = None
    args = None
    keywords = None

    def __init__(self, value):
        self._value = value

    def __gt__(self, other):
        if not hasattr(other, 'value'):
            return False
        return self.value > other.value

    def __lt__(self, other):
        if not hasattr(other, 'value'):
            return False
        return self.value < other.value

    def __eq__(self, other):
        if not hasattr(other, 'value'):
            return False
        return self.value == other.value

    @property
    def value(self):
        return self._value


class PartialPlus(_partial, Node):
    """
    A subclass of `functools.partial` that allows for
    common arithmetic/builtin operations to be performed
    on them, deferred by wrapping in another object of
    this same type. Also overrides `__call__` to suggest
    you use the recursive version, `evaluate`.

    Notable exceptions *not* implemented include __len__ and
    __iter__, because returning non-integer/iterator stuff
    from those methods tends to break things.
    """

    def __init__(self, f, *args, **kwargs):
        assert all(isinstance(a, Node) for a in args)
        assert all(isinstance(v, Node) for k, v in kwargs.iteritems())
        super(PartialPlus, self).__init__(self, f, *args, **kwargs)
        self._keywords = kwargs
        self._args = args

    def __call__(self, *args, **kwargs):
        raise TypeError("use evaluate() for %s objects" %
                        partial.__name__)

    def __add__(self, other):
        return partial(operator.add, self, other)

    def __sub__(self, other):
        return partial(operator.sub, self, other)

    def __mul__(self, other):
        return partial(operator.mul, self, other)

    def __floordiv__(self, other):
        return partial(operator.floordiv, self, other)

    def __mod__(self, other):
        return partial(operator.mod, self, other)

    def __divmod__(self, other):
        return partial(divmod, self, other)

    def __pow__(self, other, modulo=None):
        return partial(pow, self, other, modulo)

    def __lshift__(self, other):
        return partial(operator.lshift, self, other)

    def __rshift__(self, other):
        return partial(operator.rshift, self, other)

    def __and__(self, other):
        return partial(operator.and_, self, other)

    def __xor__(self, other):
        return partial(operator.xor, self, other)

    def __or__(self, other):
        return partial(operator.or_, self, other)

    def __div__(self, other):
        return partial(operator.div, self, other)

    def __truediv__(self, other):
        return partial(operator.truediv, self, other)

    def __lt__(self, other):
        return partial(operator.lt, self, other)

    def __le__(self, other):
        return partial(operator.le, self, other)

    def __gt__(self, other):
        return partial(operator.gt, self, other)

    def __ge__(self, other):
        return partial(operator.ge, self, other)

    def __neg__(self):
        return partial(operator.neg, self, '-')

    def __pos__(self):
        return partial(operator.pos, self)

    def __abs__(self):
        return partial(abs, self)

    def __invert__(self):
        return partial(abs, self)

    def __complex__(self):
        return partial(complex, self)

    def __int__(self):
        return partial(int, self)

    def __long__(self):
        return partial(long, self)

    def __float__(self):
        return partial(float, self)

    def __oct__(self):
        return partial(oct, self)

    def __hex__(self):
        return partial(hex, self)

    def __getitem__(self, item):
        if not isinstance(item, Node):
            item = as_partialplus(item)
        return partial(operator.getitem, self, item)

    @property
    def pos_args(self):
        warnings.warn("Use .args, not .pos_args")
        return self.args

    def inputs(self):
        # TODO: make this a property
        return self.args + (tuple(self.keywords.itervalues())
                            if self.keywords is not None else ())

    @property
    def arg(self):
        # TODO: bindings
        return _param_assignment(self)

    @property
    def keywords(self):
        """
        Overwrite the default keywords attribute to always have a dictionary
        in that spot rather than None sometimes, which makes for a lot of
        annoying special cases.
        """
        return self._keywords

    @property
    def args(self):
        """
        Overwrite the default args attribute so that we have more control
        over it, and can thereby append arguments.
        """
        return self._args

    def append_arg(self, arg):
        self._args = self._args + (arg,)


def variable(name, value_type, minimum=None, maximum=None, default=None,
             log_scale=False, distribution=None, **kwargs):
    """
    Create a special variable node to be replaced at evaluation time
    of a `PartialPlus` graph.

    Parameters
    ----------
    name : str
        A unique string identifier. Must be a valid Python variable name.
        TODO: validate this requirement.
    value_type : type or iterable
        One of `float`, `int`, or a sequence of possible values.
    minimum : float or int, optional
        If `value_type` is float or int, the minimum value this variable
        can take.
    maximum : float or int, optional
        If `value_type` is float or int, the maximum value this variable
        can take.
    default : object, optional
        A "default" value for this variable, used by some optimizers.
    log_scale : bool, optional
        Indicator used by some systems to determine whether a quantity
        should be treated as if varying on a logarithmic scale.
    distribution : callable(?), optional
        A prior distribution on the support of this parameter, used by
        some optimizers.

    Returns
    -------
    variable_node : PartialPlus
        A `PartialPlus` with `variable_node` as the function attribute.
    """
    d = locals()
    d.update(kwargs)  # kwargs guaranteed not to have keys already in locals()
    return partial(variable_node, **d)


def evaluate(p, **kwargs):
    """
    Evaluate a nested tree of functools.partial objects,
    used for deferred evaluation.

    Parameters
    ----------
    p : object

    """
    return _evaluate(p, bindings=kwargs)


def _handle_indexing(p, instantiate_call, bindings, recurse):
    # Assumes is_indexable has already returned True.
    obj, index = p.args
    index_val = recurse(index)
    if is_sequence_node(obj):
        elem_val = obj.args[index_val]
        if isinstance(index_val, slice):  # TODO: something more robust?
            # elem_val is a sliced out sublist, recurse on each element
            # therein and call obj.func (make_list, make_tuple) on result.
            elem_val = instantiate_call(obj.func,
                                        *[recurse(e) for e in elem_val])
        else:
            elem_val = recurse(elem_val)
        try:
            # bindings the value of this subexpression as
            int(index_val)
            bindings[p] = elem_val
        except TypeError:
            # TODO: is this even conceivably used?
            bindings[p] = instantiate_call(p.func, elem_val, index_val)
    else:  # assumes is_dict_like_node(obj) is True
        assert obj.func == call_with_list_of_pos_args
        assert all(is_tuple_node(node) and len(node.args) == 2
                   for node in obj.args[1:])
        # TODO: check length better when output-length annotation is supported.
        keys, values = zip(*(node.args for node in obj.args[1:]))
        # We could only evaluate as many keys as it takes to find the right
        # one, but this might make what gets evaluated or not kind of hard
        # to predict.
        keys = [recurse(k) for k in keys]
        try:
            ind = keys.index(index_val)
        except ValueError:
            raise KeyError(index_val)
        bindings[p] = recurse(values[ind])
    return bindings[p]


def _evaluate(p, instantiate_call=None, bindings=None):
    """
    Evaluate a nested tree of functools.partial objects,
    used for deferred evaluation.

    Parameters
    ----------
    p : object
        If `p` is a partial, or a subclass of partial, it is
        expanded recursively. Otherwise, return.
    instantiate_call : callable, optional
        Rather than call `p.func` directly, instead call
        `instantiate_call(p.func, ...)`
    bindings : dict, optional
        A dictionary mapping `Node` objects to values to use
        in their stead. Used to cache objects already evaluated.

    Returns
    -------
    q : object
        The result of evaluating `p` if `p` was a partial
        instance, or else `p` itself.

    Notes
    -----
    For large graphs this recursive implementation may hit the
    recursion limit and be kind of slow. TODO: write an
    iterative version.
    """
    instantiate_call = ((lambda f, *args, **kwargs: f(*args, **kwargs))
                        if instantiate_call is None else instantiate_call)
    bindings = {} if bindings is None else bindings

    # If we've encountered this exact partial node before,
    # short-circuit the evaluation of this branch and return
    # the pre-computed value.
    if p in bindings:
        return bindings[p]
    if isinstance(p, Literal):
        bindings[p] = p.value
        return bindings[p]

    recurse = _partial(_evaluate, instantiate_call=instantiate_call,
                       bindings=bindings)

    # When evaluating an expression of the form
    # `list(...)[item]`
    # only evaluate the element(s) of the list that we need.
    if p.func == operator.getitem and is_indexable(p):
        return _handle_indexing(p, instantiate_call, bindings, recurse)
    args = [recurse(arg) for arg in p.args]
    kw = (dict((kw, recurse(val)) for kw, val in p.keywords.iteritems())
          if p.keywords else {})

    if is_variable_node(p):
        assert 'name' in p.keywords
        name = kw['name']
        try:
            return bindings[name]
        except KeyError:
            raise KeyError("variable with name '%s' not bound" % name)

    # bindings the evaluated value (for subsequent calls that
    # will look at this bindings dictionary) and return.
    bindings[p] = instantiate_call(p.func, *args, **kw)
    return bindings[p]
