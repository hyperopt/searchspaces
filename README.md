searchspaces
------------

searchspaces is a tool for describing hyperparameter search spaces. More
generally it defines the `PartialPlus` primitive, based in spirit on
`functools.partial`, for deferred evaluation.

It currently supports import from pylearn2 YAML configuration files export to
`hyperopt.pyll` for use with the
[hyperopt][1] hyperparameter optimization
suite.

Coming very soon: completed configuration generators for [SMAC][2]. Coming in
the medium-term: [Spearmint](https://github.com/JasperSnoek/spearmint) support.

[1]: http://hyperopt.github.io/hyperopt/
[2]: http://www.cs.ubc.ca/labs/beta/Projects/SMAC/
