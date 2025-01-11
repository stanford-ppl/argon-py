# Argon-py

Argon-py is a domain-agnostic library for tracing and virtualization, inspired by David Koeplinger's work on the Argon DSL framework ([David's Thesis, see Appendix](http://purl.stanford.edu/wb222ps0455)).

The high-level goal of Argon-py is to handle the task of building staged DSL frontends -- the gap between describing the DSL and obtaining the program graph. This process is generally done on a DSL-by-DSL basis, such as the case of PyTorch's JIT, Exocompilation, and SEJITS from UC Berkeley.
Argon-py provides a framework for defining DSL frontends and provides the tracing and virtualization mechanisms needed by all staged DSLs.
Currently, Argon-py supports straight-line code, functions/function calls, `if`/`else` statements, and generic types, with ongoing work to support looping constructs such as `for` and `while`.

For example, to define a staged Boolean type (see boolean.py) we have the following:

```python
class Boolean(Ref[bool, "Boolean"]):
    """
    The Boolean class represents a boolean value in the Argon language.
    """

    @override
    def fresh(self) -> "Boolean":
        return Boolean()

    def __invert__(self) -> "Boolean":
        return stage(logical.Not[Boolean](self), ctx=SrcCtx.new(2))

    def __and__(self, other: "Boolean") -> "Boolean":
        other = typing.cast(Boolean, concrete_to_abstract(other))
        return stage(logical.And[Boolean](self, other), ctx=SrcCtx.new(2))

    def __rand__(self, other: "Boolean") -> "Boolean":
        return self & other

    def __or__(self, other: "Boolean") -> "Boolean":
        other = typing.cast(Boolean, concrete_to_abstract(other))
        return stage(logical.Or[Boolean](self, other), ctx=SrcCtx.new(2))

    def __ror__(self, other: "Boolean") -> "Boolean":
        return self | other

    def __xor__(self, other: "Boolean") -> "Boolean":
        other = typing.cast(Boolean, concrete_to_abstract(other))
        return stage(logical.Xor[Boolean](self, other), ctx=SrcCtx.new(2))

    def __rxor__(self, other: "Boolean") -> "Boolean":
        return self ^ other
```

This defines a new `Boolean` type, whose concrete type is the python `bool`. Python dunder methods are used to capture operations such as `x & y`, which are then staged into the scope.

# Installation
Argon-py requires Python 3.12 or later, as it uses several features introduced in the 3.12 release.

1. In the root directory, run:

```shell
python -m pip install .
```

2. If you are getting ModuleNotFoundError for `argon`, then add this to the `sys.path` directory list. (This should probably be done using a virtual environment.)

```shell
# On MacOSX
vim ~/.bash_profile
# Then, add: 
#      export PYTHONPATH="<root-directory>/argon-py/src"
source ~/.bash_profile
```