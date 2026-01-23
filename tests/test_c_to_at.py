from types import NoneType
import types
import typing
from argon.state import State
from argon.types.boolean import Boolean
from argon.types.function import Function
from argon.types.integer import Integer
from argon.types.null import Null
from argon.virtualization.type_mapper import concrete_to_abstract_type


def func(x):
    return x


def test_c_to_at():
    state = State()
    with state:
        a = concrete_to_abstract_type[int]
        assert a is Integer

        b = concrete_to_abstract_type[bool]
        assert b is Boolean

        c = concrete_to_abstract_type[NoneType]
        assert c is Null

        d = concrete_to_abstract_type[typing.Callable[..., int]]
        assert d is Function[Integer]

        try:
            e = concrete_to_abstract_type[types.FunctionType]
        except ValueError as e:
            assert str(e) == f"Cannot convert {types.FunctionType} to abstract type, use typing.Callable and specify the parameter and return types"
        
        f = concrete_to_abstract_type[typing.Callable[..., typing.Callable[..., bool]]]
        assert f is Function[Function[Boolean]]
    print(state)
