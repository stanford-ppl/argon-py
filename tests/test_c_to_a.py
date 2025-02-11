from types import NoneType
from argon.state import State
from argon.types.boolean import Boolean
from argon.types.integer import Integer
from argon.types.null import Null
from argon.virtualization.type_mapper import concrete_to_abstract


def func(x):
    return x


def test_c_to_a():
    state = State()
    with state:
        a = 3
        a1 = concrete_to_abstract(a)
        assert a1.C is int
        assert a1.A is Integer

        b = True
        b1 = concrete_to_abstract(b)
        assert b1.C is bool
        assert b1.A is Boolean

        c = None
        c1 = concrete_to_abstract(c)
        assert c1.C is NoneType
        assert c1.A is Null

        d = func
        d1 = concrete_to_abstract.function(d, [concrete_to_abstract(3)])
        assert d1.RETURN_TP is Integer
    print(state)
