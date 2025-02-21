from types import NoneType
from argon.state import State
from argon.types.boolean import Boolean
from argon.types.integer import Integer
from argon.virtualization.type_mapper import concrete_to_bound


def func(x):
    return x


def test_c_to_a():
    state = State()
    with state:
        a = concrete_to_bound(int, "a")
        assert a.C is int
        assert a.A is Integer
        assert a.is_bound()
        assert a.rhs.val.name == "a"

        b = concrete_to_bound(bool, "b")
        assert b.C is bool
        assert b.A is Boolean
        assert b.is_bound()
        assert b.rhs.val.name == "b"

        try:
            c = concrete_to_bound(NoneType, "c")
        except TypeError as e:
            assert str(e) == "Cannot bind NoneType"
    print(state)
