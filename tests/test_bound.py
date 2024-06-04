from argon.state import State
from argon.types.boolean import Boolean
from argon.types.integer import Integer


def test_bound():
    state = State()
    with state:
        a = Boolean().bound("x")
        assert isinstance(a, Boolean)

        b = Boolean().const(True)
        b = b.bound("y")
        assert isinstance(b, Boolean)

        c = Integer().bound("m")
        assert isinstance(c, Integer)

        d = Integer().const(1)
        d = d.bound("n")
        assert isinstance(d, Integer)
