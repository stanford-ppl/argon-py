from argon.state import State
from argon.types.integer import Integer


def test_scope():
    state = State()
    with state:
        a = Integer().const(3)
        b = Integer().const(6)
        c = a + b
        print(c, c.rhs)
        d = c + b
        print(d, d.rhs)
        print(state)
