from argon.state import State
from argon.types.integer import Integer


def test_integer():
    state = State()
    with state:
        a = Integer().const(3)
        b = 6
        c = a + b
        d = b + a
        e = a - b
        f = b - a
        g = a > b
        h = b > a
        i = a < b
        j = b < a
    print(state)
