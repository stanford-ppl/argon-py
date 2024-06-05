from argon.state import State
from argon.types.boolean import Boolean


def test_boolean():
    state = State()
    with state:
        a = Boolean().const(True)
        b = Boolean().const(False)
        c = a & b
        d = c | b
        e = ~d
        f = e ^ a
    print(state)
