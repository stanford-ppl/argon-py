from argon.state import State
from argon.types.integer import Integer
from argon.types.boolean import Boolean


def test_scope():
    state = State()
    with state:
        a = Integer().const(3)
        b = Integer().const(6)
        c = a + b
        d = c + b

def test_scope2():
    state = State()
    with state:
        a = Boolean().const(True)
        b = Boolean().const(False)
        c = a & b
        d = c | b
        e = ~d
        f = e ^ a
    print(state)
