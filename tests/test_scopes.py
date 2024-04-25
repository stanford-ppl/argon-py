from argon.node.control import stage_if_exp
from argon.state import State
from argon.types.integer import Integer
from argon.types.boolean import Boolean
from argon.wrapper import argon_function


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

@argon_function
def test_scope3():
    state = State()
    with state:
        a = Boolean().const(True)
        b = Boolean().const(False)
        c = Integer().const(3)
        d = Integer().const(6)
        e = c + d if a & b else d + c
    print(state)
