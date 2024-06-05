from argon.state import State
from argon.types.boolean import Boolean
from argon.types.integer import Integer
from argon.virtualization.wrapper import argon_function


@argon_function(calls=False, ifs=False)
def if_exps():
    a = Boolean().const(True)
    b = Boolean().const(False)
    c = Integer().const(3)
    d = Integer().const(6)
    e = Integer().const(9)
    f = Integer().const(12)
    g = c + d if a & b else e + f
    h = g + g if a | b else Integer().const(15)


def test_if_exps():
    state = State()
    with state:
        if_exps.virtualized.call_transformed()

    print(f"\ntest_if_exps")
    print(state)


@argon_function(calls=False, if_exps=False)
def ifs():
    a = Boolean().const(True)
    b = Boolean().const(False)

    c = Integer().const(3)
    d = Integer().const(6)
    e = Integer().const(9)
    f = Integer().const(12)
    g = Integer().const(15)
    h = Integer().const(18)
    i = Integer().const(21)
    j = Integer().const(24)
    k = Integer().const(27)
    l = Integer().const(30)

    if a & b:
        m = c + d
    else:
        m = e + f

    if a | b:
        m = g + h
    else:
        if a ^ b:
            m = i + j
        else:
            m = k + l


def test_ifs():
    state = State()
    with state:
        ifs.virtualized.call_transformed()

    print(f"\ntest_ifs")
    print(state)
