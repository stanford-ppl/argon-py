from argon.state import State
from argon.types.integer import Integer
from argon.types.boolean import Boolean
from argon.virtualization.wrapper import argon_function


def test_scope():
    state = State()
    with state:
        a = Integer().const(3)
        b = Integer().const(6)
        c = a + b
        d = c + b
    print(state)

    print(f"\ntest_scope")
    print(state)


def test_scope2():
    state = State()
    with state:
        a = Boolean().const(True)
        b = Boolean().const(False)
        c = a & b
        d = c | b
        e = ~d
        f = e ^ a

    print(f"\ntest_scope2")
    print(state)


@argon_function
def test_scope3():
    state = State()
    with state:
        a = Boolean().const(True)
        b = Boolean().const(False)
        c = Integer().const(3)
        d = Integer().const(6)
        e = Integer().const(9)
        f = Integer().const(12)
        g = c + d if a & b else e + f
        h = g if a | b else Integer().const(15)

    print(f"\ntest_scope3")
    print(state)


@argon_function
def test_scope4():
    state = State()
    with state:
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

    print(f"\ntest_scope4")
    print(state)
