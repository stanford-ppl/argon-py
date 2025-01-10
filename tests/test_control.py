from argon.state import State
from argon.types.integer import Integer
from argon.virtualization.wrapper import argon_function
from collections import namedtuple


@argon_function(calls=False, ifs=False)
def if_exps():
    a = True
    b = False
    c = True
    d = 3
    e = 6
    f = 9
    g = 12
    h = d + e if a & b & c else f + g
    i = h + h if a | b | c else 15


def test_if_exps():
    state = State()
    with state:
        if_exps.virtualized.call_transformed()

    print(f"\ntest_if_exps")
    print(state)


def my_func(x, y):
    return x + y


@argon_function(calls=True, if_exps=False)
def ifs():
    a = True
    b = False
    c = True
    d = 3
    e = 6
    f = 9
    g = 12
    h = 15
    i = 18
    j = 21
    k = 24

    if a & b & c:
        l = my_func(d, e)
        k = 35 if a | b | c else 33

    if a | b | c:
        l = f + g
    else:
        if a ^ b ^ c:
            l = h + i
        else:
            l = j + k


def test_ifs():
    state = State()
    with state:
        ifs.virtualized.call_transformed()

    print(f"\ntest_ifs")
    print(state)


@argon_function(loops=True)
def loops():
    a = 0
    b = 1
    c = 2 if True else 3
    while a < 10:
        a = 1
        a = a + 1
        a = a + 2
        b = b + 2
        d = c + 1
        e = 1
        f = e + 1


def test_loops():
    state = State()
    with state:
        loops.virtualized.call_transformed()

    print(f"\ntest_loops")
    print(state)
