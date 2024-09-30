from argon.state import State
from argon.virtualization.wrapper import argon_function


@argon_function(calls=False, ifs=False)
def if_exps():
    a = True
    b = False
    c = 3
    d = 6
    e = 9
    f = 12
    g = c + d if a & b else e + f
    h = g + g if a | b else 15


def test_if_exps():
    state = State()
    with state:
        if_exps.virtualized.call_transformed()

    print(f"\ntest_if_exps")
    print(state)


@argon_function(calls=False, if_exps=False)
def ifs():
    a = True
    b = False
    c = 3
    d = 6
    e = 9
    f = 12
    g = 15
    h = 18
    i = 21
    j = 24
    k = 27
    l = 30

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
