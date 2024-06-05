from argon.state import State
from argon.virtualization.wrapper import argon_function


def func(x, y):
    return x + y


@argon_function()
def function_call():
    return func(3, 4) + 5


def test_function_call():
    state = State()
    with state:
        function_call.virtualized.call_transformed()
    print(state)
