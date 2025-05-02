import typing
from argon.state import State
from argon.virtualization.wrapper import argon_function


def func2(x: int, y: int) -> int:
    return x + y


def func1(func: typing.Callable[[int, int], int], x: int, y: int, z: int) -> int:
    return func(x, y) + z


@argon_function()
def function_call() -> int:
    return func1(func2, 3, 4, 5)


def test_function_call():
    state = State()
    with state:
        function_call.virtualized.call_transformed()
    print(state)
