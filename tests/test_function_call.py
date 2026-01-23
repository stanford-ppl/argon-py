import typing
from argon.state import State
from argon.virtualization.wrapper import argon_function


def func3(x: int, y: int) -> int:
    return x + y


def func2() -> typing.Callable[[int, int], int]:
    return func3


def func1(
    func: typing.Callable[[], typing.Callable[[int, int], int]], x: int, y: int, z: int
) -> int:
    return func()(x, y) + z


@argon_function()
def function_call() -> int:
    return func1(func2, 3, 4, 5)


def test_function_call():
    state = State()
    with state:
        function_call.virtualized.call_transformed()
    print(state)


def recursion_call(x: int) -> int:
    if x < 0:
        output = 0
    else:
        output = recursion_call(x - 1) + 1
    return output


@argon_function()
def recursion() -> int:
    return recursion_call(5)


def test_recursion():
    state = State()
    with state:
        recursion.virtualized.call_transformed()
    print(state)
