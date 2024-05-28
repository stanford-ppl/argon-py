from argon.state import State
from argon.types.stream import Stop, Val, Stream, gen_rank
from argon.types.buffer import Ndarray, Buffer
from typing import Union, Tuple, override, List, TypeVar
import numpy as np
import typing


# Buffer tests
def test_buffer():
    state = State()
    with state:
        R1 = gen_rank(1)
        a = Buffer[int,R1]().const(Ndarray[int](np.ndarray([2])))
        assert a.C == Ndarray[int]
        assert a.A is Buffer[int, R1]
        assert a.BT is int
        assert a.A().BT is int


# Stream tests
def test_int():
    state = State()
    with state:
        R1 = gen_rank(1)
        a = Stream[float, R1]().const(
            [Val(1.0), Val(2.0), Stop(1), Val(3.0), Val(4.0), Stop(2)]
        )
        print(a.C)
        assert a.C == typing.List[typing.Union[Val[float], Stop]]
        assert a.A is Stream[float, R1]
        assert a.VT is float
        assert a.RK is R1
        assert a.A().VT is float
        assert a.A().RK is R1
    print(state)


# def test_float():
#     state = State()
#     with state:
#         a = Stream[float]().const(
#             [Val(1.5), Val(2.5), Stop(1), Val(3.5), Val(4.5), Stop(2)]
#         )
#         assert a.C == typing.List[typing.Union[Val, Stop]]
#         assert a.A is Stream[float]
#         assert a.T is float
#         assert a.A().T is float
#     print(state)

# def test_tuple():
#     state = State()
#     with state:
#         a = Stream[Tuple[int]]().const(
#             [Tuple[Val(1.0), Val(2.0)], Stop(1), Tuple[Val(3.0), Val(4.0)], Stop(2)]
#         )
#         assert a.C == typing.List[typing.Union[Val, Stop]]
#         assert a.A is Stream[Tuple[int]]
#         assert a.T is Tuple[int]
#         assert a.A().T is Tuple[int]
#     print(state)

# def test_buffer_in_stream():
#     state = State()
#     with state:
#         a = Buffer[int]().const(np.ndarray([1]))
#         b = Buffer[int]().const(np.ndarray([2]))
#         c = Stream[Buffer[int]]().const(
#             [Val(a), Stop(1), Val(b), Stop(2)]
#         )
#         assert c.C == typing.List[typing.Union[Val, Stop]]
#         assert c.A is Stream[Buffer[int]]
#         assert c.VT is Buffer[int]
#         assert c.A().VT is Buffer[int]
#     print(state)
