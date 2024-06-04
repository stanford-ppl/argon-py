from argon.state import State
from step.types.stream import Stop, Val, Stream
from step.types.buffer import RankGen, Ndarray, Buffer
from typing import Tuple, List, Union
import numpy as np
import typing

# Op tests
# def test_zip():
#     state = State()
#     with state:
#         R1 = RankGen().get_rank(1)
#         a = Stream[int,R1]().const(
#             [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
#         )
#         b = Stream[int,R1]().const(
#             [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
#         )
#         c = a.zip(b)
#         print(c)
        
        
        
# Buffer tests
def test_buffer():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Buffer[int,R1]().const(Ndarray[int](np.ndarray([2])))
        assert a.C == Ndarray[int]
        assert a.A is Buffer[int, R1]
        assert a.BT is int
        assert a.A().BT is int


# Stream tests
def test_int():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Stream[int,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        assert a.C == typing.List[typing.Union[Val[int], Stop]]
        assert a.A is Stream[int,R1]
        assert a.ST is int
        assert a.SRK is R1
        assert a.A().ST is int
        assert a.A().SRK is R1
    print(state)


def test_float():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Stream[float,R1]().const(
            [Val(1.5), Val(2.5), Stop(1), Val(3.5), Val(4.5), Stop(2)]
        )
        assert a.C == typing.List[typing.Union[Val[float], Stop]]
        assert a.A is Stream[float,R1]
        assert a.ST is float
        assert a.SRK is R1
        assert a.A().ST is float
        assert a.A().SRK is R1
    print(state)

def test_tuple():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Stream[Tuple[int],R1]().const(
            [Val[Tuple[1,2]], Stop(1), Val[Tuple[3,4]], Stop(2)]
        )
        assert a.C == typing.List[typing.Union[Val[Tuple[int]], Stop]]
        assert a.A is Stream[Tuple[int],R1]
        assert a.ST is Tuple[int]
        assert a.SRK is R1
        assert a.A().ST is Tuple[int]
        assert a.A().SRK is R1
    print(state)

def test_buffer_in_stream():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Buffer[int,R1]().const(Ndarray[int](np.ndarray([1])))
        b = Buffer[int,R1]().const(Ndarray[int](np.ndarray([2])))
        R2 = RankGen().get_rank(2)
        c = Stream[Buffer[int,R1],R2]().const(
            [Val(a), Stop(1), Val(b), Stop(2)]
        )
        assert c.C == typing.List[typing.Union[Val[Buffer[int,R1]], Stop]]
        assert c.A is Stream[Buffer[int,R1],R2]
        assert c.ST is Buffer[int,R1]
        assert c.SRK is R2
        assert c.A().ST is Buffer[int,R1]
        assert c.A().SRK is R2
    print(state)