from argon.state import State
from step.types.stream import Stop, Val, Index, Stream, RStream, HStream
from step.types.buffer import  Ndarray, Buffer
from step.types.rankgen import RankGen
# from step.ops.zip import Zip
from typing import Tuple, List, Union
import numpy as np
import typing

# Composite tests

# Op tests
def test_map():
    def func(x: int, y: int) -> int:
        return x + y
    
    def func2(x: int, y: float) -> float:
        return x + y
    
    state = State()
    with state:
        R2 = RankGen().get_rank(2)
        a = HStream[int,int,R2]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.map(lambda val: func(val, 1))
        print(b.A)
        c = HStream[int,float,R2]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        d = c.map(lambda val: func2(val, 1.0))
        print(d.A)
    print(state)


def test_accum():
    def func(x: int, y: int) -> int:
        return x + y
    
    state = State()
    with state:
        R1 = RankGen().get_rank(2)
        R2 = RankGen().get_rank(1)
        a = RStream[int,int,R1,R2]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.accum(lambda val: func(val, 1))
        print(b.A)
    print(state)


def test_zip():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = HStream[int,int,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = HStream[int,int,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        c = a.zip(b)
        print(c.A)
        d = HStream[int,float,R1]().const(
            [Val(1.0), Val(2.0), Stop(1), Val(3.0), Val(4.0), Stop(2)]
        )
        e = a.zip(d)
        print(e.A)
    print(state)
    

def test_repeat():
    state = State()
    with state:
        R0 = RankGen().get_rank(0)
        R1 = RankGen().get_rank(1)
        a = RStream[int,int,R1,R0]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.repeat(RStream[int,int,R1,R0]())
    print(state)


def test_bufferize():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Stream[int,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.bufferize(1)
    print(state)


def test_promote():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Stream[int,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.promote(1)
    print(state)


def test_reshape():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Stream[int,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.reshape((Index(1), Index(2)), (1,1))
        print(b.A)
        print(b.SRK)
    print(state)


def test_flatten():
    state = State()
    with state:
        R3 = RankGen().get_rank(3)
        a = Stream[int,R3]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.flatten((Index(1),))
        print(b.A)
    print(state)
    

def test_enumerate():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Stream[int,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = a.enumerate(1)
        print(b.A)
    print(state)


def test_partition():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = RStream[int,int,R1,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        b = RStream[int,int,R1,R1]().const(
            [Val(1), Val(2), Stop(1), Val(3), Val(4), Stop(2)]
        )
        c = a.partition(3, b)
        print(c.A)
    print(state)


# Buffer tests
def test_buffer():
    state = State()
    with state:
        R1 = RankGen().get_rank(1)
        a = Buffer[int,R1]().const(Ndarray[int](np.array([2])))
        assert a.C == Ndarray[int]
        assert a.A is Buffer[int, R1]
        assert a.BT is int
        assert a.A().BT is int
    print(state)


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
        a = Stream[Tuple[int,int],R1]().const(
            [Val((1,2)), Stop(1), Val((3,4)), Stop(2)]
        )
        assert a.C == typing.List[typing.Union[Val[Tuple[int,int]], Stop]]
        assert a.A is Stream[Tuple[int,int],R1]
        assert a.ST is Tuple[int,int]
        assert a.SRK is R1
        assert a.A().ST is Tuple[int,int]
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
        #assert c.C == typing.List[typing.Union[Val[Buffer[int,R1]], Stop]]
        assert c.A is Stream[Buffer[int,R1],R2]
        assert c.ST is Buffer[int,R1]
        assert c.SRK is R2
        assert c.A().ST is Buffer[int,R1]
        assert c.A().SRK is R2
    print(state)