from pydantic.dataclasses import dataclass
from typing import Union, override, List, TypeVar
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import State, stage
import typing

@dataclass
class Stop:
    level: int

    def __str__(self) -> str:
        return f"S{self.level}"
    
@dataclass
class FVal:
    value: float

    def __str__(self) -> str:
        return str(self.value)

T = TypeVar("T")
class FStream[T](Ref[List[Union[FVal,Stop]], "FStream[T]"]):

    @override
    def fresh(self) -> "FStream[T]":
        # print(f"print self = {repr(self)}")
        # print(f"From fresh: {self.T}")
        
        # freshobj = fStream(self.rank)
        # use self to set the corresponding fields for Stream
        return FStream[self.T]()
    
    def zip(self, other:"FStream[T]") -> "FStream[T]":
        import argon.node.step as step

        return stage(step.Zip[FStream[T]](self, other), ctx=SrcCtx.new(2))

U = TypeVar("U")
class UStream[U](Ref[List[Union[FVal,Stop]], "UStream[int]"]):

    @override
    def fresh(self) -> "UStream[U]":
        # print(f"print self = {repr(self)}")
        # print(f"From fresh: {self.T}")
        
        # freshobj = fStream(self.rank)
        # use self to set the corresponding fields for Stream
        return UStream[self.U]()


TP = TypeVar("TP")
class GStream[TP](Ref[List[TP], "GStream[TP]"]):

    @override
    def fresh(self) -> "GStream[TP]":
        # print(f"print self = {repr(self)}")
        # print(f"From fresh: {self.T}")
        
        # freshobj = fStream(self.rank)
        # use self to set the corresponding fields for Stream
        return GStream[self.TP]()



def test_stop_token():
    a = Stop(1)
    print(a)
    assert type(a) == Stop

def test_fixed_tp_stream():
    state = State()
    with state:
        a = FStream[int]().const([FVal(1.0),FVal(2.0),Stop(1),FVal(3.0),FVal(4.0),Stop(2)])
        assert a.C == typing.List[typing.Union[FVal,Stop]]
        assert a.A is FStream[int]
        assert a.T is int
        assert a.A().T is int
    print(state)

def test_debug_generic1():
    state = State()
    with state:
        a = UStream[str]()
        print(a)
        assert type(a) is UStream
        assert a.C == typing.List[typing.Union[FVal,Stop]]
        assert a.A is UStream[int]
        assert a.U is str
    print(state)

def test_debug_generic2():
    state = State()
    with state:
        a = UStream[str]().const([FVal(1.0),FVal(2.0),Stop(1),FVal(3.0),FVal(4.0),Stop(2)])
        # This becomes type A (i.e., UStream[int] because const returns type A)

        print(a)                                        # Const([FVal(value=1.0), FVal(value=2.0), Stop(level=1), FVal(value=3.0), FVal(value=4.0), Stop(level=2)])
        assert type(a) is UStream
        assert a.C == typing.List[typing.Union[FVal,Stop]]
        assert a.A is UStream[int]
        assert a.A().U is int       # -- a.A() is equivalent to UStream[int]()
        assert a.U is int
    print(state)


def test_gstream():
    state = State()
    with state:
        a = GStream[int]().const([1,2,3])
        assert type(a) is GStream
        assert a.C == typing.List[int]
        assert a.A is GStream[int]
        assert a.A().TP is int       # -- a.A() is equivalent to UStream[int]()
        assert a.TP is int
        
        print(f"a.C = {a.C}")
    print(state)
