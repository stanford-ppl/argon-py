from pydantic.dataclasses import dataclass
from typing import Union, override, List, TypeVar
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage

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
