from pydantic.dataclasses import dataclass
from typing import Union, override
from argon.ref import Ref
# from argon.srcctx import SrcCtx
# from argon.state import stage

@dataclass
class Stop:
    level: int

    def __str__(self) -> str:
        return f"S{self.level}"
    
@dataclass
class Val:
    value: float

    def __str__(self) -> str:
        return str(self.value)
    
@dataclass
class Token:
    value: Union[Val,Stop]

    def __str__(self) -> str:
        return str(self.value)



class Stream(Ref[list, "Stream"]):
    @override
    def fresh(self) -> "Stream":
        return Stream()

    # def __add__(self, other: "Integer") -> "Integer":
    #     import argon.node.arith as arith

    #     return stage(arith.Add[Integer](self, other), ctx=SrcCtx.new(2))
