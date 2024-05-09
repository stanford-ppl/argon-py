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
class Val:
    value: float

    def __str__(self) -> str:
        return str(self.value)
    
@dataclass
class Token:
    value: Union[Val,Stop]

    def __str__(self) -> str:
        return str(self.value)


class Stream(Ref[List[Token], "Stream"]):
    @override
    def fresh(self) -> "Stream":
        return Stream()
    
    def zip(self, other:"Stream") -> "Stream":
        import argon.node.step as step

        return stage(step.Zip[Stream](self, other), ctx=SrcCtx.new(2))


