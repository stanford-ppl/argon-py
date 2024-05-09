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
class fVal:
    value: float

    def __str__(self) -> str:
        return str(self.value)


class fStream(Ref[List[Union[fVal,Stop]], "fStream"]):
    # Add field rank & datatype (like int as a field and the type is type())
    # This does not appear in the types, but maybe based on how the dataclasses are compared, I cant do a == to see if they're the same?
    @override
    def fresh(self) -> "fStream":
        # use self to set the corresponding fields for Stream
        return fStream()
    
    def zip(self, other:"fStream") -> "fStream":
        import argon.node.step as step

        return stage(step.Zip[fStream](self, other), ctx=SrcCtx.new(2))
