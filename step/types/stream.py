from pydantic.dataclasses import dataclass
from typing import Union, Tuple, override, List, TypeVar
from argon.ref import Ref
from argon.state import stage
from argon.srcctx import SrcCtx


@dataclass
class Stop:
    level: int

    def __str__(self) -> str:
        return f"S{self.level}"

VT = TypeVar("VT")

@dataclass
class Val[VT]:
    value: VT

    def __str__(self) -> str:
        return str(self.value)

ST = TypeVar("ST")
SRK = TypeVar("SRK")
B = TypeVar("B")

class Stream[ST,SRK](Ref[List[Union[Val[ST], Stop]], "Stream[ST,SRK]"]):

    @override
    def fresh(self) -> "Stream[ST,SRK]":
        return Stream[self.ST, self.SRK]()
    
    def zip(self, other: "Stream[B,SRK]") -> "Stream[(ST,B),SRK]":
        import step.ops.zip as zip
        
        return stage(zip.Zip[ST,B,SRK](self, other), ctx=SrcCtx.new(2))