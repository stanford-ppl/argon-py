from pydantic.dataclasses import dataclass
from typing import Union, override, List, Any, TypeVar, Optional, Tuple
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage
import numpy as np
    
@dataclass
class Level:
    _format: Optional[str] = None
    _coords: Optional[List[Any]] = None

    def __str__(self) -> str:
        return f"(Format: {self._format}, Coords: {self._coords})"

@dataclass
class TensorStorage:
    value: Optional[List[Any]] = None
    levels: Optional[List[Level]] = None
    shape: Optional[Tuple[int, ...]] = None

    def __str__(self) -> str:
        return str(self)

T = TypeVar("T")
class Tensor[T](Ref[TensorStorage, "Tensor[T]"]):

    @override
    def fresh(self) -> "Tensor[T]":
        print(f"print self = {repr(self)}")
        # freshobj = fStream(self.rank)
        # use self to set the corresponding fields for Stream
        return Tensor[self.T]()
    
    def matmul(self, other:"Tensor[T]") -> "Tensor[T]":

        return stage(None, ctx=SrcCtx.new(2))