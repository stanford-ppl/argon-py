from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Union, Tuple, override, List, TypeVar
from argon.ref import Ref
from argon.types.stream import gen_rank
import numpy as np
import numpy.typing as npt

T = TypeVar("T")

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Ndarray[T]:
    value: np.ndarray[T]
    
    
BT = TypeVar("BT")
BRK = TypeVar("BRK")

class Buffer[BT,BRK](Ref[Ndarray[BT], "Buffer[BT,BRK]"]):

    @override
    def fresh(self) -> "Buffer[BT,BRK]":
        return Buffer[self.BT, self.BRK]()
    
    @override
    def const(self, c: Ndarray[BT]) -> "Buffer[BT,BRK]":
        assert type(gen_rank(c.value.shape)) == type(self.BRK)
        return super().const(c)