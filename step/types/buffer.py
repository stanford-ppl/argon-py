from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Union, Tuple, override, List, TypeVar, Any, Generic
from argon.ref import Ref
from .rankgen import RankGen
import numpy as np
from numpy.typing import NDArray

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
        assert RankGen().get_rank(c.value.ndim) == self.BRK
        return super().const(c)