from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Union, Tuple, override, List, TypeVar, Any
from argon.ref import Ref
import numpy as np

T = TypeVar("T")

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Ndarray[T]:
    value: np.ndarray[T]


class RankGen:
    generated_ranks = {"1": type("R1", (), {}), "2": type("R2", (), {})}

    def get_rank(self, c: int) -> Any:
        return self.generated_ranks[str(c)]
    
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