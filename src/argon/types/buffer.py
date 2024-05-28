from pydantic.dataclasses import dataclass
from typing import Union, Tuple, override, List, TypeVar
from argon.ref import Ref
import numpy as np

T = TypeVar("T", int, float, Tuple[int], Tuple[float])
@dataclass
class Element:
    value: T

    def __str__(self) -> str:
        return str(self.value)

BT = TypeVar("BT")

class Buffer[BT](Ref[np.ndarray, "Buffer[BT]"]):

    @override
    def fresh(self) -> "Buffer[BT]":
        return Buffer[self.BT]()