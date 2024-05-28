from pydantic.dataclasses import dataclass
from typing import Union, Tuple, override, List, TypeVar
from argon.ref import Ref
from argon.types.buffer import Element, Buffer

@dataclass
class Stop:
    level: int

    def __str__(self) -> str:
        return f"S{self.level}"

T = TypeVar("T", Element, Buffer, Tuple[Buffer])
@dataclass
class Val:
    value: T

    def __str__(self) -> str:
        return str(self.value)

VT = TypeVar("VT")
class Stream[VT](Ref[List[Union[Val, Stop]], "Stream[VT]"]):

    @override
    def fresh(self) -> "Stream[VT]":
        return Stream[self.VT]()