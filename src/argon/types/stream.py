from pydantic.dataclasses import dataclass
from typing import Union, Tuple, override, List, TypeVar
from argon.ref import Ref

@dataclass
class Stop:
    level: int

    def __str__(self) -> str:
        return f"S{self.level}"

T = TypeVar("T")
@dataclass
class Val[T]:
    value: T

    def __str__(self) -> str:
        return str(self.value)

VT = TypeVar("VT")
RK = TypeVar("RK")
class Stream[VT,RK](Ref[List[Union[Val[VT], Stop]], "Stream[VT,RK]"]):

    @override
    def fresh(self) -> "Stream[VT,RK]":
        return Stream[self.VT, self.RK]()
    
def gen_rank(rank: int):
    rank_name = 'R'+str(rank)
    return type(rank_name, (), dict(rank=rank))