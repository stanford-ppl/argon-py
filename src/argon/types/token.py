from pydantic.dataclasses import dataclass
from typing import Union, override
from argon.ref import Ref

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



class Stream(Ref[list[Token], "Stream"]):
    @override
    def fresh(self) -> "Stream":
        return Stream()
