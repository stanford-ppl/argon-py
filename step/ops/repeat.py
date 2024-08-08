import typing
import pydantic
from argon.ref import Exp, Op, Sym
from typing import Tuple, Union, Any


from pydantic.dataclasses import dataclass
from ..types.stream import Stream, RStream
from ..types.rankgen import RankGen


A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
b = typing.TypeVar("b", bound=Exp[typing.Any, typing.Any], covariant=True)
c = typing.TypeVar("c", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Repeat[A, b, c](Op[Stream[A, c]]):
    stream1: RStream[A, Any, b, c]
    stream2: RStream[A, Any, b, c]
    
    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.stream1, self.stream2]  # type: ignore