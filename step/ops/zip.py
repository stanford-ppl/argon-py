import typing
import pydantic
from argon.ref import Exp, Op, Sym
from typing import Tuple


from pydantic.dataclasses import dataclass
from ..types.stream import Stream, HStream


A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
B = typing.TypeVar("B", bound=Exp[typing.Any, typing.Any], covariant=True)
b = typing.TypeVar("b", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Zip[A, B, b](Op[Stream[Tuple[A,B], b]]):
    stream1: HStream[A, B, b]
    stream2: HStream[A, B, b]

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.stream1, self.stream2]  # type: ignore