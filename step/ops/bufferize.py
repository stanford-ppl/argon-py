import typing
import pydantic
from argon.ref import Exp, Op, Sym
from typing import Tuple


from pydantic.dataclasses import dataclass
from ..types.stream import Stream
from ..types.buffer import Buffer


A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
a = typing.TypeVar("a", bound=Exp[typing.Any, typing.Any], covariant=True)
b = typing.TypeVar("b", bound=Exp[typing.Any, typing.Any], covariant=True)
d = typing.TypeVar("d", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Bufferize[A, a, b, d](Op[Stream[Buffer[A, a], d]]):
    stream: Stream[A, b]

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.stream]  # type: ignore