import typing
import pydantic
from argon.ref import Exp, Op, Sym
from typing import Tuple, Callable


from pydantic.dataclasses import dataclass
from ..types.stream import Stream, RStream


A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
B = typing.TypeVar("B", bound=Exp[typing.Any, typing.Any], covariant=True)
a = typing.TypeVar("a", bound=Exp[typing.Any, typing.Any], covariant=True)
b = typing.TypeVar("b", bound=Exp[typing.Any, typing.Any], covariant=True)
d = typing.TypeVar("d", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Accum[A, B, a, b, d](Op[Stream[B, d]]):
    stream: RStream[A, B, a, b]
    func: Callable[[A,B], B]

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.stream, self.func]  # type: ignore