import typing
import pydantic
from argon.ref import Exp, Op, Sym
from typing import Tuple


from pydantic.dataclasses import dataclass
from ..types.stream import Stream, Index


A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
a = typing.TypeVar("a", bound=Exp[typing.Any, typing.Any], covariant=True)
b = typing.TypeVar("b", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Flatten[a, A, b](Op[Stream[A, b]]):
    stream: Stream[A, a]
    dims: Tuple[Index, ...]
    
    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.stream, self.dims]  # type: ignore