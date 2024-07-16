import typing
import pydantic
from argon.ref import Exp, Op, Sym
from typing import Tuple, Callable

from types import FunctionType


from pydantic.dataclasses import dataclass
from ..types.stream import Stream, HStream


A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
B = typing.TypeVar("B", bound=Exp[typing.Any, typing.Any], covariant=True)
a = typing.TypeVar("a", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Map[A, B, a](Op[Stream[B, a]]):
    stream: HStream[A, B, a]
    func: Callable[[A], B]
    
    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.stream, self.func]  # type: ignore

# P = typing.TypeVar("P", bound=Exp[typing.Any, typing.Any], covariant=True)

# class Permute[P]:
#     stream: Stream[A, a]
#     func: Callable[[A], B]
    
#     @property
#     @typing.override
#     def inputs(self) -> typing.List[Sym[typing.Any]]:
#         return [self.stream, self.func]  # type: ignore