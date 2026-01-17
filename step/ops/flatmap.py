import typing
import pydantic
from argon.ref import Exp, Op, Sym
from typing import Tuple, Callable


from pydantic.dataclasses import dataclass
from ..types.stream import Stream


A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
B = typing.TypeVar("B", bound=Exp[typing.Any, typing.Any], covariant=True)
a = typing.TypeVar("a", bound=Exp[typing.Any, typing.Any], covariant=True)
b = typing.TypeVar("b", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Flatmap[a, b, A, B](Op[Stream[B, a]]):
    func: Callable[[A], Stream[B, b]]
    stream: Stream[A, a]
    
    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.func, self.stream]  # type: ignore

# R = typing.TypeVar("R", bound=Exp[typing.Any, typing.Any], covariant=True)

# class Range[R](Op[Stream[B, a]]):
#     func: R
    
#     @property
#     @typing.override
#     def inputs(self) -> typing.List[Sym[typing.Any]]:
#         return [self.func, self.stream]  # type: ignore
    
# class Streamify[A, a]:
#     func: R
    
#     @property
#     @typing.override
#     def inputs(self) -> typing.List[Sym[typing.Any]]:
#         return [self.func, self.stream]  # type: ignore