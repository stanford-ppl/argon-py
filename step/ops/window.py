# import typing
# import pydantic
# from argon.ref import Exp, Op, Sym
# from typing import Tuple


# from pydantic.dataclasses import dataclass
# from ..types.stream import Stream

# A = typing.TypeVar("A", bound=Exp[typing.Any, typing.Any], covariant=True)
# N = typing.TypeVar("N", bound=Exp[typing.Any, typing.Any], covariant=True)
# S = typing.TypeVar("S", bound=Exp[typing.Any, typing.Any], covariant=True)
# a = typing.TypeVar("a", bound=Exp[typing.Any, typing.Any], covariant=True)
# b = typing.TypeVar("b", bound=Exp[typing.Any, typing.Any], covariant=True)


# @dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
# class Window[a, b, A, N, S](Op[Stream[A, a+b]]):
#     stream: Stream[a, a]
    
#     @property
#     @typing.override
#     def inputs(self) -> typing.List[Sym[typing.Any]]:
#         return [self.stream]  # type: ignore