import typing
import pydantic
from argon.ref import Exp, Op, Sym

from pydantic.dataclasses import dataclass

T = typing.TypeVar("T", bound=Exp[typing.Any, typing.Any], covariant=True)

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Not[T](Op[T]):
    a: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a] # type: ignore

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class And[T](Op[T]):
    a: T
    b: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Or[T](Op[T]):
    a: T
    b: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Xor[T](Op[T]):
    a: T
    b: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b] # type: ignore