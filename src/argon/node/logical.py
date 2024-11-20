import typing
import pydantic
from argon.ref import Exp, Op, Sym

from pydantic.dataclasses import dataclass

T = typing.TypeVar("T", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Not[T](Op[T]):
    """
    The Not[T] operation represents a logical NOT operation.

        a : T
            The value to negate.
    """

    a: T

    @property
    @typing.override
    def operands(self) -> typing.List[Sym[typing.Any]]:
        return [self.a]  # type: ignore


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class And[T](Op[T]):
    """
    The And[T] operation represents a logical AND operation.

        a : T
            The first value to compare.
        b : T
            The second value to compare.
    """

    a: T
    b: T

    @property
    @typing.override
    def operands(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Or[T](Op[T]):
    """
    The Or[T] operation represents a logical OR operation.

        a : T
            The first value to compare.
        b : T
            The second value to compare.
    """

    a: T
    b: T

    @property
    @typing.override
    def operands(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Xor[T](Op[T]):
    """
    The Xor[T] operation represents a logical XOR operation.

        a : T
            The first value to compare.
        b : T
            The second value to compare.
    """

    a: T
    b: T

    @property
    @typing.override
    def operands(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore
