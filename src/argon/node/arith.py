import typing
import pydantic
from argon.ref import Exp, Op, Sym

from pydantic.dataclasses import dataclass

from argon.types.boolean import Boolean


T = typing.TypeVar("T", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Add[T](Op[T]):
    """
    The Add[T] operation represents an addition operation.

        a : T
            The first value to add.
        b : T
            The second value to add.
    """

    a: T
    b: T

    @property
    @typing.override
    def operands(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Sub[T](Op[T]):
    """
    The Sub[T] operation represents a subtraction operation.

        a : T
            The value to subtract from.
        b : T
            The value to subtract.
    """

    a: T
    b: T

    @property
    @typing.override
    def operands(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class GreaterThan[T](Op[Boolean]):
    """
    The GreaterThan[T] operation represents a greater than comparison.

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
        return [self.a, self.b]


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class LessThan[T](Op[Boolean]):
    """
    The LessThan[T] operation represents a less than comparison.

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
        return [self.a, self.b]
