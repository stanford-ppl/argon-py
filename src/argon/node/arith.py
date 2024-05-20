import typing
import pydantic
from argon.ref import Exp, Op, Sym

# from argon.types.integer import Integer
from pydantic.dataclasses import dataclass


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
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore
