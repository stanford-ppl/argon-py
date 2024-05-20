import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.op import Op
from argon.ref import Sym
from argon.types.boolean import Boolean


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Mux[T](Op[T]):
    """
    The Mux[T] operation represents a multiplexer that selects between two values
    based on a condition.

        cond : Boolean
            The condition that determines which value to select.
        a : T
            The value to select when the condition is True.
        b : T
            The value to select when the condition is False.
    """

    cond: Boolean
    a: T
    b: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.cond, self.a, self.b]  # type: ignore
