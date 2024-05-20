import typing
import pydantic
from argon.ref import Exp, Op, Sym

# from argon.types.integer import Integer
from pydantic.dataclasses import dataclass


T = typing.TypeVar("T", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Undefined[T](Op[T]):
    """
    The Undefined[T] operation represents a variable that has not been defined
    yet.

        name : str
            The name of the variable that has not been defined yet.
    """

    name: str

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return []  # type: ignore

    @typing.override
    def dump(self, indent_level=0) -> str:
        return f"Undefined('{self.name}')"
