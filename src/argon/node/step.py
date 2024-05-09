import typing
import pydantic
from argon.ref import Exp, Op, Sym

from pydantic.dataclasses import dataclass

T = typing.TypeVar("T", bound=Exp[typing.Any, typing.Any], covariant=True)

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Zip[T](Op[T]):
    a: T
    b: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore
 