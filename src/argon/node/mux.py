import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.op import Op
from argon.ref import Sym
from argon.types.boolean import Boolean


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Mux[T](Op[T]):
    cond: Boolean
    a: T
    b: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.cond, self.a, self.b]  # type: ignore