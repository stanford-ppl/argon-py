
import typing
import pydantic
from argon.ref import Op, Sym
from argon.types.integer import Integer
from pydantic.dataclasses import dataclass

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class IntegerAdd(Op[Integer]):
    a: Integer
    b: Integer

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]
