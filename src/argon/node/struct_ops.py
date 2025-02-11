import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.op import Op
from argon.ref import Exp
from argon.types.struct import Struct


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Get[T](Op[T]):
    # Note: The proper type should be a Struct whose type parameter contains an element T
    struct: Struct
    # TODO: Only restricting keys to str for now
    key: str

    @property
    @typing.override
    def operands(self) -> typing.List[Exp[typing.Any, typing.Any]]:
        return [self.struct, self.key]
