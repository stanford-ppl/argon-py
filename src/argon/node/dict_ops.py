import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.op import Op
from argon.ref import Exp


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Get[T](Op[T]):
    dictionary: Exp
    # TODO: Only restricting keys to str for now
    key: str

    @property
    @typing.override
    def operands(self) -> typing.List[Exp[typing.Any, typing.Any]]:
        return [self.dictionary, self.key]

