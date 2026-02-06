import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.op import Op
from argon.ref import Exp
from argon.types.integer import Integer


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Index[T](Op[T]):
    """
    The Index[T] operation represents indexed access into a collection.

        collection : Exp[Any, Any]
            The collection being indexed (e.g., NNModuleList)
        index : Integer
            The index to access
    """

    collection: Exp[typing.Any, typing.Any]
    index: Integer

    @property
    @typing.override
    def operands(self) -> typing.List[Exp[typing.Any, typing.Any]]:
        return [self.collection, self.index]

    @typing.override
    def dump(self, indent_level=0) -> str:
        return f"Index({self.collection}, {self.index})"
