import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.block import Block
from argon.op import Op
from argon.ref import Exp, Sym
from argon.types.boolean import Boolean
from argon.types.null import Null


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class IfThenElse[T](Op[T]):
    """
    The IfThenElse[T] operation represents an if-then-else control flow construct
    with return type T.

        cond : Boolean
            The condition to check.
        thenBlk : Block[T]
            The block to execute if the condition is true.
        elseBlk : Block[T]
            The block to execute if the condition is false.
    """

    condBlk: Block[Boolean]
    thenBlk: Block[T]
    elseBlk: Block[T]

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return self.condBlk.inputs + self.thenBlk.inputs + self.elseBlk.inputs  # type: ignore

    @typing.override
    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        return (
            f"IfThenElse( \n"
            f"{indent}condBlk = {self.condBlk.dump(indent_level + 1)}, \n"
            f"{indent}thenBlk = {self.thenBlk.dump(indent_level + 1)}, \n"
            f"{indent}elseBlk = {self.elseBlk.dump(indent_level + 1)} \n"
            f"{no_indent})"
        )
