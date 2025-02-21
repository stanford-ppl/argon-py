import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.block import Block
from argon.op import Op
from argon.ref import Exp
from argon.types.function import FunctionWithVirt


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class FunctionNew[T](Op[T]):
    """
    The FunctionNew[T] operation represents a new function with return type T.

        name : str
            The name of the function.
        binds : List[Exp[Any]]
            The binds of the function. These will specify the parameters of the function with their types.
        body : Block[T]
            The body of the function.
        virtualized : FunctionWithVirt
            The virtualized function to be used to create the Argon function.
    """

    name: str
    binds: typing.List[Exp[typing.Any, typing.Any]]
    body: Block
    virtualized: typing.Optional[FunctionWithVirt] = None

    @property
    @typing.override
    def operands(self) -> typing.List[Exp[typing.Any, typing.Any]]:
        # TODO: figure out what inputs I should use
        return []

    @typing.override
    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        more_indent = "|   " * (indent_level + 2)
        if self.binds:
            binds_str = ", \n".join(
                f"{more_indent}{bind.dump(indent_level + 2)}" for bind in self.binds
            )
            binds_str = f"[\n{binds_str}\n{indent}]"
        else:
            binds_str = "[]"
        return (
            f"FunctionNew( \n"
            f"{indent}Name = {self.name}, \n"
            f"{indent}Binds = {binds_str}, \n"
            f"{indent}Body = {self.body.dump(indent_level + 1)}, \n"
            f"{indent}Virtualized = {self.virtualized} \n"
            f"{no_indent})"
        )
