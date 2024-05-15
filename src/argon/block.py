import typing
import pydantic
from pydantic import dataclasses

from argon.base import ArgonMeta
from argon.ref import Exp


@dataclasses.dataclass
class Block[B](ArgonMeta):
    """
    The Block[B] operation represents a block of code that contains a list of
    inputs, statements, and a result.

        inputs : List[Exp[Any, Any]]
            The list of input expressions.
        stms : List[Exp[Any, Any]]
            The list of statement expressions.
        result : Optional[Exp[Any, B]]
            The result expression of the block.
    """

    inputs: typing.List[Exp[typing.Any, typing.Any]] = pydantic.Field(
        default_factory=list
    )
    stms: typing.List[Exp[typing.Any, typing.Any]] = pydantic.Field(
        default_factory=list
    )
    result: typing.Optional[Exp[typing.Any, B]] = None

    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        more_indent = "|   " * (indent_level + 2)
        if self.inputs:
            inputs_str = ", ".join(str(input) for input in self.inputs)
            inputs_str = f"[{inputs_str}]"
        else:
            inputs_str = "[]"
        if self.stms:
            stms_str = ", \n".join(
                f"{more_indent}{stm.dump(indent_level + 2)}" for stm in self.stms
            )
            stms_str = f"[\n{stms_str}\n{indent}]"
        else:
            stms_str = "[]"
        result_str = "None" if self.result is None else str(self.result)
        return (
            f"Block( \n"
            f"{indent}inputs = {inputs_str}, \n"
            f"{indent}stms = {stms_str}, \n"
            f"{indent}result = {result_str} \n"
            f"{no_indent})"
        )
