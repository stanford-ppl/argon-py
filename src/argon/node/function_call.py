import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.op import Op
from argon.ref import Exp, Ref
from argon.types.function import Function


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class FunctionCall[T](Op[T]):
    """
    The FunctionCall[T] operation represents a function call with return type T.

        func : Function[T]
            The function to call.
        args : List[Ref[Any]]
            The arguments to pass to the function.
    """

    func: Function[T]
    args: typing.List[Ref[typing.Any, typing.Any]]

    @property
    @typing.override
    def inputs(self) -> typing.List[Exp[typing.Any, typing.Any]]:
        # TODO: figure out what inputs I should use
        pass

    @typing.override
    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        return (
            f"FunctionCall( \n"
            f"{indent}func = {self.func}, \n"
            f"{indent}args = [{', '.join([str(arg) for arg in self.args])}] \n"
            f"{no_indent})"
        )
