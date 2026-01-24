import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.op import Op
from argon.ref import Exp
from argon.types.torch import NNModule


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class NNModuleCall[T](Op[T]):
    """
    The NNModuleCall[T] operation represents a call to a nn.Module.

        module : NNModule
            The nn.Module to call.
        args : List[Exp[Any]]
            The arguments to pass to the nn.Module.
    """

    module: NNModule
    args: typing.List[Exp[typing.Any, typing.Any]]

    @property
    @typing.override
    def operands(self) -> typing.List[Exp[typing.Any, typing.Any]]:
        return [self.module, *self.args]

    @typing.override
    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        return (
            f"NNModuleCall( \n"
            f"{indent}module = {self.module}, \n"
            f"{indent}args = [{', '.join([str(arg) for arg in self.args])}] \n"
            f"{no_indent})"
        )
