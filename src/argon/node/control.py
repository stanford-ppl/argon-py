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


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Loop(Op[Null]):
    """
    The Loop operation represents both for and while loop control flow constructs.

        cond : Boolean
            The condition to check.
        body : Block[Null]
            The loop body to execute if the condition is true.
    """

    values: typing.List[Exp[typing.Any, typing.Any]]
    binds: typing.List[Exp[typing.Any, typing.Any]]
    cond: Block[Boolean]
    body: Block[Null]
    outputs: typing.Any

    @pydantic.field_validator("outputs")
    def check_outputs(cls, v):
        if not hasattr(v, "_fields") or not hasattr(v, "_asdict"):
            raise ValueError("outputs must be a namedtuple or similar structure.")
        return v

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return self.cond.inputs + self.body.inputs

    @typing.override
    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        more_indent = "|   " * (indent_level + 2)
        values_str = ", ".join(f"{value}" for value in self.values)
        values_str = f"[{values_str}]"
        if self.binds:
            binds_str = ", \n".join(
                f"{more_indent}{bind.dump(indent_level + 2)}" for bind in self.binds
            )
            binds_str = f"[\n{binds_str}\n{indent}]"
        else:
            binds_str = "[]"
        outputs_str = ", ".join(
            f"{key} = {value}" for key, value in self.outputs._asdict().items()
        )
        outputs_str = f"({outputs_str})"
        return (
            f"Loop( \n"
            f"{indent}values = {values_str}, \n"
            f"{indent}binds = {binds_str}, \n"
            f"{indent}cond = {self.cond.dump(indent_level + 1)}, \n"
            f"{indent}body = {self.body.dump(indent_level + 1)}, \n"
            # print the field value pairs of outputs
            f"{indent}outputs = {outputs_str} \n"
            f"{no_indent})"
        )
