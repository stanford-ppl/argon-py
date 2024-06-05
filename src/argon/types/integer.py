from typing import override
import typing
from argon.node import arith
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage
from argon.virtualization.type_mapper import concrete_to_abstract


class Integer(Ref[int, "Integer"]):
    """
    The Integer class represents an integer value in the Argon language.
    """

    @override
    def fresh(self) -> "Integer":
        return Integer()

    def __add__(self, other: "Integer") -> "Integer":
        other = typing.cast(Integer, concrete_to_abstract(other))

        return stage(arith.Add[Integer](self, other), ctx=SrcCtx.new(2))

    def __radd__(self, other: "Integer") -> "Integer":
        return self + other


concrete_to_abstract[int] = lambda x: Integer().const(x)
