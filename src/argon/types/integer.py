from typing import override
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage


class Integer(Ref[int, "Integer"]):
    """
    The Integer class represents an integer value in the Argon language.
    """

    @override
    def fresh(self) -> "Integer":
        return Integer()

    def __add__(self, other: "Integer") -> "Integer":
        import argon.node.arith as arith

        return stage(arith.Add[Integer](self, other), ctx=SrcCtx.new(2))
