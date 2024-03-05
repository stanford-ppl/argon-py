from typing import override
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage


class Integer(Ref[int, "Integer"]):
    @override
    @classmethod
    def fresh(cls) -> "Integer":
        return Integer()

    def __add__(self, other: "Integer") -> "Integer":
        import argon.node.arith as arith

        return stage(arith.IntegerAdd(self, other, ctx=SrcCtx.new(2)))
