from typing import override
import typing
from argon.node import arith
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage
from argon.types.boolean import Boolean
from argon.virtualization.type_mapper import concrete_to_abstract, concrete_to_bound


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
        other = typing.cast(Integer, concrete_to_abstract(other))

        return stage(arith.Add[Integer](other, self), ctx=SrcCtx.new(2))
    
    def __sub__(self, other: "Integer") -> "Integer":
        other = typing.cast(Integer, concrete_to_abstract(other))

        return stage(arith.Sub[Integer](self, other), ctx=SrcCtx.new(2))
    
    def __rsub__(self, other: "Integer") -> "Integer":
        other = typing.cast(Integer, concrete_to_abstract(other))

        return stage(arith.Sub[Integer](other, self), ctx=SrcCtx.new(2))
    
    def __gt__(self, other: "Integer") -> Boolean:
        other = typing.cast(Integer, concrete_to_abstract(other))

        return stage(arith.GreaterThan[Integer](self, other), ctx=SrcCtx.new(2))
    
    def __lt__(self, other: "Integer") -> Boolean:
        other = typing.cast(Integer, concrete_to_abstract(other))

        return stage(arith.LessThan[Integer](self, other), ctx=SrcCtx.new(2))


concrete_to_abstract[int] = lambda x: Integer().const(x)
concrete_to_bound[int] = lambda name: Integer().bound(name)
