from typing import override
import typing
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage

import argon.node.logical as logical
from argon.virtualization.type_mapper import concrete_to_abstract


class Boolean(Ref[bool, "Boolean"]):
    """
    The Boolean class represents a boolean value in the Argon language.
    """

    @override
    def fresh(self) -> "Boolean":
        return Boolean()

    def __invert__(self) -> "Boolean":
        return stage(logical.Not[Boolean](self), ctx=SrcCtx.new(2))

    def __and__(self, other: "Boolean") -> "Boolean":
        other = typing.cast(Boolean, concrete_to_abstract(other))
        return stage(logical.And[Boolean](self, other), ctx=SrcCtx.new(2))

    def __rand__(self, other: "Boolean") -> "Boolean":
        return self & other

    def __or__(self, other: "Boolean") -> "Boolean":
        other = typing.cast(Boolean, concrete_to_abstract(other))
        return stage(logical.Or[Boolean](self, other), ctx=SrcCtx.new(2))

    def __ror__(self, other: "Boolean") -> "Boolean":
        return self | other

    def __xor__(self, other: "Boolean") -> "Boolean":
        other = typing.cast(Boolean, concrete_to_abstract(other))
        return stage(logical.Xor[Boolean](self, other), ctx=SrcCtx.new(2))

    def __rxor__(self, other: "Boolean") -> "Boolean":
        return self ^ other


concrete_to_abstract[bool] = lambda x: Boolean().const(x)
