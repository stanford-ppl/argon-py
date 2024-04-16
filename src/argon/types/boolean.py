from typing import override
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage

import argon.node.logical as logical 

class Boolean(Ref[bool, "Boolean"]):
    @override
    def fresh(self) -> "Boolean":
        return Boolean()
    
    def __invert__(self) -> "Boolean":
        return stage(logical.Not[Boolean](self), ctx=SrcCtx.new(2))
    
    def __and__(self, other: "Boolean") -> "Boolean":
        return stage(logical.And[Boolean](self, other), ctx=SrcCtx.new(2))

    def __or__(self, other: "Boolean") -> "Boolean":
        return stage(logical.Or[Boolean](self, other), ctx=SrcCtx.new(2))
    
    def __xor__(self, other: "Boolean") -> "Boolean":
        return stage(logical.Xor[Boolean](self, other), ctx=SrcCtx.new(2))
        