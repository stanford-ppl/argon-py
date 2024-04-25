import ast
import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.block import Block
from argon.op import Op
from argon.ref import Exp, Sym
from argon.srcctx import SrcCtx
from argon.state import stage
from argon.types.boolean import Boolean

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class IfThenElse[T](Op[T]):
    cond: Boolean
    thenBlk: Block[T]
    elseBlk: Block[T]

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.cond] # type: ignore

def stage_if_exp(cond: Boolean, thenBody: Exp[typing.Any, typing.Any], elseBody: Exp[typing.Any, typing.Any]):
    if thenBody.tp.A != elseBody.tp.A:
        raise TypeError(f"Type mismatch: {thenBody.tp.A} != {elseBody.tp.A}")
    thenBlk = Block[thenBody.tp.A]([], [thenBody], None)
    elseBlk = Block[thenBody.tp.A]([], [elseBody], None)
    return stage(IfThenElse[thenBody.tp.A](cond, thenBlk, elseBlk), ctx=SrcCtx.new(2))

class TransformIfExpressions(ast.NodeTransformer):
    def visit_IfExp(self, node):
        # This method is called for ternary "if" expressions

        # Recursively visit the condition, the then case, and the else case
        self.generic_visit(node)

        func_call = ast.Call(
            func=ast.Name(id='stage_if_exp', ctx=ast.Load()),
            args=[
                node.test,   # the condition in the if expression
                node.body,   # the expression for the 'true' case
                node.orelse  # the expression for the 'false' case
            ],
            keywords=[]
        )
        # Replace the IfExp node with the new function call and return it
        return ast.copy_location(func_call, node)
