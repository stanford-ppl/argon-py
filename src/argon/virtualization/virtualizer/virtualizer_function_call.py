import ast
import types
import typing

from argon.node.function_call import FunctionCall
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage
from argon.virtualization.virtualizer.virtualizer_base import TransformerBase
from argon.virtualization.type_mapper import concrete_to_abstract


def stage_function_call(
    func: types.FunctionType, args: typing.List[typing.Any]
) -> Ref[typing.Any, typing.Any]:
    white_list = [
        print
    ]  # TODO: Add more whitelisted functions here that we don't want to stage
    if func in white_list:
        return func(*args)

    abstract_args = [concrete_to_abstract(arg) for arg in args]
    abstract_func = concrete_to_abstract.function(func, abstract_args)
    return stage(
        FunctionCall[abstract_func.F](abstract_func, abstract_args), ctx=SrcCtx.new(2)
    )


class TransformerFunctionCall(TransformerBase):
    # This method is called for function calls
    def visit_Call(self, node):
        # Recursively visit arguments
        prev_concrete_to_abstract_flag = self.concrete_to_abstract_flag
        self.concrete_to_abstract_flag = False
        self.generic_visit(node)
        self.concrete_to_abstract_flag = prev_concrete_to_abstract_flag

        # Do not stage the function call if the flag is set to False
        if not self.calls:
            if not self.concrete_to_abstract_flag:
                return node
            else:
                return self.concrete_to_abstract(node)

        # Wrap arguments in a list
        args_list = ast.List(elts=node.args, ctx=ast.Load())

        # Create the staged function call
        staged_call = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id="__________argon", ctx=ast.Load()),
                                attr="argon",
                                ctx=ast.Load(),
                            ),
                            attr="virtualization",
                            ctx=ast.Load(),
                        ),
                        attr="virtualizer",
                        ctx=ast.Load(),
                    ),
                    attr="virtualizer_function_call",
                    ctx=ast.Load(),
                ),
                attr="stage_function_call",
                ctx=ast.Load(),
            ),
            args=[node.func, args_list],
            keywords=[],
        )

        # Attach source location (line numbers and column offsets)
        ast.copy_location(staged_call, node)

        return staged_call
