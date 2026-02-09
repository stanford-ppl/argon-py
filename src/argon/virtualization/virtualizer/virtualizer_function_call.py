import ast
import copy
import types
import typing

from argon.node.function_call import FunctionCall
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage
from argon.virtualization.virtualizer.virtualizer_base import TransformerBase
from argon.virtualization.type_mapper import concrete_to_abstract
from argon.types.function import Function


def white_list(func: typing.Any) -> bool:
    """Check if a function is whitelisted for direct execution (not staged)."""
    whitelisted_functions = [
        print,
    ]  # TODO: Add more whitelisted functions here that we don't want to stage
    return func in whitelisted_functions


def stage_function_call(
    func: typing.Any, args: typing.List[typing.Any]
) -> Ref[typing.Any, typing.Any]:
    abstract_func = concrete_to_abstract(func)
    abstract_args = [concrete_to_abstract(arg) for arg in args]

    func_type_origin = typing.get_origin(abstract_func.A) or abstract_func.A
    if func_type_origin is Function:
        # TODO: check if arg types match the function signature
        return stage(
            FunctionCall[abstract_func.RETURN_TP](abstract_func, abstract_args),
            ctx=SrcCtx.new(2),
        )
    else:
        if not hasattr(func, "__call__"):
            raise TypeError(f"Object of type {type(func)} is not callable")
        return abstract_func(*abstract_args)


class TransformerFunctionCall(TransformerBase):
    # This method is called for function calls
    def visit_Call(self, node):
        original_node_func = copy.deepcopy(node.func)
        original_node_args = copy.deepcopy(node.args)
        original_node_keywords = copy.deepcopy(node.keywords)

        # Recursively visit arguments
        prev_concrete_to_abstract_flag = self.concrete_to_abstract_flag
        self.concrete_to_abstract_flag = self.calls
        self.generic_visit(node)
        self.concrete_to_abstract_flag = prev_concrete_to_abstract_flag

        # Do not stage the function call if the flag is set to False
        if not self.calls:
            if not self.concrete_to_abstract_flag:
                return node
            else:
                return self.concrete_to_abstract(node)

        # Create the normal function call (for whitelisted functions)
        normal_call = ast.Call(
            func=original_node_func,
            args=original_node_args,
            keywords=original_node_keywords,
        )

        # Create a whitelist check call
        whitelist_check = ast.Call(
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
                attr="white_list",
                ctx=ast.Load(),
            ),
            args=[original_node_func],
            keywords=[],
        )

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

        # Create an if expression: whitelist_check(func) ? normal_call : staged_call
        final_call = ast.IfExp(
            test=whitelist_check,
            body=normal_call,
            orelse=staged_call,
        )

        # Attach source location (line numbers and column offsets)
        ast.copy_location(final_call, node)

        # TODO: Temporary fix, assuming that potential whitelisted functions are always simple names (e.g. print)
        if isinstance(original_node_func, ast.Name):
            return final_call
        else:
            ast.copy_location(staged_call, node)
            return staged_call
