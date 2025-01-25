import ast
from collections import namedtuple
import dis
import typing

from argon.block import Block
from argon.node.control import Loop
from argon.ref import Exp, Ref
from argon.srcctx import SrcCtx
from argon.state import ScopeContext, stage
from argon.types.boolean import Boolean
from argon.types.null import Null
from argon.types.dictionary import Dictionary
from argon.virtualization.virtualizer.virtualizer_base import TransformerBase


def stage_loop(
    file_name: str,
    lineno: int,
    col_offset: int,
    values: typing.List[Exp[typing.Any, typing.Any]],
    binds: typing.List[Exp[typing.Any, typing.Any]],
    cond_scope_context: ScopeContext,
    cond: Exp[typing.Any, typing.Any],
    loop_scope_context: ScopeContext,
    outputs: namedtuple,
) -> Ref[typing.Any, typing.Any]:
    condBlk = Block[Boolean](
        cond_scope_context.scope.inputs, cond_scope_context.scope.symbols, cond
    )
    bodyBlk = Block[Null](
        loop_scope_context.scope.inputs,
        loop_scope_context.scope.symbols,
        Null().const(None),
    )
    output_types = {field: getattr(outputs, field).A for field in outputs._fields}
    return stage(
        Loop[Dictionary[output_types]](values, binds, condBlk, bodyBlk, outputs),
        ctx=SrcCtx(file_name, dis.Positions(lineno=lineno, col_offset=col_offset)),
    )


class TransformerLoop(TransformerBase):
    def visit_Break(self, node):
        raise NotImplementedError("Does not support break statements")

    def visit_Continue(self, node):
        raise NotImplementedError("Does not support continue statements")

    def visit_While(self, node):
        # Increment counter to ensure unique names for this while loop
        self.counter += 1

        # Properly set flags before recursive visits
        prev_concrete_to_abstract_flag = self.concrete_to_abstract_flag
        self.concrete_to_abstract_flag = self.loops

        # Recursively visit the condition
        self.variable_tracker.push_context()
        node.test = self.visit(node.test)
        cond_write_set = self.variable_tracker.current_write_set()
        cond_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.fold_context()

        # Recursively visit the body
        self.variable_tracker.push_context()
        node.body = [self.visit(stmt) for stmt in node.body]
        body_write_set = self.variable_tracker.current_write_set()
        body_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.fold_context()

        if node.orelse:
            raise NotImplementedError("Does not support else statements for loops")

        new_body: typing.List[ast.stmt] = []

        # Create temporary list of input values and binds
        values_name = self.generate_temp_var("values")
        binds_name = self.generate_temp_var("binds")
        new_body.append(
            ast.Assign(
                targets=[ast.Name(id=values_name, ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
            )
        )
        new_body.append(
            ast.Assign(
                targets=[ast.Name(id=binds_name, ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
            )
        )

        # Create binds
        # TODO: This part needs to be changed to only create binds for inputs
        for var in (
            self.variable_tracker.current_write_set()
            & self.variable_tracker.current_read_set()
        ):
            new_body.extend(
                ast.parse(
                    f"""
try:
    {values_name}.append(__________argon.argon.virtualization.type_mapper.concrete_to_abstract({var}))
    {var} = __________argon.argon.virtualization.type_mapper.concrete_to_abstract({var}).bound('{var}')
    {binds_name}.append({var})
except NameError:
    pass
"""
                ).body
            )

        # Run the condition under a different scope
        cond_scope_name = self.generate_temp_var("cond", "scope")
        cond_name = self.generate_temp_var("cond")
        new_body.extend(
            ast.parse(
                f"{cond_scope_name} = __________argon.argon.state.State.get_current_state().new_scope()"
            ).body
        )
        new_body.append(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(id=cond_scope_name, ctx=ast.Load())
                    )
                ],
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=cond_name, ctx=ast.Store())],
                        value=node.test,
                    )
                ],
            )
        )

        # Create a new scope to run the loop body
        loop_scope_name = self.generate_temp_var("loop", "scope")
        new_body.extend(
            ast.parse(
                f"{loop_scope_name} = __________argon.argon.state.State.get_current_state().new_scope()"
            ).body
        )
        new_body.append(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(id=loop_scope_name, ctx=ast.Load())
                    )
                ],
                body=node.body.copy(),
            )
        )

        loop_outputs_name = self.generate_temp_var("loop", "outputs")
        new_body.append(  # The loop body should be the original loop body + the following line to assign the loop outputs
            ast.Assign(
                targets=[ast.Name(id=loop_outputs_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Call(
                        func=ast.Name(id="namedtuple", ctx=ast.Load()),
                        args=[
                            ast.Constant(value=loop_outputs_name),
                            ast.List(
                                elts=[
                                    ast.Constant(value=var)
                                    for var in self.variable_tracker.current_write_set()
                                ],
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                    args=[
                        ast.Name(id=var, ctx=ast.Load())
                        for var in self.variable_tracker.current_write_set()
                    ],
                    keywords=[],
                ),
            )
        )

        # Stage the loop
        new_body.extend(
            ast.parse(
                f"{loop_outputs_name} = __________argon.argon.virtualization.virtualizer.virtualizer_loop.stage_loop('{self.file_name}', {node.lineno}, {node.col_offset}, {values_name}, {binds_name}, {cond_scope_name}, {cond_name}, {loop_scope_name}, {loop_outputs_name})"
            ).body
        )

        # Assign the loop outputs
        for var in self.variable_tracker.current_write_set():
            new_body.extend(ast.parse(f"{var} = {loop_outputs_name}['{var}']").body)

        # Delete all temporary variables
        # TODO: COMPLETE THIS PART
        new_body.append(ast.Delete(targets=[ast.Name(id=values_name, ctx=ast.Del())]))
        new_body.append(ast.Delete(targets=[ast.Name(id=binds_name, ctx=ast.Del())]))
        new_body.append(
            ast.Delete(targets=[ast.Name(id=cond_scope_name, ctx=ast.Del())])
        )
        new_body.append(ast.Delete(targets=[ast.Name(id=cond_name, ctx=ast.Del())]))
        new_body.append(
            ast.Delete(targets=[ast.Name(id=loop_scope_name, ctx=ast.Del())])
        )
        new_body.append(
            ast.Delete(targets=[ast.Name(id=loop_outputs_name, ctx=ast.Del())])
        )

        new_body.append(ast.Break())

        # Do not stage the while loop if the flag is set to False
        # TODO: error handling if condition is Argon type
        if self.loops:
            # Replace the original while body with the new wrapped body
            node.body = new_body

            # Set the condition to True and add a break statement at the end of the body
            # This lets us run the body just once
            node.test = ast.Constant(value=True)

        self.counter -= 1
        self.concrete_to_abstract_flag = prev_concrete_to_abstract_flag

        return node
