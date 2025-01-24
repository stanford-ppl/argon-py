import ast
import dis
import typing

from argon.block import Block
from argon.node.control import IfThenElse
from argon.node.phi import Phi
from argon.node.undefined import Undefined
from argon.ref import Exp, Ref
from argon.srcctx import SrcCtx
from argon.state import ScopeContext, State, stage
from argon.types.boolean import Boolean
from argon.types.null import Null
from argon.virtualization.virtualizer.virtualizer_base import TransformerBase


def stage_undefined(name, T, file_name, lineno, col_offset):
    return stage(
        Undefined[T](name),
        ctx=SrcCtx(file_name, dis.Positions(lineno=lineno, col_offset=col_offset)),
    )


def stage_phi(
    cond: Boolean,
    a: Exp[typing.Any, typing.Any],
    b: Exp[typing.Any, typing.Any],
) -> Ref[typing.Any, typing.Any]:
    if a.tp.A != b.tp.A:
        raise TypeError(f"Type mismatch: {a.tp.A} != {b.tp.A}")
    return stage(Phi[a.tp.A](cond, a, b), ctx=SrcCtx.new(2))


def stage_if_exp_with_scopes(
    condLambda: typing.Callable[[], Exp[typing.Any, typing.Any]],
    thenBodyLambda: typing.Callable[[], Exp[typing.Any, typing.Any]],
    elseBodyLambda: typing.Callable[[], Exp[typing.Any, typing.Any]],
) -> Ref[typing.Any, typing.Any]:
    # Execute the bodies in separate scopes
    state = State.get_current_state()
    cond_scope_context = state.new_scope()
    with cond_scope_context:
        cond: Exp[typing.Any, typing.Any] = condLambda()
    then_scope_context = state.new_scope()
    with then_scope_context:
        thenBody: Exp[typing.Any, typing.Any] = thenBodyLambda()
    else_scope_context = state.new_scope()
    with else_scope_context:
        elseBody: Exp[typing.Any, typing.Any] = elseBodyLambda()

    # Type check
    if thenBody.tp.A != elseBody.tp.A:
        raise TypeError(f"Type mismatch: {thenBody.tp.A} != {elseBody.tp.A}")

    condBlk = Block[Boolean](
        cond_scope_context.scope.inputs, cond_scope_context.scope.symbols, cond
    )
    thenBlk = Block[thenBody.tp.A](
        then_scope_context.scope.inputs, then_scope_context.scope.symbols, thenBody
    )
    elseBlk = Block[thenBody.tp.A](
        else_scope_context.scope.inputs, else_scope_context.scope.symbols, elseBody
    )

    return stage(
        IfThenElse[thenBody.tp.A](condBlk, thenBlk, elseBlk), ctx=SrcCtx.new(2)
    )


def stage_if(
    file_name: str,
    lineno: int,
    col_offset: int,
    cond_scope_context: ScopeContext,
    cond: Exp[typing.Any, typing.Any],
    then_scope_context: ScopeContext,
    else_scope_context: ScopeContext,
) -> Ref[typing.Any, typing.Any]:
    condBlk = Block[Boolean](
        cond_scope_context.scope.inputs, cond_scope_context.scope.symbols, cond
    )
    thenBlk = Block[Null](
        then_scope_context.scope.inputs,
        then_scope_context.scope.symbols,
        Null().const(None),
    )
    elseBlk = Block[Null](
        else_scope_context.scope.inputs,
        else_scope_context.scope.symbols,
        Null().const(None),
    )
    return stage(
        IfThenElse[Null](condBlk, thenBlk, elseBlk),
        ctx=SrcCtx(file_name, dis.Positions(lineno=lineno, col_offset=col_offset)),
    )


class TransformerIfThenElse(TransformerBase):
    # This method is called for ternary "if" expressions
    def visit_IfExp(self, node):
        # Recursively visit the condition, the then case, and the else case
        prev_concrete_to_abstract_flag = self.concrete_to_abstract_flag
        self.concrete_to_abstract_flag = self.if_exps
        self.generic_visit(node)
        self.concrete_to_abstract_flag = prev_concrete_to_abstract_flag

        # Do not stage the if expression if the flag is set to False
        if not self.if_exps:
            if not self.concrete_to_abstract_flag:
                return node
            else:
                return self.concrete_to_abstract(node)

        # Stage if expression call
        func_call = ast.Call(
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
                    attr="virtualizer_ifthenelse",
                    ctx=ast.Load(),
                ),
                attr="stage_if_exp_with_scopes",
                ctx=ast.Load(),
            ),
            args=[
                ast.Lambda(  # the condition in the if expression wrapped in a lambda
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=node.test,
                ),
                ast.Lambda(  # the expression for the 'true' case wrapped in a lambda
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=node.body,
                ),
                ast.Lambda(  # the expression for the 'false' case wrapped in a lambda
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=node.orelse,
                ),
            ],
            keywords=[],
        )

        # Attach source location (line numbers and column offsets)
        ast.copy_location(func_call, node)

        return func_call

    # This method is called for if/else statements
    def visit_If(self, node):
        # Increment counter to ensure unique names for this if statement
        self.counter += 1

        # Properly set flags before recursive visits
        prev_concrete_to_abstract_flag = self.concrete_to_abstract_flag
        self.concrete_to_abstract_flag = self.ifs
        self.variable_tracker.push_context()

        # Recursively visit the condition
        self.variable_tracker.push_context()
        node.test = self.visit(node.test)
        cond_write_set = self.variable_tracker.current_write_set()
        cond_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.fold_context()

        # Recursively visit the then body
        self.variable_tracker.push_context()
        node.body = [self.visit(stmt) for stmt in node.body]
        then_write_set = self.variable_tracker.current_write_set()
        then_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.fold_context()

        # Recursively visit the else body
        self.variable_tracker.push_context()
        node.orelse = [self.visit(stmt) for stmt in node.orelse]
        else_write_set = self.variable_tracker.current_write_set()
        else_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.fold_context()

        new_body: typing.List[ast.stmt] = []

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

        # Save the previous value of variables if had existed, otherwise set them to Undefined
        for var in then_write_set | else_write_set:
            temp_var = self.generate_temp_var(var, "old")
            temp_var_exists = self.generate_temp_var(var, "old", "exists")
            new_body.extend(
                ast.parse(
                    f"""
try:
    {temp_var} = lambda _, {var} = {var}: __________argon.argon.virtualization.type_mapper.concrete_to_abstract({var})
    {temp_var_exists} = True
except NameError:
    {temp_var} = lambda T : __________argon.argon.virtualization.virtualizer.virtualizer_ifthenelse.stage_undefined('{var}', T, '{self.file_name}', {node.lineno}, {node.col_offset})
    {temp_var_exists} = False
"""
                ).body
            )

        # Run the then body under a different scope
        then_scope_name = self.generate_temp_var("then", "scope")
        new_body.extend(
            ast.parse(
                f"{then_scope_name} = __________argon.argon.state.State.get_current_state().new_scope()"
            ).body
        )
        new_body.append(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(id=then_scope_name, ctx=ast.Load())
                    )
                ],
                body=self.modify_body(
                    node.body, "then", then_write_set
                ),  # For all variables in the 'then' body, assign temp_var = var at the end of the body
            )
        )

        # Make result from 'then' body a lambda and revert variables back to their original value before the 'then' body
        for var in then_write_set | else_write_set:
            temp_var_T = self.generate_temp_var(var, "T")
            temp_var_then = self.generate_temp_var(var, "then")
            temp_var_old = self.generate_temp_var(var, "old")
            temp_var_old_exists = self.generate_temp_var(var, "old", "exists")
            new_body.extend(
                ast.parse(
                    f"""
try:
    {temp_var_T} = {temp_var_then}.A
    {temp_var_then} = lambda _, {temp_var_then} = {temp_var_then}: {temp_var_then}
    if {temp_var_old_exists}:
        {var} = {temp_var_old}({temp_var_T})
    else:
        del {var}
except NameError:
    {temp_var_then} = {temp_var_old}
"""
                ).body
            )

        # Run the else body under a different scope
        else_scope_name = self.generate_temp_var("else", "scope")
        new_body.extend(
            ast.parse(
                f"{else_scope_name} = __________argon.argon.state.State.get_current_state().new_scope()"
            ).body
        )
        if node.orelse:
            new_body.append(
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Name(id=else_scope_name, ctx=ast.Load())
                        )
                    ],
                    body=self.modify_body(node.orelse, "else", else_write_set),
                )
            )

        # Make result from 'else' body a lambda
        for var in then_write_set | else_write_set:
            temp_var_T = self.generate_temp_var(var, "T")
            temp_var_else = self.generate_temp_var(var, "else")
            temp_var_old = self.generate_temp_var(var, "old")
            new_body.extend(
                ast.parse(
                    f"""
try:
    {temp_var_T} = {temp_var_else}.A
    {temp_var_else} = lambda _, {temp_var_else} = {temp_var_else}: {temp_var_else}
except NameError:
    {temp_var_else} = {temp_var_old}
"""
                ).body
            )

        # Stage the undefined variables in their respective scopes
        if self.variable_tracker.current_write_set():
            new_body.append(
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Name(id=then_scope_name, ctx=ast.Load())
                        )
                    ],
                    body=[
                        # Call {temp_var_then}({temp_var_T}) for each var in self.assigned_vars
                        ast.Assign(
                            targets=[
                                ast.Name(
                                    id=self.generate_temp_var(var, "then"),
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Call(
                                func=ast.Name(
                                    id=f"{self.generate_temp_var(var, 'then')}",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Name(
                                        id=self.generate_temp_var(var, "T"),
                                        ctx=ast.Load(),
                                    )
                                ],
                                keywords=[],
                            ),
                        )
                        for var in then_write_set | else_write_set
                    ],
                )
            )
            new_body.append(
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Name(id=else_scope_name, ctx=ast.Load())
                        )
                    ],
                    body=[
                        # Call {temp_var_else}({temp_var_T}) for each var in self.assigned_vars
                        ast.Assign(
                            targets=[
                                ast.Name(
                                    id=self.generate_temp_var(var, "else"),
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Call(
                                func=ast.Name(
                                    id=f"{self.generate_temp_var(var, 'else')}",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Name(
                                        id=self.generate_temp_var(var, "T"),
                                        ctx=ast.Load(),
                                    )
                                ],
                                keywords=[],
                            ),
                        )
                        for var in then_write_set | else_write_set
                    ],
                )
            )

        # Stage if call
        new_body.extend(
            ast.parse(
                f"__________argon.argon.virtualization.virtualizer.virtualizer_ifthenelse.stage_if('{self.file_name}', {node.lineno}, {node.col_offset}, {cond_scope_name}, {cond_name}, {then_scope_name}, {else_scope_name})"
            ).body
        )

        # Stage each assigned variable be a mux of the possible values from each branch
        for var in then_write_set | else_write_set:
            temp_var_then = self.generate_temp_var(var, "then")
            temp_var_else = self.generate_temp_var(var, "else")
            new_body.extend(
                ast.parse(
                    f"{var} = __________argon.argon.virtualization.virtualizer.virtualizer_ifthenelse.stage_phi({cond_name}, {temp_var_then}, {temp_var_else})"
                ).body
            )

        # Delete all temporary variables
        new_body.append(ast.Delete(targets=[ast.Name(id=cond_name, ctx=ast.Del())]))
        for var in then_write_set | else_write_set:
            temp_var_old = self.generate_temp_var(var, "old")
            temp_var_old_exists = self.generate_temp_var(var, "old", "exists")
            temp_var_then = self.generate_temp_var(var, "then")
            temp_var_else = self.generate_temp_var(var, "else")
            temp_var_T = self.generate_temp_var(var, "T")
            new_body.extend(
                [
                    ast.Delete(targets=[ast.Name(id=temp_var_old, ctx=ast.Del())]),
                    ast.Delete(
                        targets=[ast.Name(id=temp_var_old_exists, ctx=ast.Del())]
                    ),
                    ast.Delete(targets=[ast.Name(id=temp_var_then, ctx=ast.Del())]),
                    ast.Delete(targets=[ast.Name(id=temp_var_else, ctx=ast.Del())]),
                    ast.Delete(targets=[ast.Name(id=temp_var_T, ctx=ast.Del())]),
                ]
            )
        new_body.append(
            ast.Delete(targets=[ast.Name(id=cond_scope_name, ctx=ast.Del())])
        )
        new_body.append(
            ast.Delete(targets=[ast.Name(id=then_scope_name, ctx=ast.Del())])
        )
        if node.orelse:
            new_body.append(
                ast.Delete(targets=[ast.Name(id=else_scope_name, ctx=ast.Del())])
            )

        # Do not stage the if statement if the flag is set to False
        # TODO: error handling if condition is Argon type
        if self.ifs:
            # Replace the original if/else body with the new wrapped body
            node.body = new_body
            node.orelse = []

            # Set the condition to True to run both the then and else bodies under different scopes
            node.test = ast.Constant(value=True)

        self.counter -= 1
        self.concrete_to_abstract_flag = prev_concrete_to_abstract_flag
        self.variable_tracker.fold_context()

        return node
    
    def modify_body(
        self, body: typing.List[ast.stmt], scope_name: str, vars: set
    ) -> list:
        new_body = body.copy()
        for var in vars:
            temp_var = self.generate_temp_var(var, scope_name)
            temp_assign = ast.Assign(
                targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                value=ast.Name(id=var, ctx=ast.Load()),
            )
            new_body.append(temp_assign)
        return new_body