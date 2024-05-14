import ast
import typing

from argon.block import Block
from argon.node.control import IfThenElse
from argon.node.mux import Mux
from argon.ref import Exp, Ref
from argon.srcctx import SrcCtx
from argon.state import State, stage
from argon.types.boolean import Boolean
from argon.types.null import Null


def stage_mux(
    cond: Boolean,
    a: Exp[typing.Any, typing.Any],
    b: Exp[typing.Any, typing.Any],
) -> Ref[typing.Any, typing.Any]:
    if a.tp.A != b.tp.A:
        raise TypeError(f"Type mismatch: {a.tp.A} != {b.tp.A}")
    return stage(Mux[a.tp.A](cond, a, b), ctx=SrcCtx.new(2))


def stage_if_exp_with_scopes(
    cond: Boolean,
    thenBodyLambda: typing.Callable[[], Exp[typing.Any, typing.Any]],
    elseBodyLambda: typing.Callable[[], Exp[typing.Any, typing.Any]],
) -> Ref[typing.Any, typing.Any]:
    # Execute the bodies in separate scopes
    state = State.get_current_state()
    then_scope = state.new_scope()
    with then_scope:
        thenBody: Exp[typing.Any, typing.Any] = thenBodyLambda()
    else_scope = state.new_scope()
    with else_scope:
        elseBody: Exp[typing.Any, typing.Any] = elseBodyLambda()
    
    # Type check
    if thenBody.tp.A != elseBody.tp.A:
        raise TypeError(f"Type mismatch: {thenBody.tp.A} != {elseBody.tp.A}")
    
    thenBlk = Block[thenBody.tp.A]([], then_scope.scope.symbols, thenBody)
    elseBlk = Block[thenBody.tp.A]([], else_scope.scope.symbols, elseBody)
    # then_scope.scope.symbols[0].rhs.val.underlying.inputs
    
    return stage(IfThenElse[thenBody.tp.A](cond, thenBlk, elseBlk), ctx=SrcCtx.new(2))


def stage_if_exp(
    cond: Boolean,
    thenBody: Exp[typing.Any, typing.Any],
    elseBody: Exp[typing.Any, typing.Any],
) -> Ref[typing.Any, typing.Any]:
    # Type check
    if thenBody.tp.A != elseBody.tp.A:
        raise TypeError(f"Type mismatch: {thenBody.tp.A} != {elseBody.tp.A}")

    # Create blocks and stage
    thenBlk = Block[thenBody.tp.A]([], [thenBody], thenBody)
    elseBlk = Block[thenBody.tp.A]([], [elseBody], elseBody)
    return stage(IfThenElse[thenBody.tp.A](cond, thenBlk, elseBlk), ctx=SrcCtx.new(2))


def stage_if(
    cond: Boolean,
    thenBody: typing.List[Exp[typing.Any, typing.Any]],
    elseBody: typing.List[Exp[typing.Any, typing.Any]] = [],
) -> Ref[typing.Any, typing.Any]:
    thenBlk = Block[Null]([], thenBody, Null().const(None))
    elseBlk = Block[Null]([], elseBody, Null().const(None))
    return stage(IfThenElse[Null](cond, thenBlk, elseBlk), ctx=SrcCtx.new(2))


class Transformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.if_counter = 0
        self.unique_prefix = "__________"

    def visit_IfExp(self, node):
        # This method is called for ternary "if" expressions

        # Recursively visit the condition, the then case, and the else case
        self.generic_visit(node)

        # Stage if expression call
        func_call = ast.Call(
            func=ast.Name(id="stage_if_exp_with_scopes", ctx=ast.Load()),
            args=[
                node.test,  # the condition in the if expression
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
        return func_call

    def visit_If(self, node):
        # This method is called for if/else statements

        # Increment counter to ensure unique names for this if statement
        self.if_counter += 1

        # Find all assigned variables in the current 'if' statement
        then_vars = set()
        for stm in node.body:
            then_vars = then_vars | self.get_assigned_vars(stm)
        else_vars = set()
        for stm in node.orelse:
            else_vars = else_vars | self.get_assigned_vars(stm)
        assigned_vars = then_vars | else_vars

        # Recursively visit the condition, the then case, and the else case
        self.generic_visit(node)

        # Save the condition in a temporary variable
        new_body: typing.List[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=self.generate_temp_var("cond"), ctx=ast.Store())],
                value=node.test,
            )
        ]

        # Save the previous value of variables if had existed, otherwise set them to Undefined
        for var in assigned_vars:
            temp_var = self.generate_temp_var(var, "old")
            temp_var_exists = self.generate_temp_var(var, "old", "exists")
            new_body.extend(
                ast.parse(
                    f"""
try:
    {temp_var} = lambda _, {var} = {var}: {var}
    {temp_var_exists} = True
except NameError:
    {temp_var} = lambda T : stage(Undefined[T]("{var}"))
    {temp_var_exists} = False
"""
                ).body
            )

        # Run the then body under a different scope
        then_scope_name = self.generate_temp_var("then", "scope")
        new_body.extend(ast.parse(f"{then_scope_name} = state.new_scope()").body)
        new_body.append(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(id=then_scope_name, ctx=ast.Load())
                    )
                ],
                body=self.modify_body(
                    node.body, "then", then_vars
                ),  # For all variables in the 'then' body, assign temp_var = var at the end of the body
            )
        )

        # Make result from 'then' body a lambda and revert variables back to their original value before the 'then' body
        for var in assigned_vars:
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
        if node.orelse:
            new_body.extend(ast.parse(f"{else_scope_name} = state.new_scope()").body)
            new_body.append(
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Name(id=else_scope_name, ctx=ast.Load())
                        )
                    ],
                    body=self.modify_body(node.orelse, "else", else_vars),
                )
            )

        # Make result from 'else' body a lambda
        for var in assigned_vars:
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

        # Stage if call
        if node.orelse:
            new_body.extend(
                ast.parse(
                    f"stage_if({self.generate_temp_var("cond")}, {then_scope_name}.scope.symbols, {else_scope_name}.scope.symbols)"
                ).body
            )
        else:
            new_body.extend(
                ast.parse(
                    f"stage_if({self.generate_temp_var('cond')}, {then_scope_name}.scope.symbols)"
                ).body
            )

        # Stage each assigned variable be a mux of the possible values from each branch
        for var in assigned_vars:
            temp_var_cond = self.generate_temp_var("cond")
            temp_var_then = self.generate_temp_var(var, "then")
            temp_var_else = self.generate_temp_var(var, "else")
            temp_var_T = self.generate_temp_var(var, "T")
            new_body.extend(
                ast.parse(
                    f"{var} = stage_mux({temp_var_cond}, {temp_var_then}({temp_var_T}), {temp_var_else}({temp_var_T}))"
                ).body
            )

        # Delete all temporary variables
        for var in assigned_vars:
            temp_var_cond = self.generate_temp_var("cond")
            temp_var_old = self.generate_temp_var(var, "old")
            temp_var_old_exists = self.generate_temp_var(var, "old", "exists")
            temp_var_then = self.generate_temp_var(var, "then")
            temp_var_else = self.generate_temp_var(var, "else")
            temp_var_T = self.generate_temp_var(var, "T")
            new_body.extend(
                [
                    ast.Delete(targets=[ast.Name(id=temp_var_cond, ctx=ast.Del())]),
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
            ast.Delete(targets=[ast.Name(id=then_scope_name, ctx=ast.Del())])
        )
        if node.orelse:
            new_body.append(
                ast.Delete(targets=[ast.Name(id=else_scope_name, ctx=ast.Del())])
            )

        # Replace the original if/else body with the new wrapped body
        node.body = new_body
        node.orelse = []

        # Set the condition to True to run both the then and else bodies under different scopes
        node.test = ast.Constant(value=True)

        self.if_counter -= 1

        return node

    def generate_temp_var(self, *args) -> str:
        return self.unique_prefix + "_".join(args) + "_" + str(self.if_counter)

    def get_assigned_vars(self, stm: ast.stmt) -> set:
        assigned_vars = set()
        for n in ast.walk(stm):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
        return assigned_vars

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
