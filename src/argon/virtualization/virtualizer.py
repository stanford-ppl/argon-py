import ast
import types
import typing

from argon.block import Block
from argon.node.control import IfThenElse
from argon.node.function_call import FunctionCall
from argon.node.phi import Phi
from argon.ref import Exp, Node, Ref
from argon.srcctx import SrcCtx
from argon.state import State, stage
from argon.types.boolean import Boolean
from argon.types.null import Null
from argon.virtualization.type_mapper import concrete_to_abstract


def stage_phi(
    cond: Boolean,
    a: Exp[typing.Any, typing.Any],
    b: Exp[typing.Any, typing.Any],
) -> Ref[typing.Any, typing.Any]:
    if a.tp.A != b.tp.A:
        raise TypeError(f"Type mismatch: {a.tp.A} != {b.tp.A}")
    return stage(Phi[a.tp.A](cond, a, b), ctx=SrcCtx.new(2))


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

    thenBlk = Block[thenBody.tp.A](get_inputs(then_scope.scope.symbols), then_scope.scope.symbols, thenBody)
    elseBlk = Block[thenBody.tp.A](get_inputs(else_scope.scope.symbols), else_scope.scope.symbols, elseBody)
    
    return stage(IfThenElse[thenBody.tp.A](cond, thenBlk, elseBlk), ctx=SrcCtx.new(2))


def stage_if(
    cond: Boolean,
    thenBody: typing.List[Exp[typing.Any, typing.Any]],
    elseBody: typing.List[Exp[typing.Any, typing.Any]] = [],
) -> Ref[typing.Any, typing.Any]:
    thenBlk = Block[Null](get_inputs(thenBody), thenBody, Null().const(None))
    elseBlk = Block[Null](get_inputs(elseBody), elseBody, Null().const(None))
    return stage(IfThenElse[Null](cond, thenBlk, elseBlk), ctx=SrcCtx.new(2))


def get_inputs(scope_symbols: typing.List[Exp[typing.Any, typing.Any]]) -> typing.List[Exp[typing.Any, typing.Any]]:
    # We only want to consider symbols that have inputs in their rhs (i.e. Nodes)
    # We also want to exclude Mux nodes from the list of inputs
    scope_symbols = [
        symbol for symbol in scope_symbols
        if symbol.rhs != None and isinstance(symbol.rhs.val, Node) and not isinstance(symbol.rhs.val.underlying, Phi)
    ]

    # We use a symbol's id instead of just the symbol objects below because symbols
    # are not hashable and Python complains.
    # Create a dictionary mapping IDs to symbols
    symbol_map = {symbol.rhs.val.id: symbol for symbol in scope_symbols} # type: ignore -- symbol.rhs.val has already been checked to be a Node

    all_symbol_ids = set(symbol_map.keys())

    # The list of inputs in our scope constitutes the set of all inputs of all symbols 
    # minus the set of all symbols defined in this scope
    all_input_ids = set()
    for symbol in scope_symbols:
        inputs = symbol.rhs.val.underlying.inputs # type: ignore -- symbol.rhs.val has already been checked to be a Node
        inputs = [
            input for input in inputs
            if input.rhs != None and isinstance(input.rhs.val, Node)
        ]
        symbol_map.update({input.rhs.val.id: input for input in inputs}) # type: ignore -- input.rhs.val has already been checked to be a Node
        all_input_ids.update({input.rhs.val.id for input in inputs}) # type: ignore -- input.rhs.val has already been checked to be a Node
    
    result_ids = all_input_ids - all_symbol_ids
    return [symbol_map[result_id] for result_id in result_ids]


def stage_function_call(func: types.FunctionType, args: typing.List[typing.Any]) -> Ref[typing.Any, typing.Any]:
    white_list = [print] # TODO: Add more whitelisted functions here that we don't want to stage
    if func in white_list:
        return func(*args)

    abstract_args = [concrete_to_abstract(arg) for arg in args]
    abstract_func = concrete_to_abstract.function(func, abstract_args)
    return stage(FunctionCall[abstract_func.F](abstract_func, abstract_args), ctx=SrcCtx.new(2))


class Transformer(ast.NodeTransformer):
    def __init__(self, calls, ifs, if_exps):
        super().__init__()
        self.if_counter = 0
        self.unique_prefix = "__________"
        self.calls = calls
        self.ifs = ifs
        self.if_exps = if_exps
        self.concrete_to_abstract_flag = False
    
    def concrete_to_abstract(self, node):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="__________argon", ctx=ast.Load()),
                            attr="argon",
                            ctx=ast.Load()
                        ),
                        attr="virtualization",
                        ctx=ast.Load()
                    ),
                    attr="type_mapper",
                    ctx=ast.Load()
                ),
                attr="concrete_to_abstract",
                ctx=ast.Load()
            ),
            args=[node],
            keywords=[]
        )
    
    def visit_Constant(self, node):
        if self.concrete_to_abstract_flag:
            return self.concrete_to_abstract(node)
        return node
    
    def visit_Name(self, node):
        if self.concrete_to_abstract_flag:
            return self.concrete_to_abstract(node)
        return node
    
    def visit_Assign(self, node):
        return ast.Assign(targets=node.targets, value=self.visit(node.value))
    
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
                            value=ast.Name(id="__________argon", ctx=ast.Load()),
                            attr="argon",
                            ctx=ast.Load()
                        ),
                        attr="virtualization",
                        ctx=ast.Load()
                    ),
                    attr="virtualizer",
                    ctx=ast.Load()
                ),
                attr="stage_function_call",
                ctx=ast.Load()
            ),
            args=[node.func, args_list],
            keywords=[]
        )
        return staged_call

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
                            value=ast.Name(id="__________argon", ctx=ast.Load()),
                            attr="argon",
                            ctx=ast.Load()
                        ),
                        attr="virtualization",
                        ctx=ast.Load()
                    ),
                    attr="virtualizer",
                    ctx=ast.Load()
                ),
                attr="stage_if_exp_with_scopes",
                ctx=ast.Load()
            ),
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

    # This method is called for if/else statements
    def visit_If(self, node):
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
        prev_concrete_to_abstract_flag = self.concrete_to_abstract_flag
        self.concrete_to_abstract_flag = self.ifs
        self.generic_visit(node)
        self.concrete_to_abstract_flag = prev_concrete_to_abstract_flag

        # Do not stage the if statement if the flag is set to False
        if not self.ifs:
            return node

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
    {temp_var} = lambda _, {var} = {var}: __________argon.argon.virtualization.type_mapper.concrete_to_abstract({var})
    {temp_var_exists} = True
except NameError:
    {temp_var} = lambda T : __________argon.argon.state.stage(__________argon.argon.node.undefined.Undefined[T]("{var}"))
    {temp_var_exists} = False
"""
                ).body
            )

        # Run the then body under a different scope
        then_scope_name = self.generate_temp_var("then", "scope")
        new_body.extend(ast.parse(f"{then_scope_name} = __________argon.argon.state.State.get_current_state().new_scope()").body)
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
            new_body.extend(ast.parse(f"{else_scope_name} = __________argon.argon.state.State.get_current_state().new_scope()").body)
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
                    f"__________argon.argon.virtualization.virtualizer.stage_if({self.generate_temp_var("cond")}, {then_scope_name}.scope.symbols, {else_scope_name}.scope.symbols)"
                ).body
            )
        else:
            new_body.extend(
                ast.parse(
                    f"__________argon.argon.virtualization.virtualizer.stage_if({self.generate_temp_var('cond')}, {then_scope_name}.scope.symbols)"
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
                    f"{var} = __________argon.argon.virtualization.virtualizer.stage_phi({temp_var_cond}, {temp_var_then}({temp_var_T}), {temp_var_else}({temp_var_T}))"
                ).body
            )

        # Delete all temporary variables
        ast.Delete(targets=[ast.Name(id=self.generate_temp_var("cond"), ctx=ast.Del())])
        for var in assigned_vars:
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
