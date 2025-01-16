import ast
from collections import namedtuple
import dis
import types
import typing

from argon.block import Block
from argon.node.control import IfThenElse, Loop
from argon.node.function_call import FunctionCall
from argon.node.phi import Phi
from argon.node.undefined import Undefined
from argon.ref import Exp, Ref
from argon.srcctx import SrcCtx
from argon.state import State, stage
from argon.types.boolean import Boolean
from argon.types.null import Null
from argon.virtualization.type_mapper import concrete_to_abstract


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
    cond_scope = state.new_scope()
    with cond_scope:
        cond: Exp[typing.Any, typing.Any] = condLambda()
    then_scope = state.new_scope()
    with then_scope:
        thenBody: Exp[typing.Any, typing.Any] = thenBodyLambda()
    else_scope = state.new_scope()
    with else_scope:
        elseBody: Exp[typing.Any, typing.Any] = elseBodyLambda()

    # Type check
    if thenBody.tp.A != elseBody.tp.A:
        raise TypeError(f"Type mismatch: {thenBody.tp.A} != {elseBody.tp.A}")

    condBlk = Block[Boolean](
        get_inputs(cond_scope.scope.symbols), cond_scope.scope.symbols, cond
    )
    thenBlk = Block[thenBody.tp.A](
        get_inputs(then_scope.scope.symbols), then_scope.scope.symbols, thenBody
    )
    elseBlk = Block[thenBody.tp.A](
        get_inputs(else_scope.scope.symbols), else_scope.scope.symbols, elseBody
    )

    return stage(
        IfThenElse[thenBody.tp.A](condBlk, thenBlk, elseBlk), ctx=SrcCtx.new(2)
    )


def stage_if(
    file_name: str,
    lineno: int,
    col_offset: int,
    cond: typing.List[Exp[typing.Any, typing.Any]],
    thenBody: typing.List[Exp[typing.Any, typing.Any]],
    elseBody: typing.List[Exp[typing.Any, typing.Any]] = [],
) -> Ref[typing.Any, typing.Any]:
    condBlk = Block[Boolean](get_inputs(cond), cond, cond[-1])
    thenBlk = Block[Null](get_inputs(thenBody), thenBody, Null().const(None))
    elseBlk = Block[Null](get_inputs(elseBody), elseBody, Null().const(None))
    return stage(
        IfThenElse[Null](condBlk, thenBlk, elseBlk),
        ctx=SrcCtx(file_name, dis.Positions(lineno=lineno, col_offset=col_offset)),
    )


def get_inputs(
    scope_symbols: typing.List[Exp[typing.Any, typing.Any]]
) -> typing.List[Exp[typing.Any, typing.Any]]:
    # We only want to consider symbols that have inputs in their rhs (i.e. Nodes)
    # We also want to exclude Mux nodes from the list of inputs
    scope_symbols = [
        symbol
        for symbol in scope_symbols
        if symbol.is_node() and not isinstance(symbol.rhs.val.underlying, Phi)
    ]

    # We use a symbol's id instead of just the symbol objects below because symbols
    # are not hashable and Python complains.
    # Create a dictionary mapping IDs to symbols
    symbol_map = {symbol.rhs.val.id: symbol for symbol in scope_symbols}  # type: ignore -- symbol.rhs.val has already been checked to be a Node

    all_symbol_ids = set(symbol_map.keys())

    # The list of inputs in our scope constitutes the set of all inputs of all symbols
    # minus the set of all symbols defined in this scope
    all_input_ids = set()
    for symbol in scope_symbols:
        inputs = symbol.rhs.val.underlying.inputs  # type: ignore -- symbol.rhs.val has already been checked to be a Node
        symbol_map.update({input.rhs.val.id: input for input in inputs})  # type: ignore -- input.rhs.val has already been checked to be a Node
        all_input_ids.update({input.rhs.val.id for input in inputs})  # type: ignore -- input.rhs.val has already been checked to be a Node

    result_ids = all_input_ids - all_symbol_ids
    return [symbol_map[result_id] for result_id in result_ids]


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


def stage_loop(
    file_name: str,
    lineno: int,
    col_offset: int,
    values: typing.List[Exp[typing.Any, typing.Any]],
    binds: typing.List[Exp[typing.Any, typing.Any]],
    cond: typing.List[Exp[typing.Any, typing.Any]],
    body: typing.List[Exp[typing.Any, typing.Any]],
    outputs: namedtuple,
) -> Ref[typing.Any, typing.Any]:
    condBlk = Block[Boolean](get_inputs(cond), cond, cond[-1])
    bodyBlk = Block[Null](get_inputs(body), body, Null().const(None))
    return stage(
        Loop(values, binds, condBlk, bodyBlk, outputs),
        ctx=SrcCtx(file_name, dis.Positions(lineno=lineno, col_offset=col_offset)),
    )


class VariableTracker:
    def __init__(self):
        self.write_set_stack = [set()]
        self.read_set_stack = [set()]
        self.binds_stack = []

    def push_context(self):
        self.write_set_stack.append(set())
        self.read_set_stack.append(set())

    def pop_context(self):
        curr_write_set = self.write_set_stack.pop()
        self.current_write_set().update(curr_write_set)
        curr_read_set = self.read_set_stack.pop()
        self.current_read_set().update(curr_read_set)

    def push_binds_context(self):
        self.binds_stack.append(set())

    def pop_binds_context(self):
        self.binds_stack.pop()

    def current_write_set(self):
        return self.write_set_stack[-1]

    def current_read_set(self):
        return self.read_set_stack[-1]

    def current_binds(self):
        return self.binds_stack[-1]

    def add_written_var(self, var):
        if (
            (len(self.write_set_stack) > 1 and len(self.read_set_stack) > 1)
            and (
                var not in self.write_set_stack[-1]
                and var not in self.write_set_stack[-2]
            )
            and (var in self.read_set_stack[-1] or var in self.read_set_stack[-2])
        ):
            self.add_bind(var)
        self.current_write_set().add(var)

    def add_read_var(self, var):
        self.current_read_set().add(var)

    def add_bind(self, var):
        self.current_binds().add(var)


class Transformer(ast.NodeTransformer):
    def __init__(self, file_name, calls, ifs, if_exps, loops):
        super().__init__()
        self.counter = 0
        self.unique_prefix = "__________"
        self.file_name = file_name
        self.calls = calls
        self.ifs = ifs
        self.if_exps = if_exps
        self.loops = loops
        self.concrete_to_abstract_flag = False
        self.variable_tracker = VariableTracker()

    def concrete_to_abstract(self, node):
        return ast.Call(
            func=ast.Attribute(
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
                    attr="type_mapper",
                    ctx=ast.Load(),
                ),
                attr="concrete_to_abstract",
                ctx=ast.Load(),
            ),
            args=[node],
            keywords=[],
        )

    def visit_Constant(self, node):
        if self.concrete_to_abstract_flag:
            return self.concrete_to_abstract(node)
        return node

    def visit_Name(self, node):
        # Save the loaded variables
        if isinstance(node.ctx, ast.Load):
            self.variable_tracker.add_read_var(node.id)

        if self.concrete_to_abstract_flag:
            return self.concrete_to_abstract(node)
        return node

    def visit_Assign(self, node):
        # Visit RHS of assignment
        node.value = self.visit(node.value)

        # Recursively process each target to extract all written variables
        for target in node.targets:
            self._process_target(target)

        return node

    def _process_target(self, target):
        # Handle simple variable assignment
        if isinstance(target, ast.Name):
            self.variable_tracker.add_written_var(target.id)
        # Handle tuple or list unpacking
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._process_target(element)
        else:
            raise NotImplementedError("Unsupported target type on LHS of assignment")

    def visit_Break(self, node):
        raise NotImplementedError("Does not support break statements")

    def visit_Continue(self, node):
        raise NotImplementedError("Does not support continue statements")

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
                            ctx=ast.Load(),
                        ),
                        attr="virtualization",
                        ctx=ast.Load(),
                    ),
                    attr="virtualizer",
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
                            ctx=ast.Load(),
                        ),
                        attr="virtualization",
                        ctx=ast.Load(),
                    ),
                    attr="virtualizer",
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
        self.variable_tracker.pop_context()

        # Recursively visit the then body
        self.variable_tracker.push_context()
        node.body = [self.visit(stmt) for stmt in node.body]
        then_write_set = self.variable_tracker.current_write_set()
        then_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.pop_context()

        # Recursively visit the else body
        self.variable_tracker.push_context()
        node.orelse = [self.visit(stmt) for stmt in node.orelse]
        else_write_set = self.variable_tracker.current_write_set()
        else_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.pop_context()

        new_body: typing.List[ast.stmt] = []

        # Run the condition under a different scope
        cond_scope_name = self.generate_temp_var("cond", "scope")
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
                        targets=[
                            ast.Name(id=self.generate_temp_var("cond"), ctx=ast.Store())
                        ],
                        value=node.test,
                    )
                ],
            )
        )

        # Save the previous value of variables if had existed, otherwise set them to Undefined
        for var in self.variable_tracker.current_write_set():
            temp_var = self.generate_temp_var(var, "old")
            temp_var_exists = self.generate_temp_var(var, "old", "exists")
            new_body.extend(
                ast.parse(
                    f"""
try:
    {temp_var} = lambda _, {var} = {var}: __________argon.argon.virtualization.type_mapper.concrete_to_abstract({var})
    {temp_var_exists} = True
except NameError:
    {temp_var} = lambda T : __________argon.argon.virtualization.virtualizer.stage_undefined('{var}', T, '{self.file_name}', {node.lineno}, {node.col_offset})
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
        for var in self.variable_tracker.current_write_set():
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
        for var in self.variable_tracker.current_write_set():
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
                        for var in self.variable_tracker.current_write_set()
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
                        for var in self.variable_tracker.current_write_set()
                    ],
                )
            )

        # Stage if call
        new_body.extend(
            ast.parse(
                f"__________argon.argon.virtualization.virtualizer.stage_if('{self.file_name}', {node.lineno}, {node.col_offset}, {cond_scope_name}.scope.symbols, {then_scope_name}.scope.symbols, {else_scope_name}.scope.symbols)"
            ).body
        )

        # Stage each assigned variable be a mux of the possible values from each branch
        for var in self.variable_tracker.current_write_set():
            temp_var_cond = self.generate_temp_var("cond")
            temp_var_then = self.generate_temp_var(var, "then")
            temp_var_else = self.generate_temp_var(var, "else")
            new_body.extend(
                ast.parse(
                    f"{var} = __________argon.argon.virtualization.virtualizer.stage_phi({temp_var_cond}, {temp_var_then}, {temp_var_else})"
                ).body
            )

        # Delete all temporary variables
        new_body.append(
            ast.Delete(
                targets=[ast.Name(id=self.generate_temp_var("cond"), ctx=ast.Del())]
            )
        )
        for var in self.variable_tracker.current_write_set():
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
        self.variable_tracker.pop_context()

        return node

    def visit_While(self, node):
        # Increment counter to ensure unique names for this while loop
        self.counter += 1

        # Properly set flags before recursive visits
        prev_concrete_to_abstract_flag = self.concrete_to_abstract_flag
        self.concrete_to_abstract_flag = self.loops
        self.variable_tracker.push_context()
        self.variable_tracker.push_binds_context()

        # Recursively visit the condition
        self.variable_tracker.push_context()
        node.test = self.visit(node.test)
        cond_write_set = self.variable_tracker.current_write_set()
        cond_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.pop_context()

        # Recursively visit the body
        self.variable_tracker.push_context()
        node.body = [self.visit(stmt) for stmt in node.body]
        body_write_set = self.variable_tracker.current_write_set()
        body_read_set = self.variable_tracker.current_read_set()
        self.variable_tracker.pop_context()

        if node.orelse:
            raise NotImplementedError("Does not support else statements for loops")

        new_body: typing.List[ast.stmt] = []

        # Create temporary list of input values and binds
        new_body.append(
            ast.Assign(
                targets=[
                    ast.Name(id=self.generate_temp_var("values"), ctx=ast.Store())
                ],
                value=ast.List(elts=[], ctx=ast.Load()),
            )
        )
        new_body.append(
            ast.Assign(
                targets=[ast.Name(id=self.generate_temp_var("binds"), ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
            )
        )

        # Create binds
        # TODO: This part needs to be changed to only create binds for inputs
        for var in self.variable_tracker.current_binds():
            new_body.extend(
                ast.parse(
                    f"""
try:
    {self.generate_temp_var('values')}.append(__________argon.argon.virtualization.type_mapper.concrete_to_abstract({var}))
    {var} = __________argon.argon.virtualization.type_mapper.concrete_to_abstract({var}).bound('{var}')
    {self.generate_temp_var('binds')}.append({var})
except NameError:
    pass
"""
                ).body
            )

        # Run the condition under a different scope
        cond_scope_name = self.generate_temp_var("cond", "scope")
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
                        targets=[
                            ast.Name(id=self.generate_temp_var("cond"), ctx=ast.Store())
                        ],
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
                body=self.modify_body2(node.body),
            )
        )
        new_body.extend(
            ast.parse(
                f"__________argon.argon.virtualization.virtualizer.stage_loop('{self.file_name}', {node.lineno}, {node.col_offset}, {self.generate_temp_var("values")}, {self.generate_temp_var("binds")}, {cond_scope_name}.scope.symbols, {loop_scope_name}.scope.symbols, {self.generate_temp_var("loop", "outputs")})"
            ).body
        )

        # Delete all temporary variables
        # TODO: COMPLETE THIS PART
        new_body.append(
            ast.Delete(
                targets=[ast.Name(id=self.generate_temp_var("cond"), ctx=ast.Del())]
            )
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
        self.variable_tracker.pop_context()
        self.variable_tracker.pop_binds_context()

        return node

    def generate_temp_var(self, *args) -> str:
        return self.unique_prefix + "_".join(args) + "_" + str(self.counter)

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

    def modify_body2(self, body: typing.List[ast.stmt]) -> list:
        new_body = body.copy()
        temp_output_name = self.generate_temp_var("loop", "outputs")
        new_body.append(
            ast.Assign(
                targets=[ast.Name(id=temp_output_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Call(
                        func=ast.Name(id="namedtuple", ctx=ast.Load()),
                        args=[
                            ast.Constant(value=temp_output_name),
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
        return new_body
