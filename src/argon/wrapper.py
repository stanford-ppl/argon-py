import functools
import inspect
import ast

from argon.node.control import TransformIfExpressions

def argon_function(func):
    # TODO: fix ctx, when this decorator is used in test_scopes.py,
    # the ctx no longer points to the correct row number + col offset

    # Get the source code of the function
    src = inspect.getsource(func)
    # Parse it into an AST
    parsed = ast.parse(src)

    # Remove the decorators from the AST, because the modified function will
    # be passed to them anyway and we don't want them to be called twice.
    func_stmt = parsed.body[0]
    func_stmt.decorator_list = [
        dec for dec in func_stmt.decorator_list
        if not (isinstance(dec, ast.Name) and dec.id == 'argon_function')
    ]
 
    # Apply the AST transformation
    transformed = TransformIfExpressions().visit(parsed)
    transformed = ast.fix_missing_locations(transformed)

    print(ast.unparse(transformed))

    # Compile the transformed AST
    compiled = compile(transformed, filename=func.__code__.co_filename, mode='exec')
    # Create a new function from the compiled code
    func_globals = func.__globals__
    exec(compiled, func_globals)

    # Replace the original function with the transformed version
    new_func = func_globals[func.__name__]
    return functools.update_wrapper(new_func, func)
