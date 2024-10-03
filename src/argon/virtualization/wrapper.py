import functools
import inspect
import ast
import typing

from argon.types.function import FunctionWithVirt
from argon.virtualization.func import ArgonFunction
from argon.virtualization.virtualizer import Transformer


# TODO: After implementing more transformations, add relevant flags to the decorator to enable/disable them
def argon_function(calls=True, ifs=True, if_exps=True):
    """
    This decorator is used to virtualize a function. It takes three optional arguments that are by default all set to True:

        calls: bool
            Determines whether function calls will be virtualized.
        ifs: bool
            Determines whether if statements will be virtualized.
        if_exps: bool
            Determines whether if expressions will be virtualized.

    Examples:

        Tagging a function with `@argon_function()` will virtualize it with all transformations enabled.

        Tagging a function with `@argon_function(calls=False)` will virtualize it with calls disabled and all other transformations enabled.
    """

    def decorator(func) -> FunctionWithVirt:
        # TODO: fix ctx, when this decorator is used in test_scopes.py,
        # the ctx no longer points to the correct row number + col offset

        # Get the source code of the file where the function is defined
        src = inspect.getfile(func)
        # Read the entire file's source
        with open(src, "r") as file:
            file_src = file.read()
        # Parse it into an AST
        parsed = ast.parse(file_src)

        # Remove the decorators from the AST, because the modified function will
        # be passed to them anyway and we don't want them to be called twice.
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                node.decorator_list = [
                    dec
                    for dec in node.decorator_list
                    if not (
                        (
                            isinstance(dec, ast.Call)
                            and isinstance(dec.func, ast.Name)
                            and dec.func.id == "argon_function"
                        )
                    )
                ]
                node.args.kw_defaults.append(ast.Constant(value=None))
                node.args.kwonlyargs.append(
                    ast.arg(arg="__________argon", annotation=None)
                )
                func_src = node
                break

        # Apply the AST transformation
        # TODO: Add the transformation flags here too!
        transformed = Transformer(src, calls, ifs, if_exps).visit(func_src)
        transformed = ast.fix_missing_locations(transformed)

        # Create a new AST containing only the transformed function
        transformed_module = ast.Module(body=[transformed], type_ignores=[])

        # Compile the transformed AST
        compiled = compile(
            transformed_module, filename=func.__code__.co_filename, mode="exec"
        )
        # Create a new function from the compiled code
        func_globals = func.__globals__
        exec(compiled, func_globals)

        # Replace the original function with the transformed version
        virtualized_func = ArgonFunction(
            func, functools.update_wrapper(func_globals[func.__name__], func)
        )

        # Create a wrapper function that calls the transformed function
        # Pytest will not be able to collect the test functions if they are not
        def wrapper(*args, **kwargs):
            return virtualized_func.call_original(*args, **kwargs)

        wrapper.virtualized = virtualized_func  # type: ignore -- we need to add this flag to mark the function as virtualized

        return typing.cast(FunctionWithVirt, functools.update_wrapper(wrapper, func))

    return decorator
