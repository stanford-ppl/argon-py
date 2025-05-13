import abc
import dis
import types
from typing import get_args, override, Protocol
import typing
import inspect

import pydantic
from argon.block import Block
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import State, stage
from argon.virtualization.func import ArgonFunction
from argon.virtualization.type_mapper import (
    concrete_to_abstract,
    concrete_to_bound,
    concrete_to_abstract_type,
)


@typing.runtime_checkable
class FunctionWithVirt(Protocol):
    virtualized: ArgonFunction


RETURN_TP = typing.TypeVar("RETURN_TP")


class Function[RETURN_TP](Ref[FunctionWithVirt, "Function[RETURN_TP]"]):
    """
    The Function class represents a function in the Argon language.
    """

    @override
    def fresh(self) -> "Function[RETURN_TP]":  # type: ignore -- Pyright falsely detects F as an abstractproperty instead of type variable
        return Function[self.RETURN_TP]()

    # RETURN_TP shim is used to silence typing errors -- its actual definition is provided by ArgonMeta
    @abc.abstractproperty
    def RETURN_TP(self) -> typing.Type[RETURN_TP]:
        raise NotImplementedError()

    @override
    @property
    def tp_name(self) -> str:
        result = f"{self.type_name()}[{self.RETURN_TP.type_name()}"
        current_type = self.RETURN_TP
        while get_args(current_type):
            args = get_args(current_type)
            result += f"[{args[0].type_name()}"
            current_type = args[0]
        result += "]" * (result.count("[") - result.count("]"))
        return result


def function_C_to_A(c: types.FunctionType) -> Function:
    if not (hasattr(c, "virtualized") and isinstance(c.virtualized, ArgonFunction)):  # type: ignore -- Pyright complains about accessing virtualized
        from argon.virtualization.wrapper import argon_function

        c_with_virt = argon_function()(c)
        # Update the original function in its module's namespace
        module = inspect.getmodule(c)
        if module is not None:
            setattr(module, c.__name__, c_with_virt)
    else:
        c_with_virt = typing.cast(FunctionWithVirt, c)

    if c_with_virt.virtualized.abstract_func is not None:
        return c_with_virt.virtualized.abstract_func

    annotated_concrete_return_type = c_with_virt.virtualized.get_return_type()
    annotated_abstract_return_type = concrete_to_abstract_type[
        annotated_concrete_return_type
    ]
    name = c_with_virt.virtualized.get_function_name()
    body = Block[annotated_abstract_return_type]([], [], None)

    from argon.node.function_new import FunctionNew

    c_with_virt.virtualized.abstract_func = stage(
        FunctionNew[Function[annotated_abstract_return_type]](
            name,
            [],
            body,
            c_with_virt,
        ),
        ctx=SrcCtx(
            c.__code__.co_filename,
            dis.Positions(lineno=c.__code__.co_firstlineno, col_offset=0),
        ),
    )

    # get the return type of a function by actually calling it
    param_names = c_with_virt.virtualized.get_param_names()
    scope_context = State.get_current_state().new_scope()
    with scope_context:
        def create_bound_arg(param_name: str) -> Ref[typing.Any, typing.Any]:
            param_type = c_with_virt.virtualized.get_param_type(param_name)
            return concrete_to_bound[param_type](param_name)
        bound_args = [create_bound_arg(param_name) for param_name in param_names]
        ret = c_with_virt.virtualized.call_transformed(*bound_args)
        ret = concrete_to_abstract(ret)

    # check that ret.C is the same as c_with_virt's return type
    assert (
        ret.A == annotated_abstract_return_type
    ), f"Function {c.__name__} was annotated with return type {annotated_concrete_return_type}, but the actual return type is {ret.C}"

    name = c_with_virt.virtualized.get_function_name()

    body.inputs = scope_context.scope.inputs
    body.stms = scope_context.scope.symbols
    body.result = ret

    c_with_virt.virtualized.abstract_func.rhs.val.underlying.binds = bound_args
  
    return c_with_virt.virtualized.abstract_func


concrete_to_abstract[types.FunctionType] = function_C_to_A
concrete_to_bound[types.FunctionType] = (
    lambda func_tp: lambda name: concrete_to_abstract_type[func_tp]().bound(name)
)


def function_C_to_AT(func_tp: typing.Callable) -> typing.Type[Function]:
    assert len(get_args(func_tp)) == 2, f"The type annotation {func_tp} is not valid"
    func_args_tp, func_ret_tp = get_args(func_tp)
    return Function[concrete_to_abstract_type[func_ret_tp]]


concrete_to_abstract_type[types.FunctionType] = function_C_to_AT

pydantic.dataclasses.rebuild_dataclass(ArgonFunction)
