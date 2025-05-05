import abc
from collections.abc import Callable
import dis
import types
from typing import get_args, override, Protocol
import typing
from argon.block import Block
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import State, stage
from argon.types.boolean import Boolean
from argon.types.integer import Integer
from argon.types.null import Null
from argon.virtualization.func import ArgonFunction
from argon.virtualization.type_mapper import concrete_to_abstract, concrete_to_bound


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
    else:
        c_with_virt = typing.cast(FunctionWithVirt, c)

    # get the return type of a function by actually calling it
    # TODO: add some check to see if we've already run it with args of the same type
    param_names = c_with_virt.virtualized.get_param_names()
    scope_context = State.get_current_state().new_scope()
    with scope_context:
        def create_bound_arg(param_name: str) -> Ref[typing.Any, typing.Any]:
            param_type = c_with_virt.virtualized.get_param_type(param_name)
            if typing.get_origin(param_type) is Callable:
                return concrete_to_bound[types.FunctionType](param_name, param_type)
            return concrete_to_bound[param_type](param_name)
        bound_args = [create_bound_arg(param_name) for param_name in param_names]
        ret = c_with_virt.virtualized.call_transformed(*bound_args)
        ret = concrete_to_abstract(ret)

    # check that ret.C is the same as c_with_virt's return type
    assert (
        ret.C == c_with_virt.virtualized.get_return_type()
    ), f"Function {c.__name__} was annotated with return type {c_with_virt.virtualized.get_return_type()}, but the actual return type is {ret.C}"

    name = c_with_virt.virtualized.get_function_name()

    body = Block[ret.A](scope_context.scope.inputs, scope_context.scope.symbols, ret)

    from argon.node.function_new import FunctionNew

    return stage(
        FunctionNew[Function[ret.A]](name, bound_args, body, c_with_virt),
        ctx=SrcCtx(
            c.__code__.co_filename,
            dis.Positions(lineno=c.__code__.co_firstlineno, col_offset=0),
        ),
    )


concrete_to_abstract[types.FunctionType] = function_C_to_A


def function_C_to_B(name: str, func_tp: typing.Callable) -> Function:
    assert len(get_args(func_tp)) == 2, f"The type annotation {func_tp} is not valid"
    func_args_tp, func_ret_tp = get_args(func_tp)
    # TODO: still need to handle the case where func_ret_tp is yet another Callable (i.e. we have a function that returns a function)
    if func_ret_tp is None:
        return Function[Null]().bound(name)
    elif func_ret_tp is bool:
        return Function[Boolean]().bound(name)
    elif func_ret_tp is int:
        return Function[Integer]().bound(name)
    else:
        raise ValueError(f"The return type {func_ret_tp} is not supported")


concrete_to_bound[types.FunctionType] = function_C_to_B
