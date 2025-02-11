import abc
import dis
import types
from typing import override, Protocol
import typing
from argon.block import Block
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import State, stage
from argon.virtualization.func import ArgonFunction
from argon.virtualization.type_mapper import concrete_to_abstract


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


def function_C_to_A(
    c: types.FunctionType, args: typing.List[Ref[typing.Any, typing.Any]]
) -> Function:
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
        bound_args = [arg.bound(name) for arg, name in zip(args, param_names)]
        ret = c_with_virt.virtualized.call_transformed(*bound_args)
        ret = concrete_to_abstract(ret)

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
