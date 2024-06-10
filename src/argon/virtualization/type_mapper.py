import types
import typing
from argon.ref import Ref


class _CToA:
    def __init__(self):
        self.C_to_A_map = {}

    def __setitem__(self, tp_c, tp_a_initializer: typing.Callable) -> None:
        self.C_to_A_map[tp_c] = tp_a_initializer

    def function(self, c, args: typing.List[Ref[typing.Any, typing.Any]]) -> "Function":
        if not isinstance(c, types.FunctionType):
            raise ValueError(f"Expected function, got {c}")
        return self.C_to_A_map[types.FunctionType](c, args)

    def __call__(self, c) -> Ref[typing.Any, typing.Any]:
        if type(c) in self.C_to_A_map:
            return self.C_to_A_map[type(c)](c)
        elif isinstance(c, Ref):
            return c
        else:
            raise ValueError(f"Cannot convert {c} to abstract type")


concrete_to_abstract = _CToA()

from argon.types.function import Function
