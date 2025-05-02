import typing
from argon.ref import Ref


# This class is used to map an instance of a concrete type 
# to an instance of its corresponding abstract type
class _CToA:
    def __init__(self):
        self.C_to_A_map = {}

    def __setitem__(self, tp_c, tp_a_initializer: typing.Callable) -> None:
        self.C_to_A_map[tp_c] = tp_a_initializer

    def __call__(self, c) -> Ref[typing.Any, typing.Any]:
        if type(c) in self.C_to_A_map:
            return self.C_to_A_map[type(c)](c)
        elif isinstance(c, Ref):
            return c
        else:
            raise ValueError(f"Cannot convert {c} to abstract type")


concrete_to_abstract = _CToA()

# This class is used to map a concrete type to a bound variable
# of its corresponding abstract type
class _CToB:
    def __init__(self):
        self.C_to_B_map = {}
    
    def __setitem__(self, tp_c, tp_b_initializer: typing.Callable) -> None:
        self.C_to_B_map[tp_c] = tp_b_initializer
    
    # TODO: make the return type more specific
    def __getitem__(self, tp_c) -> typing.Callable:
        if tp_c in self.C_to_B_map:
            return self.C_to_B_map[tp_c]
        else:
            raise ValueError(f"Cannot convert {tp_c} to bound variable")

concrete_to_bound = _CToB()