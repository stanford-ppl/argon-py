from types import NoneType
from typing import override
from argon.ref import Ref
from argon.virtualization.type_mapper import concrete_to_abstract, concrete_to_bound


class Null(Ref[NoneType, "Null"]):
    """
    The Null class represents a value of None in the Argon language.
    """

    @override
    def fresh(self) -> "Null":
        return Null()


concrete_to_abstract[NoneType] = lambda x: Null().const(x)
concrete_to_bound[NoneType] = lambda name: (_ for _ in ()).throw(TypeError("Cannot bind NoneType"))
