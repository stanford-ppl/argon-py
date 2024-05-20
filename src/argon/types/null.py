from types import NoneType
from typing import override
from argon.ref import Ref


class Null(Ref[NoneType, "Null"]):
    """
    The Null class represents a value of None in the Argon language.
    """

    @override
    def fresh(self) -> "Null":
        return Null()
