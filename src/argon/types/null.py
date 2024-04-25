from types import NoneType
from typing import override
from argon.ref import Ref

class Null(Ref[NoneType, "Null"]):
    @override
    def fresh(self) -> "Null":
        return Null()
