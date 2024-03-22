from typing import override
from argon.ref import Ref
import fxpmath


class FixedPoint(Ref[fxpmath.Fxp, "FixedPoint"]):
    @override
    @classmethod
    def fresh(cls) -> "FixedPoint":
        return FixedPoint()
