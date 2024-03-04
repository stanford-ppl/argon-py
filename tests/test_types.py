
import typing
from argon.ref import ExpType
from argon.types import IntType


def test_inttype():
    assert IntType.L() is int
    assert IntType.R() is IntType


T = typing.TypeVar("T")
class GType[T](ExpType[int, T]):
    pass

class IType(GType[str]):
    pass

def test_generic_type():
    print(IType.L(), IType.R())
    assert IType.L() is int
    assert IType.R() is str
