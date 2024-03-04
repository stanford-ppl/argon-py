import typing
from argon.ref import ExpType
from argon.types.integer import Integer


def test_inttype():
    assert Integer.L() is int
    assert Integer.R() is Integer


T = typing.TypeVar("T")


class GType[T](ExpType[int, T]):
    pass


class IType(GType[str]):
    pass


def test_generic_type():
    print(IType.L(), IType.R())
    assert IType.L() is int
    assert IType.R() is str
