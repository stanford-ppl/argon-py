import typing
from argon.base import ArgonMeta
from argon.ref import ExpType
from argon.types.integer import Integer

# class for defining
T = typing.TypeVar("T")


class GType[T](ExpType[int, T]):
    pass


U = typing.TypeVar("U")


class IType[U](GType[U]):
    @typing.override
    def fresh(self) -> U:
        return IType[self.U]()  # type: ignore


def test_generic_type():
    tp_instance = IType[str]()
    assert tp_instance.A is str
    assert tp_instance.C is int


class TestMeta[T](ArgonMeta):
    pass


class TestConcrete(TestMeta[int]):
    pass


class TestRecursive(TestMeta["TestRecursive"]):
    pass


F = typing.TypeVar("F")


class TestRecursive2[F](TestMeta["TestRecursive2[F]"]):
    pass


def test_concretized():
    assert TestConcrete().T is int  # type: ignore


def test_recursive():
    assert TestRecursive().T is TestRecursive  # type: ignore


def test_recursive_generic():
    assert TestRecursive2[int]().F is int  # type: ignore
