import typing
from argon.ref import ExpType
from argon.types.integer import Integer

# class for defining
T = typing.TypeVar("T")


class GType[T](ExpType[int, T]):
    pass


U = typing.TypeVar("U")


class IType[U](GType[U]):
    @typing.override
    def fresh(self):
        return IType[self.U]()


def test_generic_type():
    tp_instance = IType[str]()
    # print(tp_instance.C, tp_instance.A)
    assert tp_instance.A is str
    assert tp_instance.C is int
