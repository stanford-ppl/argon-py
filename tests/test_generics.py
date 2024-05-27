from pydantic.dataclasses import dataclass
from typing import Union, override, List, TypeVar
from argon.ref import Ref
from argon.state import State
import typing


@dataclass
class Stop:
    level: int

    def __str__(self) -> str:
        return f"S{self.level}"


@dataclass
class FVal:
    value: float

    def __str__(self) -> str:
        return str(self.value)


T = TypeVar("T")


class FStream[T](Ref[List[Union[FVal, Stop]], "FStream[T]"]):

    @override
    def fresh(self) -> "FStream[T]":
        return FStream[self.T]()  # type: ignore -- PyRight falsely report that it cannot access the type parameter


def test_tparam_in_staged_tp():
    state = State()
    with state:
        a = FStream[int]().const(
            [FVal(1.0), FVal(2.0), Stop(1), FVal(3.0), FVal(4.0), Stop(2)]
        )
        assert a.C == typing.List[typing.Union[FVal, Stop]]
        assert a.A is FStream[int]
        assert a.T is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
        assert a.A().T is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
    print(state)


I = TypeVar("I")


class IStream[I](Ref[List[Union[FVal, Stop]], "IStream[int]"]):

    @override
    def fresh(
        self,
    ) -> (
        "IStream[int]"
    ):  # In the function signature of fresh, the return type should be A
        return IStream[int]()


def test_cls_tparam_retrieve():
    state = State()
    with state:
        a = IStream[str]()

        assert type(a) is IStream
        assert a.C == typing.List[typing.Union[FVal, Stop]]
        assert a.A is IStream[int]
        assert a.I is str  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
    print(state)


def test_staged_tp_tparam_retrieve():
    state = State()
    with state:
        a = IStream[str]().const(
            [FVal(1.0), FVal(2.0), Stop(1), FVal(3.0), FVal(4.0), Stop(2)]
        )
        # The becomes A (i.e., IStream[int]) because const returns an instance of type A

        assert type(a) is IStream
        assert a.C == typing.List[typing.Union[FVal, Stop]]
        assert a.A is IStream[int]
        assert a.A().I is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
        # -- a.A() is equivalent to IStream[int]()
        assert a.I is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
        # As 'a' is now an instance of type A (IStream[int]), the type param 'I' should be int
    print(state)


D = TypeVar("D")


class DStream[D](Ref[List[D], "DStream[D]"]):

    @override
    def fresh(self) -> "DStream[D]":
        return DStream[self.D]()  # type: ignore -- PyRight falsely reports that it cannot access the type parameter


def test_gtparam_in_denotational_tp():
    state = State()
    with state:
        a = DStream[int]().const([1, 2, 3])
        assert type(a) is DStream
        assert a.C == typing.List[int]  # THIS IS THE MAIN THING WE WANT TO TEST
        assert a.A is DStream[int]  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
        assert a.A().D is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
        assert a.D is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
    print(state)


E = TypeVar("E")


@dataclass
class GVal[E]:
    value: E

    def __str__(self) -> str:
        return str(self.value)


TP = TypeVar("TP")


class GStream[TP](Ref[List[Union[GVal[TP], Stop]], "GStream[TP]"]):

    @override
    def fresh(self) -> "GStream[TP]":
        return GStream[self.TP]()  # type: ignore -- PyRight falsely reports that it cannot access the type parameter


def test_fully_generic_tp():
    state = State()
    with state:
        a = GStream[float]().const(
            [
                GVal[float](1.0),
                GVal[float](2.0),
                Stop(1),
                GVal[float](3.0),
                GVal[float](4.0),
                Stop(2),
            ]
        )
        # This becomes type A (i.e., GStream[int]) because const returns an instance of type A

        assert type(a) is GStream
        assert a.C == typing.List[typing.Union[GVal[float], Stop]]
        assert a.A is GStream[float]
        assert a.A().TP is float  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
        assert a.TP is float  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
    print(state)


def test_fully_generic_tp_int():
    state = State()
    with state:
        b = GStream[int]().const(
            [
                GVal[int](1),
                GVal[int](2),
                Stop(1),
                GVal[int](3),
                GVal[int](4),
                Stop(2),
            ]
        )
        # This becomes type A (i.e., GStream[int]) because const returns an instance of type A

        assert type(b) is GStream
        assert b.C == typing.List[typing.Union[GVal[int], Stop]]
        assert b.A is GStream[int]
        assert b.A().TP is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
        assert b.TP is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
    print(state)


TP1 = TypeVar("TP1")


class TPStream[TP1](Ref[List[Union[GVal[TP1], Stop]], GStream[List[TP1]]]):
    @override
    def fresh(self) -> GStream[List[TP1]]:
        return GStream[List[TP1]]()


def test_stream_as_staged_type():
    # type: ignore -- PyRight falsely reports that it cannot access the type parameter

    assert TPStream[int]().TP1 is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
    assert TPStream[int]().const([GVal[int](3), Stop(1)]).TP1 == List[int]  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
    # note: The return type of TPStream[int]().const([GVal[int](3), Stop(1)]) is the staged type of the class which is, GStream[List[TP]]]


# Create a case where the parent and class share the same type parameters
class ChildStream[TP](GStream[List[TP]]):
    @override
    def fresh(self) -> "ChildStream[TP]":
        return ChildStream[self.TP]()  # type: ignore -- PyRight falsely reports that it cannot access the type parameter


def test_parent_child_t_param():
    assert ChildStream[int]().TP is int  # type: ignore -- PyRight falsely reports that it cannot access the type parameter
