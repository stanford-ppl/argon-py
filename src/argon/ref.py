import pydantic
from pydantic.dataclasses import dataclass

import abc
import typing

from argon.utils import compute_types, resolve_class


C = typing.TypeVar("C")
A = typing.TypeVar("A")
C_co = typing.TypeVar("C_co")
A_co = typing.TypeVar("A_co")


class ExpType[C_co, A](abc.ABC):
    _cached_lr: typing.ClassVar[typing.Optional[typing.Mapping[str, typing.Type]]] = (
        None
    )

    @classmethod
    def _compute_LR(cls):
        if cls._cached_lr:
            return
        cls._cached_lr = compute_types(cls, ExpType, {cls.__name__: cls})

    @classmethod
    def L(cls) -> typing.Type[C_co]:
        cls._compute_LR()
        assert cls._cached_lr is not None
        return cls._cached_lr["C_co"]

    @classmethod
    def R(cls) -> typing.Type[A]:
        cls._compute_LR()
        assert cls._cached_lr is not None
        return cls._cached_lr["A"]

    @classmethod
    @abc.abstractmethod
    def fresh(cls) -> A:
        raise NotImplementedError()

    @classmethod
    def _new(cls, d: "Def[C_co, A]") -> A:
        right_type = cls.R()
        assert issubclass(right_type, Ref)
        empty_val = right_type.fresh()
        empty_val.rhs = d
        return empty_val

    @classmethod
    def const(cls, c: C_co) -> A:
        return cls._new(Def(Const(c)))


@dataclass
class Bound[A]:
    id: int
    def_type: typing.Literal["Bound"] = "Bound"


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Node[A]:
    id: int
    underlying: "Op[A]"
    def_type: typing.Literal["Node"] = "Node"


@dataclass
class Const[C_co]:
    value: C_co
    def_type: typing.Literal["Const"] = "Const"


@dataclass
class Def[C, A]:
    val: typing.Union[Const[C], Bound[A], Node[A]] = pydantic.Field(
        discriminator="def_type"
    )


class Exp[C_co, A_co](abc.ABC):
    """Exp[C, A] defines an expression with denotational type C, and staged type A."""

    rhs: typing.Optional[Def[C_co, A_co]] = None

    @abc.abstractproperty
    def tp(self) -> ExpType[C_co, A_co]:
        raise NotImplementedError()


class Ref[C_co, A_co](ExpType[C_co, A_co], Exp[C_co, A_co]):

    @property
    @typing.override
    def tp(self) -> ExpType[C_co, A_co]:
        return self


S_co = typing.TypeVar("S_co", covariant=True)
type Sym[S_co] = Exp[typing.Any, S_co]

type Type[S] = ExpType[typing.Any, S]

from argon.op import Op
