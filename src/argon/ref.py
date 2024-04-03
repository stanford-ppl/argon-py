import pydantic
from pydantic.dataclasses import dataclass

import abc
import typing
from argon.base import ArgonMeta
from argon.srcctx import SrcCtx

from argon.utils import compute_types


C = typing.TypeVar("C")
A = typing.TypeVar("A")


class ExpType[C, A](ArgonMeta, abc.ABC):
    @abc.abstractmethod
    def fresh(self) -> A:
        raise NotImplementedError()

    def _new(self, d: "Def[C, A]", ctx: SrcCtx) -> A:
        right_type = self.A
        assert issubclass(right_type, Ref)
        empty_val = right_type.fresh()
        empty_val.rhs = d
        empty_val.ctx = ctx
        return empty_val

    def const(self, c: C) -> A:
        return self._new(Def(Const(c)), SrcCtx.new(2))


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
class Const[C]:
    value: C
    def_type: typing.Literal["Const"] = "Const"


@dataclass
class TypeRef:
    def_type: typing.Literal["TypeRef"] = "TypeRef"


@dataclass
class Def[C, A]:
    val: typing.Union[Const[C], Bound[A], Node[A], TypeRef] = pydantic.Field(
        discriminator="def_type"
    )


@dataclass
class Exp[C, A](ArgonMeta, abc.ABC):
    """Exp[C, A] defines an expression with denotational type C, and staged type A."""

    rhs: typing.Optional[Def[C, A]] = None
    ctx: typing.Optional[SrcCtx] = None

    @property
    @abc.abstractmethod
    def tp(self) -> ExpType[C, A]:
        raise NotImplementedError()


class Ref[C, A](ExpType[C, A], Exp[C, A]):

    @property
    @typing.override
    def tp(self) -> ExpType[C, A]:
        return self


S_co = typing.TypeVar("S_co", covariant=True)
type Sym[S_co] = Exp[typing.Any, S_co]

type Type[A] = ExpType[typing.Any, A]

from argon.op import Op
