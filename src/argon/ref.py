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
    """
    ExpType[C, A] defines the type of an expression with denotational type C, and staged
    type A. This class should be subclassed to define custom expression types.
    """

    @classmethod
    def type_name(cls) -> str:
        return cls.__name__

    @abc.abstractmethod
    def fresh(self) -> A:
        raise NotImplementedError()

    def _new(self, d: "Def[C, A]", ctx: SrcCtx) -> A:  # type: ignore -- Pyright falsely detects C and A as abstractproperty instead of type variables
        right_type = typing.cast(type, self.A)

        empty_val: Ref = typing.cast(Ref, right_type().fresh())

        empty_val.rhs = d
        empty_val.ctx = ctx
        return typing.cast(A, empty_val)  # type: ignore -- Pyright falsely reports the type is not compatible with the return type

    def bound(self, name: str) -> A:
        from argon.state import State

        return self._new(
            Def(Bound(State.get_current_state().next_id(), name)), SrcCtx.new(2)
        )

    def const(self, c: C) -> A:
        return self._new(Def(Const(c)), SrcCtx.new(2))  # type: ignore -- Pyright falsely reports the type is not compatible with the return type

    # A, C shims are used to silence typing errors -- their actual definitions are provided by ArgonMeta
    @abc.abstractproperty
    def A(self) -> typing.Type[A]:
        raise NotImplementedError()

    @abc.abstractproperty
    def C(self) -> typing.Type[C]:
        raise NotImplementedError()

    # A shim method that silences typing errors as something like Integer() or tp()
    # is used to instantiate a type.
    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> typing.Any:
        raise NotImplementedError()

    @property
    def tp_name(self) -> str:
        return self.type_name()


@dataclass
class Bound[A]:
    id: int
    name: str
    def_type: typing.Literal["Bound"] = "Bound"

    def dump(self, indent_level=0) -> str:
        return f"b{self.id} = Bound('{self.name}')"

    def __str__(self) -> str:
        return f"b{self.id}"


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Node[A]:
    id: int
    underlying: "Op[A]"
    def_type: typing.Literal["Node"] = "Node"

    def dump(self, indent_level=0) -> str:
        return f"x{self.id} = {self.underlying.dump(indent_level)}"

    def __str__(self) -> str:
        return f"x{self.id}"


@dataclass
class Const[C]:
    value: C
    def_type: typing.Literal["Const"] = "Const"

    def dump(self, indent_level=0) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Const({self.value})"


@dataclass
class TypeRef:
    def_type: typing.Literal["TypeRef"] = "TypeRef"

    def dump(self, indent_level=0) -> str:
        return str(self)

    def __str__(self) -> str:
        return "TypeRef()"


@dataclass
class Def[C, A]:
    val: typing.Union[Const[C], Bound[A], Node[A], TypeRef] = pydantic.Field(
        discriminator="def_type"
    )

    def dump(self, indent_level=0) -> str:
        return self.val.dump(indent_level)

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class Exp[C, A](ArgonMeta, abc.ABC):
    """
    Exp[C, A] defines an expression with denotational type C, and staged type A. This
    class should be subclassed to define custom expressions.

        rhs : Optional[Def[C, A]]
            The definition of the expression.
        ctx : Optional[SrcCtx]
            The source context of the expression.
    """

    rhs: typing.Optional[Def[C, A]] = None
    ctx: typing.Optional[SrcCtx] = None

    @property
    @abc.abstractmethod
    def tp(self) -> ExpType[C, A]:
        raise NotImplementedError()

    def is_bound(self) -> bool:
        return self.rhs != None and isinstance(self.rhs.val, Bound)

    def is_node(self) -> bool:
        return self.rhs != None and isinstance(self.rhs.val, Node)

    def is_const(self) -> bool:
        return self.rhs != None and isinstance(self.rhs.val, Const)

    def is_typeref(self) -> bool:
        return self.rhs != None and isinstance(self.rhs.val, TypeRef)

    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        rhs_str = "None" if self.rhs is None else self.rhs.dump(indent_level)
        return (
            f"{rhs_str}( \n"
            f"{indent}tp: {self.tp.tp_name} \n"
            f"{indent}ctx: {self.ctx} \n"
            f"{no_indent})"
        )

    def __str__(self) -> str:
        return str(self.rhs)


class Ref[C, A](ExpType[C, A], Exp[C, A]):

    @property
    @typing.override
    def tp(self) -> ExpType[C, A]:
        return self


S_co = typing.TypeVar("S_co", covariant=True)
type Sym[S_co] = Exp[typing.Any, S_co]

type Type[A] = ExpType[typing.Any, A]

from argon.op import Op
