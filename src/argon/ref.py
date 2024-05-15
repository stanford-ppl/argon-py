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
        print(self.__orig_class__)
        breakpoint()
        right_type = typing.cast(type, self.A)
        print(self.__orig_class__)
        breakpoint()
        print(type(right_type))
        empty_val: Ref = typing.cast(Ref, right_type().fresh())
        print(type(right_type))
        print(self.__orig_class__)
        breakpoint()
        empty_val.rhs = d
        empty_val.ctx = ctx
        print(self.__orig_class__)
        breakpoint()
        return typing.cast(A, empty_val)

    def const(self, c: C) -> A:
        breakpoint()
        return self._new(Def(Const(c)), SrcCtx.new(2))

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
        return self.__class__.__name__


@dataclass
class Bound[A]:
    id: int
    def_type: typing.Literal["Bound"] = "Bound"

    def dump(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Bound({self.id})"


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Node[A]:
    id: int
    underlying: "Op[A]"
    def_type: typing.Literal["Node"] = "Node"

    def dump(self) -> str:
        return f"x{self.id} = {self.underlying}"
    
    def __str__(self) -> str:
        return f"x{self.id}"


@dataclass
class Const[C]:
    value: C
    def_type: typing.Literal["Const"] = "Const"

    def dump(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"Const({self.value})"


@dataclass
class TypeRef:
    def_type: typing.Literal["TypeRef"] = "TypeRef"

    def dump(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "TypeRef()"


@dataclass
class Def[C, A]:
    val: typing.Union[Const[C], Bound[A], Node[A], TypeRef] = pydantic.Field(
        discriminator="def_type"
    )

    def dump(self) -> str:
        return self.val.dump()
    
    def __str__(self) -> str:
        return str(self.val)


@dataclass
class Exp[C, A](ArgonMeta, abc.ABC):
    """Exp[C, A] defines an expression with denotational type C, and staged type A."""

    rhs: typing.Optional[Def[C, A]] = None
    ctx: typing.Optional[SrcCtx] = None

    @property
    @abc.abstractmethod
    def tp(self) -> ExpType[C, A]:
        raise NotImplementedError()
    
    def dump(self, indent_level = 0) -> str:
        no_indent = '|   ' * indent_level
        indent = '|   ' * (indent_level + 1)
        rhs_str = "None" if self.rhs is None else self.rhs.dump()
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
