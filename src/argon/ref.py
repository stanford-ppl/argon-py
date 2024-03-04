import pydantic
from pydantic.dataclasses import dataclass

import abc
import typing

from argon.op import Op

C = typing.TypeVar("C")
A = typing.TypeVar("A")
C_co = typing.TypeVar("C_co")
A_co = typing.TypeVar("A_co")

class ExpType[C_co, A](abc.ABC):
    @classmethod
    def _get_params(cls) -> typing.Tuple[typing.Type[C_co], typing.Type[A]]:
        print(cls.__orig_bases__)
        expt = cls.__orig_bases__[0]  # type: ignore[attr-defined]
        augmented_locals = locals() | {cls.__name__: cls}

        return tuple(
            (
                klass._evaluate(globals(), augmented_locals, frozenset())
                if isinstance(klass, typing.ForwardRef)
                else klass
            )
            for klass in typing.get_args(expt)
        )

    @classmethod
    def L(cls) -> typing.Type[C_co]:
        return cls._get_params()[0]

    @classmethod
    def R(cls) -> typing.Type[A]:
        return cls._get_params()[1]

@dataclass
class Bound[A]:
    id: int
    def_type: typing.Literal["Bound"] = "Bound"


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Operation[A]:
    underlying: Op[A]
    def_type: typing.Literal["Op"] = "Op"


@dataclass
class Const[C_co]:
    value: C_co
    def_type: typing.Literal["Const"] = "Const"


@dataclass
class Def[C, A]:
    val: typing.Union[Const[C], Bound[A], Operation[A]] = pydantic.Field(
        discriminator="def_type"
    )


class Exp[C_co, A_co](abc.ABC):
    """Exp[C, A] defines an expression with denotational type C, and staged type A."""

    @abc.abstractproperty
    def tp(self) -> ExpType[C_co, A_co]:
        raise NotImplementedError()

    @abc.abstractproperty
    def rhs(self) -> Def[C_co, A_co]:
        raise NotImplementedError()


class Ref[C_co, A_co](ExpType[C_co, A_co], Exp[C_co, A_co]):
    pass
