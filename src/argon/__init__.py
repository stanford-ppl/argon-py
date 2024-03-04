
import typing
from argon.ref import Exp, ExpType

S_co = typing.TypeVar("S_co", covariant=True)
type Sym[S_co] = Exp[typing.Any, S_co]

S = typing.TypeVar("S")
type Type[S] = ExpType[typing.Any, S]
