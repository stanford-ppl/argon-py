import abc
import typing

A = typing.TypeVar("A", covariant=True)

class Op[A](abc.ABC):
    pass
