import abc
import typing
from argon.utils import compute_types


class Op[A](abc.ABC):
    _tp_cache: typing.ClassVar[typing.Optional[typing.Type]] = None

    @classmethod
    def R(cls) -> typing.Type[A]:
        if cls._tp_cache:
            return cls._tp_cache
        cls._tp_cache = compute_types(cls, Op, {cls.__name__: cls})["A"]
        assert cls._tp_cache is not None
        return cls._tp_cache

    @abc.abstractproperty
    def inputs(self) -> typing.List["Sym[typing.Any]"]:
        raise NotImplementedError()


from argon.ref import Sym
