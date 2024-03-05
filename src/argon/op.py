import abc
import typing
from argon.srcctx import SrcCtx
from argon.utils import compute_types
from pydantic.dataclasses import dataclass
import pydantic

# Experimentally extracted value which
# grabs the context which called stage()
_PYDANTIC_SCOPE_DEPTH = 3


@dataclass
class Op[A](abc.ABC):
    ctx: SrcCtx = pydantic.Field(
        default_factory=lambda: SrcCtx.new(_PYDANTIC_SCOPE_DEPTH), kw_only=True
    )

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
