import abc
import typing
from argon.base import ArgonMeta
from argon.srcctx import SrcCtx
from argon.utils import compute_types
from pydantic.dataclasses import dataclass
import pydantic

# Experimentally extracted value which
# grabs the context which called stage()
_PYDANTIC_SCOPE_DEPTH = 3


@dataclass
class Op[R](ArgonMeta, abc.ABC):
    ctx: SrcCtx = pydantic.Field(
        default_factory=lambda: SrcCtx.new(_PYDANTIC_SCOPE_DEPTH), kw_only=True
    )

    @abc.abstractproperty
    def inputs(self) -> typing.List["Sym[typing.Any]"]:
        raise NotImplementedError()


from argon.ref import Sym
