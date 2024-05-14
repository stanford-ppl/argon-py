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
    """
    The Op[R] class represents an operation that computes a result of type R. This
    class should be subclassed to define custom operations.
    """

    @abc.abstractproperty
    def inputs(self) -> typing.List["Sym[typing.Any]"]:
        raise NotImplementedError()

    def dump(self, indent_level=0) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(str, self.inputs))})"


from argon.ref import Sym
