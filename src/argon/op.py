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

    _tp_cache: typing.Optional[typing.Type] = pydantic.Field(default=None, kw_only=True)

    def R(self) -> typing.Type[A]:
        if self._tp_cache:
            return self._tp_cache
        ns: dict[str, typing.Type] = {self.__class__.__name__: self.__class__}
        if hasattr(self, "__orig_class__"):
            # Was instantiated from a generic, populate the inference scope
            # The following lines are tagged with type: ignore because tools don't properly handle hasattr inference.
            args = typing.get_args(self.__orig_class__)  # type: ignore
            params = self.__orig_class__.__origin__.__type_params__  # type: ignore
            ns.update({p.__name__: a for p, a in zip(params, args)})
        tp = compute_types(self.__class__, Op, ns)["A"]
        if isinstance(tp, typing.TypeVar):
            raise TypeError(
                f"We do not currently support type inference. The type of {self.__class__} was underspecified, got type {tp}"
            )
        self._tp_cache = tp
        assert self._tp_cache is not None
        return self._tp_cache

    @abc.abstractproperty
    def inputs(self) -> typing.List["Sym[typing.Any]"]:
        raise NotImplementedError()


from argon.ref import Sym
