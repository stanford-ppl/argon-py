import abc
import typing
from argon.utils import resolve_class


def _compute_optype_helper(
    cls: typing.Type,
    resolutions: typing.Dict[str, typing.Type],
    options: typing.Set[typing.Type],
):
    for base in cls.__orig_bases__:
        origin = typing.get_origin(base) or base
        match origin:
            case _ if origin is Op:
                options.add(resolve_class(typing.get_args(base)[0], resolutions))
            case _ if not issubclass(origin, ExpType):
                continue

            # The base is generic, so we should dissect it a bit.
            case _:
                typenames = (tparam.__name__ for tparam in origin.__type_params__)
                new_resolutions = resolutions | dict(
                    zip(typenames, typing.get_args(base))
                )
                _compute_optype_helper(origin, new_resolutions, options)


def _compute_optype(
    cls: typing.Type, resolutions: typing.Dict[str, typing.Type]
) -> typing.Type:
    options = set()
    _compute_optype_helper(cls, resolutions, options)
    if len(options) != 1:
        raise TypeError(f"Failed to unify R types on {cls}: {options}")
    return options.pop()


class Op[A](abc.ABC):
    _tp_cache: typing.ClassVar[typing.Optional[typing.Type]] = None

    @classmethod
    def R(cls) -> typing.Type[A]:
        if cls._tp_cache:
            return cls._tp_cache
        cls._tp_cache = _compute_optype(cls, {cls.__name__: cls})
        assert cls._tp_cache is not None
        return cls._tp_cache

    @abc.abstractproperty
    def inputs(self) -> typing.List["Sym[typing.Any]"]:
        raise NotImplementedError()


from argon.ref import ExpType, Sym
