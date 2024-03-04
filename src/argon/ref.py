import pydantic
from pydantic.dataclasses import dataclass

import abc
import typing

C = typing.TypeVar("C")
A = typing.TypeVar("A")
C_co = typing.TypeVar("C_co")
A_co = typing.TypeVar("A_co")


def _resolve_class(cls, resolutions) -> typing.Type:
    match cls:
        case typing.ForwardRef():
            if result := cls._evaluate(globals(), resolutions, frozenset()):
                return typing.cast(typing.Type, result)
            raise TypeError(
                f"Failed to resolve forward reference: {cls} with resolutions {resolutions}"
            )
        case typing.TypeVar() if cls.__name__ in resolutions:
            return _resolve_class(resolutions[cls.__name__], resolutions)
        case _:
            return cls


def _compute_exptype_helper(
    cls: typing.Type,
    resolutions: typing.Dict[str, typing.Type],
    l_options: typing.Set[typing.Type],
    r_options: typing.Set[typing.Type],
):
    for base in cls.__orig_bases__:
        origin = typing.get_origin(base) or base
        match origin:
            case _ if origin is ExpType:
                # Since we've reached ExpType, then we should be able to resolve all the params.
                [lopt, ropt] = [
                    _resolve_class(klass, resolutions)
                    for klass in typing.get_args(base)
                ]
                l_options.add(lopt)
                r_options.add(ropt)

            case _ if not issubclass(origin, ExpType):
                continue

            # The base is generic, so we should dissect it a bit.
            case _:
                typenames = (tparam.__name__ for tparam in origin.__type_params__)
                new_resolutions = resolutions | dict(
                    zip(typenames, typing.get_args(base))
                )
                _compute_exptype_helper(origin, new_resolutions, l_options, r_options)


def _compute_exptype(
    cls: typing.Type, resolutions: typing.Dict[str, typing.Type]
) -> typing.Tuple[typing.Type, typing.Type]:
    l_options = set()
    r_options = set()
    _compute_exptype_helper(cls, resolutions, l_options, r_options)
    if len(r_options) != 1:
        raise TypeError(f"Failed to unify R types on {cls}: {r_options}")
    if len(l_options) != 1:
        raise TypeError(f"Failed to unify L types on {cls}: {l_options}")
    return l_options.pop(), r_options.pop()


class ExpType[C_co, A](abc.ABC):
    _cached_lr: typing.ClassVar[
        typing.Optional[typing.Tuple[typing.Type, typing.Type]]
    ] = None

    @classmethod
    def _compute_LR(cls):
        if cls._cached_lr:
            return
        cls._cached_lr = _compute_exptype(cls, {cls.__name__: cls})

    @classmethod
    def L(cls) -> typing.Type[C_co]:
        cls._compute_LR()
        assert cls._cached_lr is not None
        return cls._cached_lr[0]

    @classmethod
    def R(cls) -> typing.Type[A]:
        cls._compute_LR()
        assert cls._cached_lr is not None
        return cls._cached_lr[1]

    @classmethod
    @abc.abstractmethod
    def fresh(cls) -> A:
        raise NotImplementedError()

    @classmethod
    def _new(cls, d: "Def[C_co, A]") -> A:
        right_type = cls.R()
        assert issubclass(right_type, Ref)
        empty_val = right_type.fresh()
        empty_val.rhs = d
        return empty_val

    @classmethod
    def const(cls, c: C_co) -> A:
        return cls._new(Def(Const(c)))


@dataclass
class Bound[A]:
    id: int
    def_type: typing.Literal["Bound"] = "Bound"


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Node[A]:
    id: int
    underlying: "Op[A]"
    def_type: typing.Literal["Node"] = "Node"


@dataclass
class Const[C_co]:
    value: C_co
    def_type: typing.Literal["Const"] = "Const"


@dataclass
class Def[C, A]:
    val: typing.Union[Const[C], Bound[A], Node[A]] = pydantic.Field(
        discriminator="def_type"
    )


class Exp[C_co, A_co](abc.ABC):
    """Exp[C, A] defines an expression with denotational type C, and staged type A."""

    rhs: typing.Optional[Def[C_co, A_co]] = None

    @abc.abstractproperty
    def tp(self) -> ExpType[C_co, A_co]:
        raise NotImplementedError()


class Ref[C_co, A_co](ExpType[C_co, A_co], Exp[C_co, A_co]):

    @property
    @typing.override
    def tp(self) -> ExpType[C_co, A_co]:
        return self


S_co = typing.TypeVar("S_co", covariant=True)
type Sym[S_co] = Exp[typing.Any, S_co]

type Type[S] = ExpType[typing.Any, S]


def _compute_optype_helper(
    cls: typing.Type,
    resolutions: typing.Dict[str, typing.Type],
    options: typing.Set[typing.Type],
):
    for base in cls.__orig_bases__:
        origin = typing.get_origin(base) or base
        match origin:
            case _ if origin is Op:
                options.add(_resolve_class(typing.get_args(base)[0], resolutions))
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
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        raise NotImplementedError()
