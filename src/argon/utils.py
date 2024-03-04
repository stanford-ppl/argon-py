import collections
import typing


def resolve_class(cls, resolutions) -> typing.Type:
    match cls:
        case typing.ForwardRef():
            if result := cls._evaluate(globals(), resolutions, frozenset()):
                return typing.cast(typing.Type, result)
            raise TypeError(
                f"Failed to resolve forward reference: {cls} with resolutions {resolutions}"
            )
        case typing.TypeVar() if cls.__name__ in resolutions:
            return resolve_class(resolutions[cls.__name__], resolutions)
        case _:
            return cls


def _type_helper(
    cls: typing.Type,
    target: typing.Type,
    resolutions: typing.Dict[str, typing.Type],
    options: typing.MutableMapping[str, typing.Set[typing.Type]],
):
    for base in cls.__orig_bases__:
        origin = typing.get_origin(base) or base
        match origin:
            case _ if origin is target:
                args = [
                    resolve_class(klass, resolutions) for klass in typing.get_args(base)
                ]
                names = [tparam.__name__ for tparam in origin.__type_params__]
                for name, arg in zip(names, args):
                    options[name].add(arg)

            case _ if not issubclass(origin, target):
                continue

            # The base is generic, so we should dissect it a bit.
            case _:
                typenames = (tparam.__name__ for tparam in origin.__type_params__)
                new_resolutions = resolutions | dict(
                    zip(typenames, typing.get_args(base))
                )
                _type_helper(origin, target, new_resolutions, options)


def _unify_types(
    name: str,
    options: typing.Set[typing.Type],
    covariant: bool = False,
    contravariant: bool = False,
) -> typing.Type:
    if not covariant and not contravariant:
        if len(options) != 1:
            raise TypeError(
                f"Failed to unify type parameter {name}: found options {options}"
            )
        return options.pop()

    raise NotImplementedError("Have not implemented proper type unification yet.")


def compute_types(
    cls: typing.Type, target: typing.Type, resolutions: typing.Dict[str, typing.Type]
) -> typing.Mapping[str, typing.Type]:
    options = collections.defaultdict(set)
    _type_helper(cls, target, resolutions, options)
    return {k: _unify_types(k, v, False, False) for k, v in options.items()}
