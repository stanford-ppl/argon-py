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
