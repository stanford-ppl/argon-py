import typing
import pydantic
from pydantic.dataclasses import dataclass

import argon


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class ArgonFunction:
    original_func: typing.Callable
    transformed_func: typing.Callable
    return_type: typing.Type | typing.Callable
    param_types: typing.Dict[str, typing.Type | typing.Callable]
    abstract_func: typing.Optional["Function"]
    bound_instance: typing.Any = None  # If set, this is a bound method

    @property
    def argon(self):
        return argon

    def call_original(self, *args, **kwargs):
        if self.bound_instance is not None:
            return self.original_func(self.bound_instance, *args, **kwargs)
        return self.original_func(*args, **kwargs)

    def call_transformed(self, *args, **kwargs):
        if self.bound_instance is not None:
            return self.transformed_func(self.bound_instance, *args, **kwargs, __________argon=self)
        return self.transformed_func(*args, **kwargs, __________argon=self)

    get_function_name = lambda self: self.original_func.__name__

    def get_param_names(self) -> list[str]:
        return list(self.param_types.keys())

    def get_param_type(self, param_name: str) -> typing.Type | typing.Callable:
        return self.param_types[param_name]

    def get_return_type(self) -> typing.Type | typing.Callable:
        return self.return_type

    def bind(self, instance) -> None:
        """Bind this ArgonFunction to the given instance and return self."""
        self.bound_instance = instance
