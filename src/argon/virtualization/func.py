import typing
from pydantic.dataclasses import dataclass

import argon


@dataclass
class ArgonFunction:
    original_func: typing.Callable
    transformed_func: typing.Callable

    @property
    def argon(self):
        return argon

    def call_original(self, *args, **kwargs):
        return self.original_func(*args, **kwargs)

    def call_transformed(self, *args, **kwargs):
        return self.transformed_func(*args, **kwargs, __________argon=self)
