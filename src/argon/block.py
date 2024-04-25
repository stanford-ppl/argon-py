import typing
import pydantic
from pydantic import dataclasses

from argon.base import ArgonMeta
from argon.ref import Exp


@dataclasses.dataclass
class Block[B](ArgonMeta):
    inputs: typing.List[Exp[typing.Any, typing.Any]] = pydantic.Field(
        default_factory=list
    )
    stms: typing.List[Exp[typing.Any, typing.Any]] = pydantic.Field(
        default_factory=list
    )
    result: typing.Optional[Exp[typing.Any, B]] = None
