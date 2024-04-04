import typing
from pydantic import dataclasses

from argon.base import ArgonMeta
from argon.ref import Sym


@dataclasses.dataclass
class Block[B](ArgonMeta):
    inputs: typing.List[Sym[typing.Any]]
    stms: typing.List[Sym[typing.Any]]
    result: Sym[B]
