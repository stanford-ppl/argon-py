from typing import override
import typing
from argon.node.dict_ops import Get
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage


T = typing.TypeVar("T")


class Dictionary[T](Ref[dict, "Dictionary[T]"]):
    """
    The NamedTuple[T] class represents a namedtuple in the Argon
    language. The type parameter T is a namedtuple with keys
    mapping to the types of the values in the NamedTuple.
    """

    @override
    def fresh(self) -> "Dictionary[T]":
        return Dictionary[self.T]()

    # TODO: Only restricting keys to str for now
    def __getitem__(self, key: str) -> T:
        try:
            item_tp = self.T[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in dictionary {self}")
        return stage(Get[item_tp](self, key), ctx=SrcCtx.new(2))
