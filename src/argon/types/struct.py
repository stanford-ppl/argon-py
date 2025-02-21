import abc
from typing import override
import typing
from argon.ref import Ref
from argon.srcctx import SrcCtx
from argon.state import stage


MEMBERS_TP = typing.TypeVar("MEMBERS_TP")


class Struct[MEMBERS_TP](Ref[dict, "Struct[MEMBERS_TP]"]):
    """
    The Struct[MEMBERS_TP] class represents a namedtuple in the Argon
    language. The type parameter MEMBERS_TP is a namedtuple with keys
    mapping to the types of the values in the NamedTuple.
    """

    @override
    def fresh(self) -> "Struct[MEMBERS_TP]":  # type: ignore -- Pyright falsely detects MEMBERS_TP as an abstractproperty instead of type variable
        return Struct[self.MEMBERS_TP]()

    # MEMBERS_TP shim is used to silence typing errors -- its actual definition is provided by ArgonMeta
    @abc.abstractproperty
    def MEMBERS_TP(self) -> typing.Type[MEMBERS_TP]:
        raise NotImplementedError()

    # TODO: Only restricting keys to str for now
    def __getitem__(self, key: str) -> MEMBERS_TP:  # type: ignore -- Pyright falsely detects MEMBERS_TP as an abstractproperty instead of type variable
        try:
            item_tp = self.MEMBERS_TP[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in dictionary {self}")

        from argon.node.struct_ops import Get

        return stage(Get[item_tp](self, key), ctx=SrcCtx.new(2))
