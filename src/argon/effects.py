import enum
import typing
import pydantic
from pydantic.dataclasses import dataclass

from argon.ref import Sym


class Perhaps(enum.Enum):
    FALSE = 0
    TRUE = 1
    MAYBE = 2


@dataclass
class Effects:
    # Whether each invocation of the op MUST return a new value
    # This is primarily for making mutable/stateful objects such as counters
    unique: Perhaps = Perhaps.FALSE
    idempotent: Perhaps = Perhaps.FALSE
    reads: typing.Set[Sym[typing.Any]] = pydantic.Field(default_factory=set)
    writes: typing.Set[Sym[typing.Any]] = pydantic.Field(default_factory=set)

    @property
    def may_cse(self) -> bool:
        # TODO: Refine this characterization w.r.t. an actual memory model.
        return self.idempotent == Perhaps.TRUE
