from contextvars import ContextVar
import typing

import pydantic
from pydantic.dataclasses import dataclass
import argon

from argon.ref import Op, Sym
import argon.ref as ref

@dataclass
class State:
    _id: int = -1
    scope: "Scope" = pydantic.Field(default_factory=lambda: Scope())

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def new_scope(self) -> "StateScope":
        return StateScope(state=self, scope=Scope(parent=self.scope))
    
    def stage[R](self, op: Op[R]) -> R:
        return self.register(op, lambda: self._symbol(op.R(), op), lambda sym: None)  # type: ignore
    
    def register[R](self, op: Op[R], symbol: typing.Callable[[], R], flow: typing.Callable[[Sym[R]], None]) -> R:
        lhs = symbol()
        sym = typing.cast(Sym[R], op.R())
        flow(sym)
        return lhs


    def _symbol[A](self, tp: ref.Type[A], op: Op[A]) -> A:
        return tp._new(ref.Def(ref.Node(self.next_id(), op)))

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Scope:
    parent: typing.Optional["Scope"] = None
    scope: typing.List[Sym[typing.Any]] = pydantic.Field(default_factory=list)
    cache: typing.Mapping[Op[typing.Any], Sym[typing.Any]] = pydantic.Field(
        default_factory=dict
    )


@dataclass
class StateScope:
    state: State
    scope: Scope
    prev_scope: typing.Optional[Scope] = None

    def __enter__(self):
        self.prev_scope = self.state.scope
        self.state.scope = self.scope

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.prev_scope is not None
        self.state.scope = self.prev_scope
        self.prev_scope = None


_state: ContextVar[State] = ContextVar("state")
def get_current_state() -> State:
    if current := _state.get(None):
        return current
    newstate = State()
    _state.set(newstate)
    return newstate


def stage[A](op: Op[A]) -> A:
    state = get_current_state()
    return state.stage(op)
