from contextvars import ContextVar
import typing

import pydantic
from pydantic.dataclasses import dataclass
from argon.errors import StagingError

from argon.ref import Op, Sym
import argon.ref as ref

_state: ContextVar[typing.Optional["State"]] = ContextVar("state", default=None)
_sentinel = object()


@dataclass
class State:
    _id: int = -1
    scope: "Scope" = pydantic.Field(default_factory=lambda: Scope())

    @staticmethod
    def get_current_state() -> "State":
        cur_state = _state.get(None)
        if cur_state is None:
            raise StagingError("Attempting to get the current state, but it was None")
        return cur_state

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def new_scope(self) -> "ScopeContext":
        return ScopeContext(state=self, scope=Scope(parent=self.scope))

    def stage[R](self, op: Op[R]) -> R:
        return self.register(op, lambda: self._symbol(op.R(), op), lambda sym: None)  # type: ignore

    def register[
        R
    ](
        self,
        op: Op[R],
        symbol: typing.Callable[[], R],
        flow: typing.Callable[[Sym[R]], None],
    ) -> R:
        lhs = symbol()
        sym = typing.cast(Sym[R], lhs)
        self.scope.symbols.add(sym)

        flow(sym)
        return lhs

    def _symbol[A](self, tp: ref.Type[A], op: Op[A]) -> A:
        return tp._new(ref.Def(ref.Node(self.next_id(), op)))

    prev_state: typing.Optional["State"] = None

    # Code to support using State as a context manager.
    def __enter__(self) -> "State":
        previous = _state.get(_sentinel)
        if previous is not _sentinel:
            self.prev_state = previous  # type: ignore
        _state.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prev_state is not _sentinel:
            _state.set(self.prev_state)
        self.prev_state = None


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True), repr=True)
class Scope:
    parent: typing.Optional["Scope"] = None
    symbols: typing.Set[Sym[typing.Any]] = pydantic.Field(default_factory=set)
    cache: typing.Mapping[Op[typing.Any], Sym[typing.Any]] = pydantic.Field(
        default_factory=dict
    )


@dataclass
class ScopeContext:
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


def stage[A](op: Op[A]) -> A:
    state = State.get_current_state()
    return state.stage(op)
