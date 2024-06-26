from contextvars import ContextVar
import typing

import pydantic
from pydantic.dataclasses import dataclass
from argon.errors import StagingError

from argon.ref import Exp, Op, Sym
import argon.ref as ref
from argon.srcctx import SrcCtx

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

    def stage[R](self, op: Op[R], ctx: SrcCtx | None = None) -> R:
        ctx = ctx or SrcCtx.new(2)
        return self.register(op, lambda: self._symbol(op.R, op, ctx), lambda sym: None)  # type: ignore

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
        self.scope.symbols.append(sym)

        flow(sym)
        return lhs

    def _symbol[A](self, tp: ref.Type[A], op: Op[A], ctx: SrcCtx) -> A:
        return tp()._new(ref.Def(ref.Node(self.next_id(), op)), ctx)

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

    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        scope_str = "None" if self.scope is None else self.scope.dump(indent_level + 1)
        prev_state_str = (
            "None"
            if self.prev_state is None
            else self.prev_state.dump(indent_level + 1)
        )
        return (
            f"\n"
            f"State( \n"
            f"{indent}scope={scope_str}, \n"
            f"{indent}prev_state={prev_state_str} \n"
            f"{no_indent})"
        )

    def __str__(self) -> str:
        return self.dump()


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True), repr=True)
class Scope:
    parent: typing.Optional["Scope"] = None
    # These need to be explicitly spelled out instead of using Sym[typing.Any] because that breaks Pydantic.
    symbols: typing.List[Exp[typing.Any, typing.Any]] = pydantic.Field(
        default_factory=list
    )
    cache: typing.MutableMapping[Op[typing.Any], Exp[typing.Any, typing.Any]] = (
        pydantic.Field(default_factory=dict)
    )

    def dump(self, indent_level=0) -> str:
        no_indent = "|   " * indent_level
        indent = "|   " * (indent_level + 1)
        more_indent = "|   " * (indent_level + 2)
        parent_str = (
            "None" if self.parent is None else self.parent.dump(indent_level + 1)
        )
        if self.symbols:
            symbols_str = ", \n".join(
                f"{more_indent}{symbol.dump(indent_level + 2)}"
                for symbol in self.symbols
            )
            symbols_str = f"[\n{symbols_str}\n{indent}]"
        else:
            symbols_str = "[]"
        return (
            f"Scope( \n"
            f"{indent}parent={parent_str}, \n"
            f"{indent}symbols={symbols_str}, \n"
            f"{indent}cache={self.cache} \n"
            f"{no_indent})"
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


def stage[A](op: Op[A], ctx: SrcCtx | None = None) -> A:
    state = State.get_current_state()
    ctx = ctx or SrcCtx.new(2)
    return state.stage(op, ctx)
