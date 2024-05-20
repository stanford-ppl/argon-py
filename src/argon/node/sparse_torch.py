import typing
import pydantic
from argon.state import State
from argon.ref import Exp, Op, Sym
# from argon.extern_mlir.compiler import process_state

# from argon.types.integer import Integer
from pydantic.dataclasses import dataclass


T = typing.TypeVar("T", bound=Exp[typing.Any, typing.Any], covariant=True)


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Unary[T](Op[T]):
    a: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a]
    

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Binary[T](Op[T]):
    a: T
    b: T

    @property
    @typing.override
    def inputs(self) -> typing.List[Sym[typing.Any]]:
        return [self.a, self.b]  # type: ignore


@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Add[T](Binary):
    pass

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Mul[T](Binary):
    pass

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Matmul[T](Binary):
    pass

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Div[T](Binary):
    pass
    
@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Sub[T](Binary):
    pass
    
@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Softmax[T](Unary):
    pass
    
@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class ReLU[T](Unary):
    pass

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Exp[T](Unary):
    pass

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class Reduce[T](Unary):
    pass

@dataclass(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class MaxReduce[T](Unary):
    pass

def compile(state: State):
    process_state(state)