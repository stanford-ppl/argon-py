from pydantic.dataclasses import dataclass
from pydantic import field_validator
from typing import override, List, TypeVar, Optional, Tuple, get_args, Any
from sympy import Expr, Number, Integer
from argon.ref import Ref, Const
from argon.srcctx import SrcCtx
from argon.state import stage
from enum import Enum

T = TypeVar("T")


@dataclass
class DynDim:
    dim: Any

    @field_validator("symbol", mode="before")
    def validate_symbol(cls, value):
        if not isinstance(value, Expr) or isinstance(value, Number):
            raise ValueError("Must be a SymPy Symbol")
        return value


@dataclass
class StaticDim:
    dim: Any

    @field_validator("symbol", mode="before")
    def validate_symbol(cls, value):
        if not isinstance(value, Integer):
            raise ValueError("Must be a SymPy Symbol")
        return value


@dataclass
class TensorFormat[T]:
    dims: List[DynDim | StaticDim]
    buff_dim: int
    data_type: type

    def __str__(self) -> str:
        return f"Dims: {str(self.dims)}, Buff Dim: {self.buff_dim}, Data Type: {self.data_type}"

    def get_dims(self) -> List[Expr]:
        return self.dims

    def get_buff_dim(self) -> int:
        return self.buff_dim

    def get_data_type(self) -> type:
        return self.data_type


class Tensor[T](Ref[TensorFormat[T], "Tensor[T]"]):
    @override
    def fresh(self) -> "Tensor[T]":
        return Tensor[self.T]()

    def new(
        dims: List[DynDim | StaticDim], buff_dim: int, data_type: type
    ) -> "Tensor[T]":
        return Tensor[data_type]().const(
            TensorFormat[data_type](dims, buff_dim, data_type)
        )

    def get_dims(self) -> List[DynDim | StaticDim]:
        if type(self.rhs.val) != Const:
            return get_args(self.rhs.val.underlying.T)[0].dims
        return self.rhs.val.value.dims

    def get_buff_dim(self) -> int:
        if type(self.rhs.val) != Const:
            return get_args(self.rhs.val.underlying.T)[0].buff_dim
        return self.rhs.val.value.buff_dim

    def get_data_type(self) -> type:
        if type(self.rhs.val) != Const:
            return get_args(self.rhs.val.underlying.T)[0].data_type
        return self.rhs.val.value.data_type
