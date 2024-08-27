from pydantic.dataclasses import dataclass
from typing import override, List, TypeVar, Optional, Tuple, get_args
from sympy import Expr
from argon.ref import Ref, Const
from argon.srcctx import SrcCtx
from argon.state import stage
from enum import Enum

T = TypeVar("T")


@dataclass
class TensorFormat[T]:
    dims: List[Expr]
    buff_dim: int
    data_type: T

    def __str__(self) -> str:
        return str(self.dims) + "; buff_dim: " + str(self.buff_dim)

    def get_dims(self) -> List[Expr]:
        return self.dims

    def get_buff_dim(self) -> int:
        return self.buff_dim

    def get_data_type(self) -> T:
        return self.data_type


class Tensor[T](Ref[TensorFormat[T], "Tensor[T]"]):

    @override
    def fresh(self) -> "Tensor[T]":
        return Tensor[self.T]()

    def new(self, dims: List[Expr], buff_dim: int, data_type: T) -> "Tensor[T]":
        return Tensor().const(TensorFormat(dims, buff_dim, data_type))
