from pydantic.dataclasses import dataclass
from typing import Union, override, List, Any, TypeVar, Optional, Tuple, get_origin
from argon.ref import Ref, Const
from argon.srcctx import SrcCtx
from argon.state import stage
import argon.node.sparse_torch as sparse_torch

T = TypeVar("T")


@dataclass
class LevelFormat:
    level_format: Optional[str] = None
    # coords: Optional[List[int]] = None

    def __str__(self) -> str:
        # return f"(Format: {self.level_format}, Coords: {str(self.coords)})"
        return f"Format: {self.level_format}"


@dataclass
class TensorFormat:
    level_formats: Optional[List[LevelFormat]]

    def __str__(self) -> str:
        return ", ".join([str(form) for form in self.level_formats])


@dataclass
class TensorStorage:
    # value: Optional[List[Any]] = None
    tensor_format: Optional[TensorFormat] = None
    shape: Optional[Tuple[int, ...]] = None

    def __str__(self) -> str:
        return str(self.tensor_format)

    def shape(self):
        return self.shape


class Tensor[T](Ref[TensorStorage, "Tensor[T]"]):

    @override
    def fresh(self) -> "Tensor[T]":
        return Tensor[self.T]()

    def __mul__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m, k0) = self.shape()
        (k1, n) = other.shape()
        assert k0 == k1, f"Invalid shapes for matrix multiplication with expected signature: (n,k),(k,m)->(n,m) -- actual shape: ({
            m}, {k0}), ({k1}, {n})"
        # print(sparse_torch.Mul[Tensor[self.T]](self, other))
        return stage(sparse_torch.Mul[Tensor[T := TensorStorage(tensor_format=TensorFormat([LevelFormat("dense")]), shape=(self.shape()[0], other.shape()[1]))]](self, other), ctx=SrcCtx.new(2))

    def __add__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.shape()
        (m1, n1) = other.shape()
        assert m0 == m1 and n0 == n1, f"Invalid shapes for matrix addition with expected signature: (m,n),(m,n)->(m,n)"
        return stage(sparse_torch.Add[Tensor[self.T]](self, other), ctx=SrcCtx.new(2))

    def shape(self):
        if type(self.rhs.val) != Const:
            print(self.rhs.val.T)
            type_args = self.rhs.val.underlying.R.T()
            print("Type args: ", type_args)
            # tensor_storage_type = type(type_args)
            # print(tensor_storage_type)
            # shape_origin = get_origin(tensor_storage_type.__dict__["shape"])
            raise TypeError("Not a const object")
        return self.rhs.val.value.shape

    def format(self):
        return self.rhs.val.value.tensor_format
