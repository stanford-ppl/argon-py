from pydantic.dataclasses import dataclass
from typing import override, List, TypeVar, Optional, Tuple, get_args
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

    def __matmul__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m, k0) = self.get_shape()
        (k1, n) = other.get_shape()
        assert k0 == k1, f"Invalid shapes for matrix multiplication with expected signature: (n,k),(k,m)->(n,m) -- actual shape: ({
            m}, {k0}), ({k1}, {n})"
        return stage(sparse_torch.Matmul[Tensor[T := TensorStorage(tensor_format=TensorFormat([LevelFormat("compressed"), LevelFormat("compressed")]), shape=(self.get_shape()[0], other.get_shape()[1]))]](self, other), ctx=SrcCtx.new(2))

    def __add__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.get_shape()
        (m1, n1) = other.get_shape()
        assert m0 == m1 and n0 == n1, f"Invalid shapes for element-wise addition with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Add[Tensor[T := TensorStorage(tensor_format=TensorFormat([LevelFormat("compressed"), LevelFormat("compressed")]), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))

    def __sub__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.get_shape()
        (m1, n1) = other.get_shape()
        assert m0 == m1 and n0 == n1, f"Invalid shapes for element-wise addition with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Sub[Tensor[T := TensorStorage(tensor_format=TensorFormat([LevelFormat("compressed"), LevelFormat("compressed")]), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))

    def __mul__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.get_shape()
        (m1, n1) = other.get_shape()
        assert m0 == m1 and n0 == n1, f"Invalid shapes for element-wise multiplication with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Mul[Tensor[T := TensorStorage(tensor_format=TensorFormat([LevelFormat("compressed"), LevelFormat("compressed")]), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))

    def __truediv__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.get_shape()
        (m1, n1) = other.get_shape()
        assert m0 == m1 and n0 == n1, f"Invalid shapes for element-wise addition with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Div[Tensor[T := TensorStorage(tensor_format=TensorFormat([LevelFormat("compressed"), LevelFormat("compressed")]), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))
    
    def matmul(self, other: "Tensor[T]") -> "Tensor[T]":
        return self @ other

    def add(self, other: "Tensor[T]") -> "Tensor[T]":
        return self + other

    def sub(self, other: "Tensor[T]") -> "Tensor[T]":
        return self - other

    def mul(self, other: "Tensor[T]") -> "Tensor[T]":
        return self * other
    
    def relu(self):
        return stage(sparse_torch.ReLU[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self))

    def softmax(self):
        return stage(sparse_torch.Softmax[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self))
    
    def exp(self):
        return stage(sparse_torch.Exp[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self))
    
    def reduce(self):
        return stage(sparse_torch.Reduce[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self))

    def max_reduce(self):
        return stage(sparse_torch.MaxReduce[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self))
    
    def get_shape(self) -> Tuple[int, ...]:
        if type(self.rhs.val) != Const:
            return get_args(self.rhs.val.underlying.T)[0].shape
        return self.rhs.val.value.shape

    def get_format(self) -> TensorFormat:
        if type(self.rhs.val) != Const:
            return get_args(self.rhs.val.underlying.T)[0].tensor_format
        return self.rhs.val.value.tensor_format
