from pydantic.dataclasses import dataclass
from typing import override, List, TypeVar, Optional, Tuple, get_args
from argon.ref import Ref, Const
from argon.srcctx import SrcCtx
from argon.state import stage
from argon.node import sparse_torch
from enum import Enum

T = TypeVar("T")


class Format(Enum):
    DENSE = 0
    COMPRESSED = 1

    # def __str__(self) -> str:
        # match 


@dataclass(frozen=True)
class LevelFormat:
    level_format: Optional[str] = None
    # coords: Optional[List[int]] = None

    def __str__(self) -> str:
        # return f"(Format: {self.level_format}, Coords: {str(self.coords)})"
        return self.level_format


@dataclass(frozen=True)
class TensorFormat:
    level_formats: Optional[List[LevelFormat]]

    def __str__(self) -> str:
        return "Format: (" + ", ".join([str(form) for form in self.level_formats]) + ")"
    
    def format(self):
        return self.level_formats


@dataclass
class TensorStorage:
    # value: Optional[List[Any]] = None
    tensor_format: Optional[TensorFormat] = None
    shape: Optional[Tuple[int, ...]] = None

    def __str__(self) -> str:
        return str(self.tensor_format) + "; Shape: " + str(self.shape)

    def shape(self):
        return self.shape
    
    def format(self):
        return self.tensor_format


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
        assert (m0 == m1 and n0 == n1) or (n0 == 1 or n1 == 1), f"Invalid shapes for element-wise add with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Add[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))

    def __sub__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.get_shape()
        (m1, n1) = other.get_shape()
        assert (m0 == m1 and n0 == n1) or (n0 == 1 or n1 == 1), f"Invalid shapes for element-wise sub with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Sub[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))

    def __mul__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.get_shape()
        (m1, n1) = other.get_shape()
        assert m0 == m1 and n0 == n1, f"Invalid shapes for element-wise multiplication with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Mul[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))

    def __truediv__(self, other: "Tensor[T]") -> "Tensor[T]":
        (m0, n0) = self.get_shape()
        (m1, n1) = other.get_shape()
        assert (m0 == m1 and n0 == n1) or (n0 == 1 or n1 == 1), f"Invalid shapes for element-wise division with expected signature: (m,n),(m,n)->(m,n) -- actual shape: ({m0}, {n0}), ({m1}, {n1})"
        return stage(sparse_torch.Div[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape())]](self, other), ctx=SrcCtx.new(2))
    
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
        return stage(sparse_torch.Reduce[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape()[:1]+(1,))]](self))

    def max_reduce(self):
        return stage(sparse_torch.MaxReduce[Tensor[T := TensorStorage(tensor_format=self.get_format(), shape=self.get_shape()[:1]+(1,))]](self))
    
    def get_shape(self) -> Tuple[int, ...]:
        if type(self.rhs.val) != Const:
            return get_args(self.rhs.val.underlying.T)[0].shape
        return self.rhs.val.value.shape

    def get_format(self) -> TensorFormat:
        if type(self.rhs.val) != Const:
            return get_args(self.rhs.val.underlying.T)[0].tensor_format
        return self.rhs.val.value.tensor_format
    
    def __hash__(self):
        return hash(tuple(self.get_format().format()))
