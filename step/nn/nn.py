from step.types.tensor import Tensor, TensorFormat
from step.node.mlop import SoftmaxOp
from argon.srcctx import SrcCtx
from argon.state import stage
from typing import TypeVar, Any
from argon.ref import Exp

T = TypeVar("T", bound=Exp[Any, Any], covariant=True)


class Softmax:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x: Tensor[T]) -> Tensor[T]:
        # (m, k0) = self.get_shape()
        # (k1, n) = other.get_shape()
        # assert k0 == k1, f"Invalid shapes for matrix multiplication with expected signature: (n,k),(k,m)->(n,m) -- actual shape: ({
        #     m}, {k0}), ({k1}, {n})"
        return stage(
            SoftmaxOp[
                Tensor[
                    T := TensorFormat(
                        dims=x.get_dims(),
                        buff_dim=x.get_buff_dim(),
                        data_type=x.get_data_type(),
                    )
                ]
            ](x, self.dim),
            ctx=SrcCtx.new(2),
        )
