from argon.state import State
from step.types.tensor import TensorFormat, Tensor, DynDim, StaticDim
from typing import override, List, TypeVar
from sympy import Expr, Symbol, Integer
from argon.ref import Ref, Const
from argon.srcctx import SrcCtx
from argon.state import stage
import step as torch


def test_tensor():
    softmax = torch.nn.Softmax(dim=1)
    state = State()
    with state:
        B = DynDim(Symbol("B"))
        N = DynDim(Symbol("N"))
        E = DynDim(Integer(64))
        a = Tensor.new(dims=[B, N, E], buff_dim=0, data_type=float)
        print("class names!!!!!!!!")
        print(a.__class__.__name__)
        print(a.C)
        print(a.A)
        b = softmax(a)
    print(state)
