from argon.state import State
from argon.types.tensor import LevelFormat, TensorStorage, TensorFormat, Tensor
from argon.extern_mlir.compiler import process_state


def test_dummy_attention():
    state = State()
    with state:
        q = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("compressed")]), (512, 64)))
        k = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("compressed")]), (64, 512)))
        v = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("compressed")]), (512, 64)))
        d = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("compressed")]), (512, 512)))
        att = q @ k
        s = att - att.max_reduce()
        exp = s.exp()
        div = exp / exp.reduce()
        qkv = att @ v
        # qk = qkv + d

    print(state)
    process_state(state)
