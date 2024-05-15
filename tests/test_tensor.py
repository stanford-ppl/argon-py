from argon.state import State
from argon.types.tensor import LevelFormat, TensorStorage, TensorFormat, Tensor
from argon.extern_mlir.compiler import process_state


def test_matmul_softmax():
    state = State()
    with state:
        q = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("compressed")]), (512, 64)))
        k = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("compressed")]), (64, 512)))
        v = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("compressed")]), (512, 64)))
        c = q @ k 
        # d = c.relu()
        # e = (d - d.max_reduce()).exp()
        # f = e / e.reduce()
        # g = f @ v

        # print(d.get_shape())
        # print(c.tensor_format())
    print(state)
    process_state(state)

