from argon.state import State
from argon.wrapper import argon_function
from argon.types.tensor import LevelFormat, TensorStorage, TensorFormat, Tensor
# from compiler import dump_mlir


def test_tensor():
    state = State()
    with state:
        a = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("compressed"), LevelFormat("dense")]), (1, 2)))
        b = Tensor().const(TensorStorage(TensorFormat(
            [LevelFormat("dense"), LevelFormat("dense")]), (2, 2)))
        c = a * b + a * b + a
        print(c.T)
        # print(c.tensor_format())
    print(state)
