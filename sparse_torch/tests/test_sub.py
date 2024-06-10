from sparse_torch.state import ProgramState
from sparse_torch.extern_mlir.compiler import compile_to_mlir
from sparse_torch.types.tensor import LevelFormat, TensorStorage, TensorFormat, Tensor, Format


def test_simple_sub():
    state = ProgramState()
    with state:
        a = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(10, 10))
        b = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(10, 10))
        c = a - b

    print(state)
    module = state.compile(compiler=compile_to_mlir)
    print(module)