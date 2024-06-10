from sparse_torch.state import ProgramState
from sparse_torch.extern_mlir.compiler import compile_to_mlir
from sparse_torch.types.tensor import LevelFormat, TensorStorage, TensorFormat, Tensor, Format


def test_dummy_attention():
    state = ProgramState()
    with state:
        q = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(512, 64))
        k = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(64, 512))
        v = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(512, 64))
        att = q @ k
        s = att - att.max_reduce()
        exp = s.exp()
        div = exp / exp.reduce()
        qkv = att @ v

    print(state)
    module = state.compile(compiler=compile_to_mlir)
    print(module)