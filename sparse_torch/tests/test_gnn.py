from sparse_torch.state import ProgramState
from sparse_torch.extern_mlir.compiler import compile_to_mlir
from sparse_torch.types.tensor import LevelFormat, TensorStorage, TensorFormat, Tensor, Format


def test_dummy_gcn():
    state = ProgramState()
    with state:
        adj1 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
        in1 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
        w1 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
        w2 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
        v = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(512, 512))

        out = adj1 @ in1
        out = w1 @ out
        out = out.relu()
        out = adj1 @ out
        out = w2 @ out
        out = out.relu()
        out = out - out.max_reduce()
        out = out.exp()
        out = out / out.reduce()

    print(state)
    module = state.compile(compiler=compile_to_mlir)
    print(module)