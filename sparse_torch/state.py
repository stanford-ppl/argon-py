from argon.state import State
from sparse_torch.extern_mlir.compiler import compile_to_mlir

class ProgramState(State):
    def __init__(self):
        super().__init__()
    def compile(self, compiler=compile_to_mlir):
        return compiler(self)
