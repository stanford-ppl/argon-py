from argon.types.tensor import Level, TensorStorage, Tensor
from argon.state import State
from argon.wrapper import argon_function
import numpy as np

@argon_function
def test_tensor():
    state = State()
    with state:
        #breakpoint()
        tensor = Tensor[TensorStorage]().const(TensorStorage(np.random((1,1)), None, None))
        #breakpoint()
        print(tensor)