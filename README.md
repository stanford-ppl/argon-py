# SparseTorch: A DSL for Sparse Machine Learning
SparseTorch is a domain-specific language (DSL) that simplifies the development of sparse machine learning applications. This tutorial demonstrates how to create sparse tensors and perform basic operations like addition using the SparseTorch DSL.
## Installation
1. In the root directory, run:
```shell
python -m pip install .
```
2. If you are getting ModuleNotFoundError for `argon`, then add this to the `sys.path` directory list. (This should probably be done using a virtual environment.)

```shell
# On MacOSX
vim ~/.bash_profile
# Then, add: 
#      export PYTHONPATH="<root-directory>/argon-py/src"
source ~/.bash_profile
```

## Core Components
1. **Tensor Class** 
	* Represents a sparse tensor
	* Key attributes:
		* *format*: Specifies the per-dimension level format of the tensor (e.g., Format.COMPRESSED)
		* *shape*: Defines the dimensions of the tensor
2. **Node Operations**
	* Provide a variety of sparse operations (e.g. add, matmul, div, etc...).
	* Operations are applied to the Tensor objects
3. **Program State**
	* Manages the execution of operations and the state of tensors
	* Operations are typically defined within a *with ProgramState()* block
4. **Compile**
	* After capturing program state, user can compile program to MLIR IR using *compile* function on the program state

## Example: Sparse Matrix Addition
```python
from sparse_torch.types.tensor import * 
from sparse_torch.node import sparse_torch 
from sparse_torch.state import ProgramState 

def test_simple_add():
	state = ProgramState() 
	with state: 
		a = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(10, 10)) 
		b = Tensor.new(format=[Format.COMPRESSED, Format.COMPRESSED], shape=(10, 10)) 
		c = a + b 

	print(state)
	print(state.compile())
```
The example above can be executed using pytest:
```
pytest -s sparse_torch/tests/test_add.py
```
This example should produce the captured program graph for this add program:
```
State( 
|   scope=Scope( 
|   |   parent=None, 
|   |   symbols=[
|   |   |   x0 = Add(Const(Format: (COMPRESSED, COMPRESSED); Shape: (10, 10)), Const(Format: (COMPRESSED, COMPRESSED); Shape: (10, 10)))( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/tests/test_add.py:11:12 
|   |   |   )
|   |   ], 
|   |   cache={} 
|   ), 
|   prev_state=None 
)
```
It should also produce the generated MLIR IR representation for this test:
```
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : dense) }>
module {
  func.func @generated_func() -> tensor<10x10xf32, #sparse> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<10x10xf32, #sparse>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x10xf32, #sparse>) -> tensor<10x10xf32, #sparse>
    %2 = tensor.empty() : tensor<10x10xf32, #sparse>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<10x10xf32, #sparse>) -> tensor<10x10xf32, #sparse>
    %4 = linalg.add ins(%1, %1 : tensor<10x10xf32, #sparse>, tensor<10x10xf32, #sparse>) outs(%3 : tensor<10x10xf32, #sparse>) -> tensor<10x10xf32, #sparse>
    return %4 : tensor<10x10xf32, #sparse>
  }
}
```
### Explanation
1.  **Import Statements:** Import the necessary SparseTorch modules.
2.  **Program State:** Create a `ProgramState` object.
3.  **Tensor Creation:**
    -   Create two sparse tensors `a` and `b` using `Tensor.new`.
    -   Specify the compressed format (`Format.COMPRESSED`) for both dimensions and a shape of (10, 10).
4.  **Sparse Addition:**
    -   Use the `+` operator to perform element-wise addition on tensors `a` and `b`.
    -   The result is stored in a new tensor `c`.
5.  **Printing Program State:**
    -   The `print(state)` statement showcases the computation graph required to compute tensor `c` from its components. It does not print the numerical values of the tensors.
 6.  **Compiling to MLIR:**
	 * The `print(state.compile())` statement compiles the captured program graph to MLIR.
