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
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>
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

## Example: Graph Convolutional Neural Network
```python
from sparse_torch.types.tensor import * 
from sparse_torch.node import sparse_torch 
from sparse_torch.state import ProgramState

def test_simple_gcn():
    state = ProgramState()
    with state:
        adj1 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
        in1 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
        w1 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
        w2 = Tensor.new(format=[Format.DENSE, Format.COMPRESSED], shape=(512, 512))
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
    print(state.compile())
```
The example above can be executed using pytest:
```
pytest -s sparse_torch/tests/test_gnn.py
```
This example should produce the captured program graph:
```
State( 
|   scope=Scope( 
|   |   parent=None, 
|   |   symbols=[
|   |   |   x0 = Matmul(Const(Format: (DENSE, COMPRESSED); Shape: (512, 512)), Const(Format: (DENSE, COMPRESSED); Shape: (512, 512)))( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/tests/test_gnn.py:14:14 
|   |   |   ), 
|   |   |   x1 = Matmul(Const(Format: (DENSE, COMPRESSED); Shape: (512, 512)), x0)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/tests/test_gnn.py:15:14 
|   |   |   ), 
|   |   |   x2 = ReLU(x1)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/types/tensor.py:109:15 
|   |   |   ), 
|   |   |   x3 = Matmul(Const(Format: (DENSE, COMPRESSED); Shape: (512, 512)), x2)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/tests/test_gnn.py:17:14 
|   |   |   ), 
|   |   |   x4 = Matmul(Const(Format: (DENSE, COMPRESSED); Shape: (512, 512)), x3)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/tests/test_gnn.py:18:14 
|   |   |   ), 
|   |   |   x5 = ReLU(x4)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/types/tensor.py:109:15 
|   |   |   ), 
|   |   |   x6 = MaxReduce(x5, -1)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/types/tensor.py:121:15 
|   |   |   ), 
|   |   |   x7 = Sub(x5, x6)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/tests/test_gnn.py:20:14 
|   |   |   ), 
|   |   |   x8 = Exp(x7)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/types/tensor.py:115:15 
|   |   |   ), 
|   |   |   x9 = Reduce(x8, -1)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/types/tensor.py:118:15 
|   |   |   ), 
|   |   |   x10 = Div(x8, x9)( 
|   |   |   |   tp: Tensor 
|   |   |   |   ctx: /home/rubensl/Documents/repos/argon-py/sparse_torch/tests/test_gnn.py:22:14 
|   |   |   )
|   |   ], 
|   |   cache={} 
|   ), 
|   prev_state=None 
)
```
It should also produce the generated MLIR IR representation:
```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
module {
  func.func @generated_func() -> tensor<512x512xf32, #sparse> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<512x512xf32, #sparse>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %2 = tensor.empty() : tensor<512x512xf32, #sparse>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %4 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %1 : tensor<512x512xf32, #sparse>, tensor<512x512xf32, #sparse>) outs(%3 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %5 = tensor.empty() : tensor<512x512xf32, #sparse>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %7 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %4 : tensor<512x512xf32, #sparse>, tensor<512x512xf32, #sparse>) outs(%6 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %8 = tensor.empty() : tensor<512x512xf32, #sparse>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<512x512xf32, #sparse>) outs(%9 : tensor<512x512xf32, #sparse>) {
    ^bb0(%in: f32, %out: f32):
      %37 = arith.cmpf ugt, %in, %cst : f32
      %38 = arith.select %37, %in, %cst : f32
      linalg.yield %38 : f32
    } -> tensor<512x512xf32, #sparse>
    %11 = tensor.empty() : tensor<512x512xf32, #sparse>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %13 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %10 : tensor<512x512xf32, #sparse>, tensor<512x512xf32, #sparse>) outs(%12 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %14 = tensor.empty() : tensor<512x512xf32, #sparse>
    %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %16 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1, %13 : tensor<512x512xf32, #sparse>, tensor<512x512xf32, #sparse>) outs(%15 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %17 = tensor.empty() : tensor<512x512xf32, #sparse>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<512x512xf32, #sparse>) outs(%18 : tensor<512x512xf32, #sparse>) {
    ^bb0(%in: f32, %out: f32):
      %37 = arith.cmpf ugt, %in, %cst : f32
      %38 = arith.select %37, %in, %cst : f32
      linalg.yield %38 : f32
    } -> tensor<512x512xf32, #sparse>
    %20 = tensor.empty() : tensor<512x1xf32, #sparse>
    %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<512x1xf32, #sparse>) -> tensor<512x1xf32, #sparse>
    %22 = tensor.empty() : tensor<512x1xi64>
    %23 = linalg.fill ins(%c0_i64 : i64) outs(%22 : tensor<512x1xi64>) -> tensor<512x1xi64>
    %24:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%19 : tensor<512x512xf32, #sparse>) outs(%21, %23 : tensor<512x1xf32, #sparse>, tensor<512x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %37 = linalg.index 1 : index
      %38 = arith.index_cast %37 : index to i64
      %39 = arith.maximumf %in, %out : f32
      %40 = arith.cmpf ogt, %in, %out : f32
      %41 = arith.select %40, %38, %out_1 : i64
      linalg.yield %39, %41 : f32, i64
    } -> (tensor<512x1xf32, #sparse>, tensor<512x1xi64>)
    %25 = tensor.empty() : tensor<512x512xf32, #sparse>
    %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %27 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %24#0 : tensor<512x512xf32, #sparse>, tensor<512x1xf32, #sparse>) outs(%26 : tensor<512x512xf32, #sparse>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %37 = arith.subf %in, %in_1 : f32
      linalg.yield %37 : f32
    } -> tensor<512x512xf32, #sparse>
    %28 = tensor.empty() : tensor<512x512xf32, #sparse>
    %29 = linalg.fill ins(%cst : f32) outs(%28 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %30 = linalg.exp ins(%27 : tensor<512x512xf32, #sparse>) outs(%29 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %31 = tensor.empty() : tensor<512x1xf32, #sparse>
    %32 = linalg.fill ins(%cst : f32) outs(%31 : tensor<512x1xf32, #sparse>) -> tensor<512x1xf32, #sparse>
    %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%30 : tensor<512x512xf32, #sparse>) outs(%32 : tensor<512x1xf32, #sparse>) {
    ^bb0(%in: f32, %out: f32):
      %37 = arith.addf %in, %out : f32
      linalg.yield %37 : f32
    } -> tensor<512x1xf32, #sparse>
    %34 = tensor.empty() : tensor<512x512xf32, #sparse>
    %35 = linalg.fill ins(%cst : f32) outs(%34 : tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    %36 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%30, %33 : tensor<512x512xf32, #sparse>, tensor<512x1xf32, #sparse>) outs(%35 : tensor<512x512xf32, #sparse>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %37 = arith.divf %in, %in_1 : f32
      linalg.yield %37 : f32
    } -> tensor<512x512xf32, #sparse>
    return %36 : tensor<512x512xf32, #sparse>
  }
}
```
