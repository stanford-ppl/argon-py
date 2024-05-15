import ctypes
import numpy as np
import os
import sys

from mlir import ir
from mlir import runtime as rt

from mlir.dialects import sparse_tensor as st
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects.linalg.opdsl import lang as dsl
from mlir.dialects.linalg.opdsl.ops.core_named_ops import *
from mlir.dialects._ods_common import *
from mlir.dialects import tensor as t
from mlir.dialects import arith
from mlir.ir import *

from argon.node import sparse_torch
from argon.ref import Exp, Op, Sym, Const
from argon.types.tensor import Tensor, TensorFormat, LevelFormat
from argon.state import State

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)


# TODO: Not needed for matmul anymore but might need it for einsums
@dsl.linalg_structured_op
def matmul_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.N),
    C=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N, output=True),
):
    C[dsl.D.m, dsl.D.n] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k, dsl.D.n]


def build_matmul(args: List[ir.RankedTensorType]):
    """Build SpMM kernel.

    This method generates a linalg op with for matrix multiplication using
    just the Python API. Effectively, a generic linalg op is constructed
    that computes C(i,j) += A(i,k) * B(k,j) for annotated matrix A.
    """
    return matmul(args[0], args[1], outs=[args[2]])


def build_div(args: List[ir.RankedTensorType]):
    """Build SpMM kernel.

    This method generates a linalg op with for matrix multiplication using
    just the Python API. Effectively, a generic linalg op is constructed
    that computes C(i,j) += A(i,k) * B(k,j) for annotated matrix A.
    """
    return div(args[0], args[1], outs=[args[2]])


def build_add(args: List[ir.RankedTensorType]):
    """Build SpMM kernel.

    This method generates a linalg op with for matrix multiplication using
    just the Python API. Effectively, a generic linalg op is constructed
    that computes C(i,j) += A(i,k) * B(k,j) for annotated matrix A.
    """
    return add(args[0], args[1], outs=[args[2]])


def build_sub(args: List[ir.RankedTensorType]):
    """Build SpMM kernel.

    This method generates a linalg op with for matrix multiplication using
    just the Python API. Effectively, a generic linalg op is constructed
    that computes C(i,j) += A(i,k) * B(k,j) for annotated matrix A.
    """
    return sub(args[0], args[1], outs=[args[2]])


def get_attr(tensor: Tensor):
    level = []
    tensor_format = tensor.get_format()
    for lvl_format in tensor_format.format():
        if lvl_format == LevelFormat("dense"):
            level.append(st.LevelFormat.dense)
        elif lvl_format == LevelFormat("compressed"):
            level.append(st.LevelFormat.compressed)
    ordering = ir.AffineMap.get_permutation([0, 1])
    return st.EncodingAttr.get(level, ordering, None, 0, 0)


def process_state(state: State):
    ops = []
    inputs = []
    outputs = []
    tensorToMLIRTensor = {}

    with ir.Context() as ctx, ir.Location.unknown():
        f32 = ir.F32Type.get()
        for symbol in state.scope.symbols:
            for tensor_input in symbol.rhs.val.underlying.inputs:
                if tensor_input not in state.scope.symbols:
                    # TODO: Encode tensor type in Tensor type
                    # TODO: Figure out how to get order for tensor
                    attr = get_attr(tensor_input)
                    mlir_tensor = ir.RankedTensorType.get(
                        list(tensor_input.get_shape()), f32, attr)
                    inputs.append(mlir_tensor)
                    # tensorToMLIRTensor[tensor_input] = mlir_tensor
                # inputs.append()
        print(inputs)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            @func.FuncOp.from_py_func()
            def generated_func():
                for item in state.scope.symbols:
                    match type(item.rhs.val.underlying):
                        case sparse_torch.Matmul:
                            op_input = []
                            for tensor_input in item.rhs.val.underlying.inputs:
                                if tensor_input in tensorToMLIRTensor:
                                    op_input.append(
                                        tensorToMLIRTensor[tensor_input])
                                else:
                                    attr = get_attr(tensor_input)
                                    tensor_type = ir.RankedTensorType.get(
                                        list(tensor_input.get_shape()), f32, attr)
                                    mlir_tensor = arith.constant(tensor_type, DenseElementsAttr.get(
                                        np.full(tensor_input.get_shape(), 0, np.float32)))
                                    tensorToMLIRTensor[tensor_input] = mlir_tensor
                                    op_input.append(mlir_tensor)
                            # Create rankedtensortype for output
                            if item in tensorToMLIRTensor:
                                print("FOUND OP IN TENOSR MLIR")
                                op_input.append(tensorToMLIRTensor[item])
                            else:
                                print("CANNOT FIND OP")
                                attr = get_attr(item)
                                # mlir_tensor = ir.RankedTensorType.get(
                                #     list(item.get_shape()), f32, attr)
                                # print(mlir_tensor)
                                # print("GETTING EMPTY OP")
                                mlir_tensor = get_op_result_or_value(
                                    t.EmptyOp(item.get_shape(), f32))
                                tensorToMLIRTensor[item] = mlir_tensor
                                op_input.append(mlir_tensor)

                            print(op_input)
                            build_matmul(op_input)

                            ops.append(item.rhs.val.underlying)
                            inputs.append(item.rhs.val.underlying.inputs)
                            outputs.append(item)
                        case sparse_torch.Add:
                            level = [st.LevelFormat.dense,
                                     st.LevelFormat.dense]
                            ops.append(item.rhs.val.underlying)
                            inputs.append(item.rhs.val.underlying.inputs)
                            outputs.append(item)
                        case sparse_torch.Exp:
                            ops.append(item.rhs.val.underlying)
                            inputs.append(item.rhs.val.underlying.inputs)
                            outputs.append(item)
                        case sparse_torch.Reduce:
                            ops.append(item.rhs.val.underlying)
                            inputs.append(item.rhs.val.underlying.inputs)
                            outputs.append(item)
                        case sparse_torch.MaxReduce:
                            ops.append(item.rhs.val.underlying)
                            inputs.append(item.rhs.val.underlying.inputs)
                            outputs.append(item)
                        case sparse_torch.Div:
                            ops.append(item.rhs.val.underlying)
                            inputs.append(item.rhs.val.underlying.inputs)
                            outputs.append(item)

                        case _:
                            pass
                            print("DEFAULT REACHED")
            # print(ops)


def main():
    support_lib = os.getenv("SUPPORT_LIB")
    # assert support_lib is not None, "SUPPORT_LIB is undefined"
    # if not os.path.exists(support_lib):
    # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)

    with ir.Context() as ctx, ir.Location.unknown():
        count = 0
        # Loop over various ways to compile and annotate the SpMM kernel with
        # a *single* sparse tensor. Note that we deliberate do not exhaustively
        # search the full state space to reduce runtime of the test. It is
        # straightforward to adapt the code below to explore more combinations.

        vl = 1
        e = False
        opt = f"parallelization-strategy=none"
        levels = [
            [st.LevelFormat.dense, st.LevelFormat.dense],
            [st.LevelFormat.dense, st.LevelFormat.compressed],
            [st.LevelFormat.compressed, st.LevelFormat.dense],
            [st.LevelFormat.compressed, st.LevelFormat.compressed],
        ]
        orderings = [
            ir.AffineMap.get_permutation([0, 1]),
            ir.AffineMap.get_permutation([1, 0]),
        ]
        bitwidths = [0]
        # compiler = sparse_compiler.SparseCompiler(
        #     options=opt, opt_level=0, shared_libs=[support_lib]
        # )
        for level in levels:
            for ordering in orderings:
                for pwidth in bitwidths:
                    for iwidth in bitwidths:
                        attr = st.EncodingAttr.get(
                            level, ordering, None, pwidth, iwidth
                        )
                        print(attr)
                        build_matmul(attr, None)
                        count = count + 1
        # CHECK: Passed 8 tests
        print("Passed ", count, "tests")


if __name__ == "__main__":
    main()
