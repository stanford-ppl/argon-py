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
from mlir.dialects import tensor
from mlir.dialects import arith
from mlir.dialects import affine
from mlir.dialects import linalg
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


@dsl.linalg_structured_op
def reduce_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N, output=True),
):
    pass
    # C[dsl.D.m, dsl.D.n] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k, dsl.D.n]


def build_matmul(args: List[ir.RankedTensorType]):
    return matmul(args[0], args[1], outs=[args[2]])

# @linalg_structured_op(op_name="custom_op_name")
# def reduce_func(I=TensorDef(T, *S.DimList), O=TensorDef(T, *S.DimList[:-1], output=True)):
#     O[*S.Dimlist[:-1]] += I[*S.DimList]


def build_reduce(args: List[ir.RankedTensorType]):

    # t = reduce_func(args[0], outs=[args[1]])
    inputs = args[0]
    output = args[1]
    input_shape = inputs.type.shape
    output_shape = output.type.shape
    map0 = affine.AffineMap.get(len(input_shape), 0, [
                                affine.AffineExpr.get_dim(i) for i in range(len(input_shape))])
    new_affine_exprs = [affine.AffineExpr.get_dim(
        i) for i in range(len(input_shape) - 1)]
    new_affine_exprs.append(affine.AffineExpr.get_constant(0))
    map1 = affine.AffineMap.get(len(input_shape), 0, new_affine_exprs)
    indexing_maps = [map0, map1]
    iterator_types_str = ["parallel" for i in range(len(input_shape) - 1)]
    iterator_types_str.append("reduction")
    iterator_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{s}>") for s in iterator_types_str])
    generic_op = linalg.GenericOp([output.type], [inputs], [
                                  output], indexing_maps, iterator_types_attr)

    with ir.InsertionPoint(generic_op.regions[0].blocks.append(*[inputs.type.element_type, output.type.element_type])):
        in_ = generic_op.regions[0].blocks[0].arguments[0]
        out_ = generic_op.regions[0].blocks[0].arguments[1]
        red = arith.addf(in_, out_)
        linalg.YieldOp([red])

    return get_op_result_or_op_results(generic_op)


def build_max_reduce(args: List[ir.RankedTensorType]):
    # i64 = ir.IntegerType.get_signless(64)
    inputs = args[0]
    output = args[1]
    select_output = args[2]
    input_shape = inputs.type.shape
    output_shape = output.type.shape
    map0 = affine.AffineMap.get(len(input_shape), 0, [
                                affine.AffineExpr.get_dim(i) for i in range(len(input_shape))])
    new_affine_exprs = [affine.AffineExpr.get_dim(
        i) for i in range(len(input_shape) - 1)]
    new_affine_exprs.append(affine.AffineExpr.get_constant(0))
    map1 = affine.AffineMap.get(len(input_shape), 0, new_affine_exprs)
    indexing_maps = [map0, map1, map1]
    iterator_types_str = ["parallel" for i in range(len(input_shape) - 1)]
    iterator_types_str.append("reduction")
    iterator_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{s}>") for s in iterator_types_str])

    generic_op = linalg.GenericOp([output.type, select_output.type], [inputs], [
                                  output, select_output], indexing_maps, iterator_types_attr)
    with ir.InsertionPoint(generic_op.regions[0].blocks.append(*[inputs.type.element_type, output.type.element_type, select_output.type.element_type])):
        in_ = generic_op.regions[0].blocks[0].arguments[0]
        out_ = generic_op.regions[0].blocks[0].arguments[1]
        out_1 = generic_op.regions[0].blocks[0].arguments[2]
        index_1 = linalg.IndexOp(len(inputs.type.shape) - 1).result
        index_1_i64 = arith.IndexCastOp(out_1.type, index_1).result
        max_f = arith.MaximumFOp(in_, out_).result
        cmp_f = arith.CmpFOp(arith.CmpFPredicate.OGT, in_, out_).result
        select = arith.SelectOp(cmp_f, index_1_i64, out_1).result
        linalg.YieldOp([max_f, select])
    return get_op_result_or_op_results(generic_op)[0]


def build_div(args: List[ir.RankedTensorType]):
    return div(args[0], args[1], outs=[args[2]])


def build_add(args: List[ir.RankedTensorType]):
    return add(args[0], args[1], outs=[args[2]])


def build_sub(args: List[ir.RankedTensorType]):
    return sub(args[0], args[1], outs=[args[2]])


def build_exp(args: List[ir.RankedTensorType]):
    return exp(args[0], outs=[args[1]])


def test_build():
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):

            f32 = ir.F32Type.get()
            i64 = ir.IntegerType.get_signless(64)

            # Input tensor type
            tensor_1x10xf32 = ir.RankedTensorType.get([1, 10], f32)

            # Output tensor types
            tensor_1x1xf32 = ir.RankedTensorType.get([1, 1], f32)
            tensor_1x1xi64 = ir.RankedTensorType.get([1, 1], i64)

            # Indexing maps
            map0 = affine.AffineMap.get(4, 0, [affine.AffineDimExpr.get(
                # (d0, d1, d2, d3) -> (d0, d1)
                0), affine.AffineDimExpr.get(1)])
            # (d0, d1, d2, d3) -> (d2)
            map1 = affine.AffineMap.get(4, 0, [affine.AffineDimExpr.get(2)])
            # (d0, d1, d2, d3) -> (d3)
            map2 = affine.AffineMap.get(4, 0, [affine.AffineDimExpr.get(3)])

            # Build the linalg.generic operation
            generic_op = linalg.GenericOp(
                [tensor_1x1xf32, tensor_1x1xi64],  # Output operand types
                [tensor_1x10xf32],                  # Input operand types
                [tensor_1x1xf32, tensor_1x1xi64],  # Output operand types
                [map0, map1, map2],                 # Indexing maps
                ["parallel", "reduction"],           # Iterator types
                doc=None, library_call=None)

            # Region for the body of the operation
            with ir.InsertionPoint(generic_op.regions[0].blocks.append()):
                in_ = generic_op.regions[0].blocks[0].arguments[0]
                out_ = generic_op.regions[0].blocks[0].arguments[1]
                out_5 = generic_op.regions[0].blocks[0].arguments[2]

                index_1 = linalg.IndexOp(1).result
                index_1_i64 = arith.IndexCastOp(i64, index_1).result
                max_f = arith.MaximumFOp(in_, out_).result
                cmp_f = arith.CmpFOp(arith.CmpFPredicate.OGT, in_, out_).result
                select = arith.SelectOp(cmp_f, index_1_i64, out_5).result
                linalg.YieldOp([max_f, select])

            # Print the generated IR
            # print(module)


def get_attr(val: Tensor):
    level = []
    tensor_format = val.get_format()
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

    # test_build()
    # exit(0)

    with ir.Context() as ctx, ir.Location.unknown():
        f32 = ir.F32Type.get()
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            @func.FuncOp.from_py_func()
            def generated_func():
                output = None
                for op_output in state.scope.symbols:
                    op_operands = []
                    for tensor_input in op_output.rhs.val.underlying.inputs:
                        if tensor_input in tensorToMLIRTensor:
                            op_operands.append(
                                tensorToMLIRTensor[tensor_input])
                        else:
                            attr = get_attr(tensor_input)
                            tensor_type = ir.RankedTensorType.get(
                                list(tensor_input.get_shape()), f32, attr)
                            # TODO: Replace with actual values
                            mlir_val_tensor = arith.constant(
                                result=f32, value=0.0)
                            mlir_tensor = fill(mlir_val_tensor, outs=[
                                tensor.empty(tensor_type, [])])
                            tensorToMLIRTensor[tensor_input] = mlir_tensor
                            op_operands.append(mlir_tensor)
                    # Create rankedtensortype for output
                    if op_output in tensorToMLIRTensor:
                        op_operands.append(tensorToMLIRTensor[op_output])
                    else:
                        # If output rankedTensorType does not exist, create an empty op and set the output to that
                        attr = get_attr(op_output)
                        tensor_type = ir.RankedTensorType.get(
                            op_output.get_shape(), f32, attr)
                        mlir_tensor = tensor.empty(tensor_type, [])
                        tensorToMLIRTensor[op_output] = mlir_tensor
                        op_operands.append(mlir_tensor)
                    if type(op_output.rhs.val.underlying) == sparse_torch.MaxReduce:
                        i64 = ir.IntegerType.get_signless(64)
                        i64_select_out = ir.RankedTensorType.get(op_output.get_shape(), i64)
                        init_tensor = tensor.empty(i64_select_out, [])
                        const_val = arith.constant(result=i64, value=0)
                        fill_val = linalg.fill(const_val, outs=[init_tensor])
                        op_operands.append(fill_val)
                    match type(op_output.rhs.val.underlying):
                        case sparse_torch.Matmul:
                            output = build_matmul(op_operands)
                        case sparse_torch.Add:
                            output = build_add(op_operands)
                        case sparse_torch.Exp:
                            output = build_exp(op_operands)
                        case sparse_torch.Reduce:
                            output = build_reduce(op_operands)
                        case sparse_torch.MaxReduce:
                            output = build_max_reduce(op_operands)
                        case sparse_torch.Div:
                            output = build_div(op_operands)
                        case _:
                            pass
                    tensorToMLIRTensor[op_output] = output
                return output
        print(module)


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
                        build_matmul(attr, None)
                        count = count + 1
        # CHECK: Passed 8 tests
        print("Passed ", count, "tests")


if __name__ == "__main__":
    main()
