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


def build_matmul(args: List[ir.RankedTensorType]):
    return matmul(args[0], args[1], outs=[args[2]])


def build_reduce(args: List[ir.RankedTensorType]):
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
    inputs = [args[0], args[1]]
    output = args[2]
    input_shape = [in_val.type.shape for in_val in inputs]
    output_shape = output.type.shape

    indexing_maps = []

    for in_shape in input_shape:
        if in_shape[-1] == 1:
            new_affine_exprs = [affine.AffineExpr.get_dim(i) for i in range(len(input_shape) - 1)]
            new_affine_exprs.append(affine.AffineExpr.get_constant(0))
            indexing_maps.append(affine.AffineMap.get(len(input_shape), 0, new_affine_exprs))
        else:
            indexing_maps.append(affine.AffineMap.get(len(input_shape), 0, [
                                        affine.AffineExpr.get_dim(i) for i in range(len(input_shape))]))
    
    # Append result map
    indexing_maps.append(affine.AffineMap.get(len(input_shape), 0, [
                                affine.AffineExpr.get_dim(i) for i in range(len(input_shape))]))

    iterator_types_str = ["parallel" for i in range(len(input_shape))]
    iterator_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{s}>") for s in iterator_types_str])
    generic_op = linalg.GenericOp([output.type], inputs, [
                                  output], indexing_maps, iterator_types_attr)

    with ir.InsertionPoint(generic_op.regions[0].blocks.append(*[inputs[0].type.element_type, inputs[1].type.element_type, output.type.element_type])):
        in_ = generic_op.regions[0].blocks[0].arguments[0]
        out_ = generic_op.regions[0].blocks[0].arguments[1]
        red = arith.divf(in_, out_)
        linalg.YieldOp([red])
    
    return generic_op


def build_add(args: List[ir.RankedTensorType]):
    return add(args[0], args[1], outs=[args[2]])

def to_affine(maps : AffineMap) -> AffineMap:
    return maps
    # t = maps.isinstance(AffineMap)
    # print(t)


def build_sub(args: List[ir.RankedTensorType]):
    inputs = [args[0], args[1]]
    output = args[2]
    input_shape = [in_val.type.shape for in_val in inputs]
    output_shape = output.type.shape

    indexing_maps = []

    for in_shape in input_shape:
        if in_shape[-1] == 1:
            new_affine_exprs = [affine.AffineExpr.get_dim(i) for i in range(len(input_shape) - 1)]
            new_affine_exprs.append(affine.AffineExpr.get_constant(0))
            indexing_maps.append(affine.AffineMap.get(len(input_shape), 0, new_affine_exprs))
        else:
            indexing_maps.append(affine.AffineMap.get(len(input_shape), 0, [
                                        affine.AffineExpr.get_dim(i) for i in range(len(input_shape))]))
    
    # Append result map
    indexing_maps.append(affine.AffineMap.get(len(input_shape), 0, [
                                affine.AffineExpr.get_dim(i) for i in range(len(input_shape))]))

    iterator_types_str = ["parallel" for i in range(len(input_shape))]
    iterator_types_attr = ArrayAttr.get(
        [Attribute.parse(f"#linalg.iterator_type<{s}>") for s in iterator_types_str])
    generic_op = linalg.GenericOp([output.type], inputs, [
                                  output], indexing_maps, iterator_types_attr)

    with ir.InsertionPoint(generic_op.regions[0].blocks.append(*[inputs[0].type.element_type, inputs[1].type.element_type, output.type.element_type])):
        in_ = generic_op.regions[0].blocks[0].arguments[0]
        out_ = generic_op.regions[0].blocks[0].arguments[1]
        red = arith.subf(in_, out_)
        linalg.YieldOp([red])
    
    return generic_op


def build_exp(args: List[ir.RankedTensorType]):
    return exp(args[0], outs=[args[1]])


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
    tensorToMLIRTensor = {}

    with ir.Context() as ctx, ir.Location.unknown():
        i64 = ir.IntegerType.get_signless(64)
        f32 = ir.F32Type.get()
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            @func.FuncOp.from_py_func()
            def generated_func():
                output = None
                zero_f = arith.constant(
                    result=f32, value=0.0)
                zero_i = arith.constant(
                    result=i64, value=0)
                max_const = arith.constant(
                    result=f32, value=arith.FloatAttr.get_f32(float("-inf")))
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
                            init_tensor = tensor.empty(tensor_type, [])
                            mlir_tensor = fill(zero_f, outs=[
                                init_tensor])
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
                        init_tensor = tensor.empty(tensor_type, [])
                        mlir_tensor = fill(max_const if type(
                            op_output.rhs.val.underlying) == sparse_torch.MaxReduce else zero_f, outs=[init_tensor])
                        tensorToMLIRTensor[op_output] = mlir_tensor
                        op_operands.append(mlir_tensor)
                    if type(op_output.rhs.val.underlying) == sparse_torch.MaxReduce:
                        i64_select_out = ir.RankedTensorType.get(
                            op_output.get_shape(), i64)
                        init_tensor = tensor.empty(i64_select_out, [])
                        fill_val = linalg.fill(zero_i, outs=[init_tensor])
                        op_operands.append(fill_val)
                    match type(op_output.rhs.val.underlying):
                        case sparse_torch.Matmul:
                            output = build_matmul(op_operands)
                        case sparse_torch.Add:
                            output = build_add(op_operands)
                        case sparse_torch.Sub:
                            output = build_sub(op_operands)
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
                    # print()
                    # print(op_output, output)
                    # print()
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
