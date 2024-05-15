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

from argon.node import sparse_torch
from argon.ref import Exp, Op, Sym
from argon.types.tensor import Tensor, TensorFormat
from argon.state import State

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)


@dsl.linalg_structured_op
def matmul_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.N),
    C=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N, output=True),
):
    C[dsl.D.m, dsl.D.n] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k, dsl.D.n]


def build_SpMM(attr: st.EncodingAttr):
    """Build SpMM kernel.

    This method generates a linalg op with for matrix multiplication using
    just the Python API. Effectively, a generic linalg op is constructed
    that computes C(i,j) += A(i,k) * B(k,j) for annotated matrix A.
    """
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    a = ir.RankedTensorType.get([3, 4], f64, attr)
    b = ir.RankedTensorType.get([4, 2], f64)
    c = ir.RankedTensorType.get([3, 2], f64)
    arguments = [a, b, c]
    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(*arguments)
        def spMxM(*args):
            return matmul_dsl(args[0], args[1], outs=[args[2]])

    return module


def build_compile_and_run_matmul(attr: st.EncodingAttr, compiler):
    # Build.
    module = build_SpMM(attr)
    module.dump()
    func = str(module.operation.regions[0].blocks[0].operations[0].operation)


def process_state(state: State):
    state.scope.dump()
    # FIXME: Figure out how to retrieve ops from state
    # for item in state.scope.symbols:
    # #     if isinstance(item, Sym):
            # print(item)
            # print(item.get_shape())


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
                        build_compile_and_run_SpMM(attr, None)
                        count = count + 1
        # CHECK: Passed 8 tests
        print("Passed ", count, "tests")


if __name__ == "__main__":
    main()