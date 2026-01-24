from typing import override
import torch.nn as nn
from torch import Tensor
from argon.ref import Ref
from argon.types.integer import Integer
from argon.srcctx import SrcCtx
from argon.state import stage
from argon.virtualization.type_mapper import concrete_to_abstract

"""
This file contains the types for the torch library.

- nn.ModuleList
- Tensor

Add `import argon.types.torch` to src/argon/types/__init__.py to use these types.
"""


class TorchTensor(Ref[Tensor, "TorchTensor"]):
    """
    The TorchTensor class represents a tensor in the Argon language.
    """

    @override
    def fresh(self) -> "TorchTensor":
        return TorchTensor()


concrete_to_abstract[Tensor] = lambda x: TorchTensor().const(x)


class NNModuleList(Ref[nn.ModuleList, "NNModuleList"]):
    """
    The NNModuleList class represents a list of nn.Module objects in the Argon language.
    """

    @override
    def fresh(self) -> "NNModuleList":
        return NNModuleList()

    def __getitem__(self, index: Integer) -> "NNModule":
        """
        Index into the ModuleList to get a specific module.

        OPTION 1: Returns an Index operation that captures:
        - collection: self (the NNModuleList)
        - index: index (which layer)

        When this module is later called with hidden_states, the FunctionCall
        will capture:
        - func: the result of this Index operation (the NNModule)
        - args: [hidden_states] (the input tensor)

        So the full information (which layer + what input) is preserved in
        the operation graph.
        """
        from argon.node.index_ops import Index

        # Option 1: Return Index[NNModule]
        return stage(Index[NNModule](self, index), ctx=SrcCtx.new(2))

        # Option 3 Alternative: Return Index[IndexedNNModule]
        # which stores parent and index in the type itself
        # return stage(Index[IndexedNNModule](self, index), ctx=SrcCtx.new(2))


concrete_to_abstract[nn.ModuleList] = lambda x: NNModuleList().const(x)
# func that takes the instance (concrete) used in the code and maps that to make a const


class NNModule(Ref[nn.Module, "NNModule"]):
    """
    The NNModule class represents a nn.Module object in the Argon language.
    """

    @override
    def fresh(self) -> "NNModule":
        return NNModule()

    def __call__(self, *input) -> TorchTensor:
        from argon.node.module_call import NNModuleCall

        return stage(NNModuleCall[TorchTensor](self, list(input)), ctx=SrcCtx.new(2))


# Option 3: IndexedNNModule that stores index information
# This would be used if you want the index information embedded in the type
# rather than just in the operation. You'd modify Index to return IndexedNNModule.
class IndexedNNModule(NNModule):
    """
    An NNModule that knows its index in the parent ModuleList.

    OPTION 3: When Index returns IndexedNNModule, the information is captured as:
    - The Index operation stores: collection (parent), index
    - The IndexedNNModule type provides access to this information
    - When called with hidden_states, FunctionCall captures: args: [hidden_states]

    To use Option 3, modify __getitem__ to return Index[IndexedNNModule]
    instead of Index[NNModule].
    """

    @override
    def fresh(self) -> "IndexedNNModule":
        return IndexedNNModule()

    # The parent and index information would come from the Index operation
    # that created this IndexedNNModule. You'd need to access it via the
    # operation graph or store it as metadata in the Ref.

    # nn.ModuleList (virtualized) & int
    # nn.ModuleList (virtualized) & Integer
    # nn.ModuleList (concrete) & int => as long as nn.Module is virtualized
    # nn.ModuleList (concrete) & Integer


concrete_to_abstract[nn.Module] = lambda x: NNModule().const(x)
