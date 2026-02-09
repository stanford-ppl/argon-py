"""
This file tests capturing multi-layer models in PyTorch with dynamic shapes
using Argon while_loop.

We compare how the captured graph differs when using it with
1) standard torch.compile (FX-level ops)
2) torch.compile + aot_autograd (ATen-level ops)

The batch dimension is marked as dynamic in these tests.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

from argon.state import State
from argon.types.struct import Struct
from argon.virtualization.wrapper import argon_function
from collections import namedtuple

from argon.virtualization.type_mapper import concrete_to_abstract


@argon_function()
def multilayer_forward_full(
    hidden_states: torch.Tensor, layers: nn.ModuleList
) -> torch.Tensor:
    i = 0
    while i < 2:
        hidden_states = layers[i](hidden_states)
        i = i + 1
    return hidden_states


def test_loops_in_function_full():
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # We use the config to load the model to avoid downloading the large weights
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    layers = nn.ModuleList([Qwen3MoeMLP(config) for _ in range(2)])

    batch_size, seq_length, hidden_size = 1, 8, config.hidden_size
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)

    torch._dynamo.mark_dynamic(hidden_states, 0)
    state = State()
    with state:
        result = multilayer_forward_full.virtualized.call_transformed(
            hidden_states, layers
        )

        # Convert the result to abstract to verify it was properly virtualized
        # module_abstract = concrete_to_abstract(result)

    print(f"\ntest_loops")
    print(state)


class TwoLayerQwen3MLP(nn.Module):
    """
    A custom module that runs two consecutive Qwen3MoeMLP layers.

    This mimics how transformer models use for loops to express
    a stack of decoder layers.
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3MoeMLP(config) for _ in range(2)])

    @argon_function()
    def forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        layers = self.layers
        i = 0
        while i < 2:
            hidden_states = layers[i](hidden_states)
            i = i + 1
        return hidden_states


def test_loops_in_class():
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # We use the config to load the model to avoid downloading the large weights
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = TwoLayerQwen3MLP(config)
    model.eval()

    # Prepare dummy inputs matching input shape
    batch_size, seq_length, hidden_size = 1, 8, config.hidden_size
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)

    torch._dynamo.mark_dynamic(hidden_states, 0)

    state = State()
    with state:
        result = model.forward.virtualized.call_transformed(hidden_states)

    print(state)
