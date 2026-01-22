"""
Per-token contribution analysis for understanding transformer information flow.

This module provides tools for:
- Computing token-wise attention contribution to outputs
- Measuring attention entropy across layers
- Identifying attention sink tokens
- Analyzing layer-wise token importance patterns

Based on research showing transformers exhibit distinct layer-wise patterns:
- Attention sinks: early tokens absorbing disproportionate attention
- Entropy decrease: broader early layers, focused later layers
- 97% natural sparsity in attention patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class TokenContribution:
    """Contribution analysis for a single token."""

    token_id: int
    token_str: str
    position: int
    attention_received: float  # How much attention this token receives
    attention_given: float  # How much attention this token gives to others
    value_norm: float  # Magnitude of value vector
    is_sink: bool  # Whether this appears to be an attention sink


@dataclass
class LayerAttentionStats:
    """Attention statistics for a single layer."""

    layer_idx: int
    entropy_mean: float
    entropy_std: float
    max_attention_mean: float
    sparsity: float  # Fraction of attention weights below threshold
    sink_positions: list[int]


def compute_attention_entropy(
    attention_weights: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute entropy of attention distributions.

    Lower entropy = more focused attention.
    Higher entropy = more distributed attention.

    Args:
        attention_weights: Attention tensor [batch, heads, query, key]
        eps: Small value to prevent log(0)

    Returns:
        Entropy tensor [batch, heads, query]
    """
    # Clamp to prevent numerical issues
    attn = torch.clamp(attention_weights, min=eps)
    entropy = -torch.sum(attn * torch.log(attn), dim=-1)
    return entropy


def compute_attention_sparsity(
    attention_weights: Tensor,
    threshold: float = 0.01,
) -> float:
    """
    Compute fraction of attention weights below threshold.

    Research shows 97% natural sparsity in attention patterns.

    Args:
        attention_weights: Attention tensor [batch, heads, query, key]
        threshold: Weight threshold for considering "sparse"

    Returns:
        Fraction of weights below threshold
    """
    total = attention_weights.numel()
    sparse = (attention_weights < threshold).sum().item()
    return sparse / total


def identify_attention_sinks(
    attention_weights: Tensor,
    value_vectors: Optional[Tensor] = None,
    attention_threshold: float = 0.3,
    value_norm_threshold: float = 0.1,
) -> list[int]:
    """
    Identify attention sink positions.

    Attention sinks are positions that receive high attention but have
    low value vector norms - they absorb attention without contributing
    meaningful information.

    Args:
        attention_weights: Attention tensor [batch, heads, query, key]
        value_vectors: Optional value vectors [batch, heads, seq, head_dim]
        attention_threshold: Min attention to consider as sink candidate
        value_norm_threshold: Max value norm for sink classification

    Returns:
        List of position indices identified as sinks
    """
    # Average attention received per position across heads and queries
    mean_attention = attention_weights.mean(dim=(0, 1, 2))  # [key_positions]

    # Find positions with high attention
    high_attn_mask = mean_attention > attention_threshold

    if value_vectors is not None:
        # Compute value vector norms per position
        v_norms = torch.norm(value_vectors, dim=-1).mean(dim=(0, 1))  # [positions]
        low_value_mask = v_norms < value_norm_threshold
        sink_mask = high_attn_mask & low_value_mask
    else:
        # Without value vectors, use attention alone
        sink_mask = high_attn_mask

    return torch.where(sink_mask)[0].tolist()


def compute_token_contribution(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    target_position: int = -1,
    aggregation: str = "mean",
) -> list[tuple[str, float]]:
    """
    Compute how much each input token contributes to a target position.

    Aggregates attention across all layers and heads to measure
    information flow from each source position to the target.

    Args:
        model: The model to analyze
        tokenizer: The model's tokenizer
        text: Input text to analyze
        target_position: Position to analyze contributions to (-1 = last)
        aggregation: How to aggregate - "mean", "max", "sum"

    Returns:
        List of (token_string, contribution_score) tuples
    """
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attention from all layers: [layers, batch, heads, seq, seq]
    all_attentions = torch.stack(outputs.attentions)

    # Aggregate across layers and heads
    if aggregation == "mean":
        aggregated = all_attentions.mean(dim=(0, 2))  # [batch, seq, seq]
    elif aggregation == "max":
        aggregated = all_attentions.max(dim=0).values.max(dim=1).values  # [batch, seq, seq]
    elif aggregation == "sum":
        aggregated = all_attentions.sum(dim=(0, 2))  # [batch, seq, seq]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Get contributions to target position
    contributions = aggregated[0, target_position, :]  # [seq]

    # Get token strings
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return list(zip(tokens, contributions.tolist()))


def analyze_layer_attention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    sparsity_threshold: float = 0.01,
) -> list[LayerAttentionStats]:
    """
    Analyze attention patterns layer by layer.

    Provides insights into:
    - How attention entropy changes across layers
    - Which layers are most sparse
    - Where attention sinks appear

    Args:
        model: The model to analyze
        tokenizer: The model's tokenizer
        text: Input text to analyze
        sparsity_threshold: Threshold for sparsity calculation

    Returns:
        List of LayerAttentionStats for each layer
    """
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    stats = []
    for layer_idx, attn in enumerate(outputs.attentions):
        # attn shape: [batch, heads, query, key]
        entropy = compute_attention_entropy(attn)
        sparsity = compute_attention_sparsity(attn, sparsity_threshold)
        sinks = identify_attention_sinks(attn)

        max_attn = attn.max(dim=-1).values.mean().item()

        stats.append(LayerAttentionStats(
            layer_idx=layer_idx,
            entropy_mean=entropy.mean().item(),
            entropy_std=entropy.std().item(),
            max_attention_mean=max_attn,
            sparsity=sparsity,
            sink_positions=sinks,
        ))

    return stats


def compute_layer_contribution_flow(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    source_positions: Optional[list[int]] = None,
) -> dict[int, Tensor]:
    """
    Trace how information from source positions flows through layers.

    For each layer, computes the attention-weighted contribution of
    source positions to all other positions.

    Args:
        model: The model to analyze
        tokenizer: The model's tokenizer
        text: Input text to analyze
        source_positions: Positions to trace (None = all)

    Returns:
        Dictionary mapping layer_idx to contribution tensor [source, target]
    """
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]

    if source_positions is None:
        source_positions = list(range(seq_len))

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    flow = {}
    for layer_idx, attn in enumerate(outputs.attentions):
        # Average across heads: [batch, query, key]
        mean_attn = attn.mean(dim=1).squeeze(0)

        # Extract contribution from source positions
        # Transpose so rows are sources, columns are targets
        contribution = mean_attn[:, source_positions].T  # [source, target]
        flow[layer_idx] = contribution

    return flow


def find_information_bottlenecks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    entropy_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Find layer-position pairs where attention becomes highly focused.

    Bottlenecks indicate where the model "makes decisions" about
    which information to preserve.

    Args:
        model: The model to analyze
        tokenizer: The model's tokenizer
        text: Input text to analyze
        entropy_threshold: Entropy below which to flag as bottleneck

    Returns:
        List of (layer_idx, position) bottleneck locations
    """
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    bottlenecks = []
    for layer_idx, attn in enumerate(outputs.attentions):
        entropy = compute_attention_entropy(attn)
        # Average entropy across heads per position
        pos_entropy = entropy.mean(dim=1).squeeze(0)

        low_entropy_positions = torch.where(pos_entropy < entropy_threshold)[0]
        for pos in low_entropy_positions.tolist():
            bottlenecks.append((layer_idx, pos))

    return bottlenecks


def compute_residual_contribution(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
) -> dict:
    """
    Analyze contribution of residual stream vs attention/MLP outputs.

    Helps understand whether the model relies more on "passing through"
    information via residual connections or transforming it.

    Args:
        model: The model to analyze
        tokenizer: The model's tokenizer
        text: Input text to analyze

    Returns:
        Dictionary with residual contribution analysis
    """
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get layer outputs via hooks
    layer_outputs = {}
    layer_inputs = {}

    def make_hooks(layer_idx):
        def input_hook(module, input, output):
            if isinstance(input, tuple):
                layer_inputs[layer_idx] = input[0].detach()
            else:
                layer_inputs[layer_idx] = input.detach()

        def output_hook(module, input, output):
            if isinstance(output, tuple):
                layer_outputs[layer_idx] = output[0].detach()
            else:
                layer_outputs[layer_idx] = output.detach()

        return input_hook, output_hook

    handles = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        for idx, layer in enumerate(layers):
            in_hook, out_hook = make_hooks(idx)
            handles.append(layer.register_forward_pre_hook(
                lambda m, i, idx=idx: layer_inputs.__setitem__(idx, i[0].detach() if isinstance(i, tuple) else i.detach())
            ))
            handles.append(layer.register_forward_hook(out_hook))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # Analyze residual vs transformation contribution
    contributions = {}
    for layer_idx in layer_outputs:
        if layer_idx not in layer_inputs:
            continue

        inp = layer_inputs[layer_idx]
        out = layer_outputs[layer_idx]

        # Residual is passed through unchanged
        # Transformation is out - inp
        transformation = out - inp

        residual_norm = torch.norm(inp, dim=-1).mean().item()
        transform_norm = torch.norm(transformation, dim=-1).mean().item()
        output_norm = torch.norm(out, dim=-1).mean().item()

        contributions[layer_idx] = {
            "residual_norm": residual_norm,
            "transformation_norm": transform_norm,
            "output_norm": output_norm,
            "residual_ratio": residual_norm / (output_norm + 1e-8),
            "transform_ratio": transform_norm / (output_norm + 1e-8),
        }

    return contributions
