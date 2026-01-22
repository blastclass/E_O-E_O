"""
Activation steering and hook utilities for runtime model manipulation.

This module provides tools for:
- Computing contrastive steering vectors from prompt pairs
- Hook-based activation injection and gating
- Layer-wise intervention through forward hooks

Based on activation steering research showing that model behavior can be
controlled by adding direction vectors computed from contrastive prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class SteeringVector:
    """A computed steering vector with metadata."""

    vector: Tensor
    positive_prompt: str
    negative_prompt: str
    layer_idx: int
    method: str
    magnitude: float


class SteeringHook:
    """
    Hook for adding steering vectors to layer activations.

    Implements activation steering by adding a direction vector scaled by
    a coefficient to layer outputs during forward pass.

    Attributes:
        steering_vector: The direction to steer toward
        coefficient: Scaling factor (higher = stronger steering)
        token_positions: Optional positions to steer (None = all)
        active: Whether the hook is currently active
    """

    def __init__(
        self,
        steering_vector: Tensor,
        coefficient: float = 0.4,
        token_positions: Optional[list[int]] = None,
    ) -> None:
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self.token_positions = token_positions
        self.active = True
        self._call_count = 0

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: Union[Tensor, tuple],
    ) -> Union[Tensor, tuple]:
        """Apply steering to layer output."""
        if not self.active:
            return output

        self._call_count += 1

        # Handle both tensor and tuple outputs
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Ensure steering vector is on correct device/dtype
        steer = self.steering_vector.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Apply steering
        modified = hidden_states.clone()
        if self.token_positions is None:
            modified = modified + self.coefficient * steer
        else:
            for pos in self.token_positions:
                if pos < modified.shape[1]:
                    modified[:, pos, :] = modified[:, pos, :] + self.coefficient * steer

        if rest is not None:
            return (modified,) + rest
        return modified

    def set_coefficient(self, coefficient: float) -> None:
        """Update the steering coefficient."""
        self.coefficient = coefficient

    def set_active(self, active: bool) -> None:
        """Enable or disable the hook."""
        self.active = active


class GatingHook:
    """
    Hook for gating (suppressing) specific tokens at a layer.

    Multiplies activations for specified tokens by a gate value,
    allowing selective suppression of token contributions.

    Attributes:
        token_ids: Token IDs to gate
        gate_value: Multiplier (0.0 = full suppression, 1.0 = no effect)
        input_ids_fn: Function to get current input_ids
        active: Whether the hook is currently active
    """

    def __init__(
        self,
        token_ids: list[int],
        gate_value: float = 0.0,
        input_ids_fn: Optional[Callable[[], Optional[Tensor]]] = None,
    ) -> None:
        self.token_ids = token_ids
        self.gate_value = gate_value
        self.input_ids_fn = input_ids_fn
        self.active = True
        self._token_tensor: Optional[Tensor] = None

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: Union[Tensor, tuple],
    ) -> Union[Tensor, tuple]:
        """Apply gating to layer output."""
        if not self.active:
            return output

        # Get current input_ids
        input_ids = self.input_ids_fn() if self.input_ids_fn else None
        if input_ids is None:
            return output

        # Handle both tensor and tuple outputs
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Lazy initialize token tensor on correct device
        if self._token_tensor is None or self._token_tensor.device != hidden_states.device:
            self._token_tensor = torch.tensor(
                self.token_ids,
                device=hidden_states.device,
            )

        # Create mask for tokens to gate
        mask = torch.isin(input_ids, self._token_tensor)

        if mask.any():
            modified = hidden_states.clone()
            # Apply gate value to matching positions
            modified[:, mask.squeeze(), :] *= self.gate_value

            if rest is not None:
                return (modified,) + rest
            return modified

        return output

    def set_gate_value(self, gate_value: float) -> None:
        """Update the gate value."""
        self.gate_value = gate_value

    def set_active(self, active: bool) -> None:
        """Enable or disable the hook."""
        self.active = active


class AttentionModificationHook:
    """
    Hook for modifying attention patterns directly.

    Can be used to:
    - Zero out attention to specific tokens
    - Boost attention to specific tokens
    - Implement custom attention masking
    """

    def __init__(
        self,
        token_ids: list[int],
        mode: str = "zero",  # "zero", "boost", "mask"
        boost_factor: float = 2.0,
        input_ids_fn: Optional[Callable[[], Optional[Tensor]]] = None,
    ) -> None:
        self.token_ids = token_ids
        self.mode = mode
        self.boost_factor = boost_factor
        self.input_ids_fn = input_ids_fn
        self.active = True

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: tuple,
    ) -> tuple:
        """Modify attention weights."""
        if not self.active:
            return output

        # Attention output typically: (attn_output, attn_weights, ...)
        if len(output) < 2:
            return output

        attn_output, attn_weights = output[0], output[1]
        rest = output[2:] if len(output) > 2 else ()

        input_ids = self.input_ids_fn() if self.input_ids_fn else None
        if input_ids is None or attn_weights is None:
            return output

        # Find positions of target tokens
        token_tensor = torch.tensor(self.token_ids, device=input_ids.device)
        mask = torch.isin(input_ids, token_tensor)

        if not mask.any():
            return output

        modified_weights = attn_weights.clone()

        if self.mode == "zero":
            # Zero attention to target tokens
            modified_weights[:, :, :, mask.squeeze()] = 0
            # Renormalize
            modified_weights = modified_weights / (modified_weights.sum(dim=-1, keepdim=True) + 1e-8)
        elif self.mode == "boost":
            # Boost attention to target tokens
            modified_weights[:, :, :, mask.squeeze()] *= self.boost_factor
            # Renormalize
            modified_weights = modified_weights / (modified_weights.sum(dim=-1, keepdim=True) + 1e-8)
        elif self.mode == "mask":
            # Set to -inf before softmax would be applied
            # This hook runs after softmax, so we set to 0 instead
            modified_weights[:, :, :, mask.squeeze()] = 0
            modified_weights = modified_weights / (modified_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return (attn_output, modified_weights) + rest


def compute_contrastive_steering_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    positive_prompt: str,
    negative_prompt: str,
    layer_idx: int,
    method: str = "mean_diff",
) -> SteeringVector:
    """
    Compute a steering vector from contrastive prompt pairs.

    The steering direction is computed as the difference between activations
    for positive and negative prompts at a specific layer.

    Args:
        model: The model to extract activations from
        tokenizer: The model's tokenizer
        positive_prompt: Prompt representing desired behavior
        negative_prompt: Prompt representing undesired behavior
        layer_idx: Which layer to extract from
        method: How to compute difference - "mean_diff", "last_diff", "pca"

    Returns:
        SteeringVector with computed direction

    Example:
        >>> vector = compute_contrastive_steering_vector(
        ...     model, tokenizer,
        ...     positive_prompt="I feel happy and optimistic",
        ...     negative_prompt="I feel sad and pessimistic",
        ...     layer_idx=15
        ... )
        >>> # Use vector to steer model toward positive sentiment
    """
    # Get layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise AttributeError("Cannot find layers in model architecture")

    activations = {}

    def capture_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

    # Register hook on target layer
    handle = layers[layer_idx].register_forward_hook(capture_hook("target"))

    try:
        # Get activations for positive prompt
        pos_inputs = tokenizer(positive_prompt, return_tensors="pt")
        pos_inputs = {k: v.to(model.device) for k, v in pos_inputs.items()}
        with torch.no_grad():
            model(**pos_inputs)
        pos_activations = activations["target"].clone()

        # Get activations for negative prompt
        neg_inputs = tokenizer(negative_prompt, return_tensors="pt")
        neg_inputs = {k: v.to(model.device) for k, v in neg_inputs.items()}
        with torch.no_grad():
            model(**neg_inputs)
        neg_activations = activations["target"].clone()

    finally:
        handle.remove()

    # Compute steering vector based on method
    if method == "mean_diff":
        # Mean difference across all positions
        pos_mean = pos_activations.mean(dim=1).squeeze()
        neg_mean = neg_activations.mean(dim=1).squeeze()
        vector = pos_mean - neg_mean
    elif method == "last_diff":
        # Difference at last token position
        pos_last = pos_activations[:, -1, :].squeeze()
        neg_last = neg_activations[:, -1, :].squeeze()
        vector = pos_last - neg_last
    elif method == "pca":
        # First principal component of difference
        combined = torch.cat([
            pos_activations.reshape(-1, pos_activations.shape[-1]),
            neg_activations.reshape(-1, neg_activations.shape[-1]),
        ])
        labels = torch.cat([
            torch.ones(pos_activations.shape[1]),
            torch.zeros(neg_activations.shape[1]),
        ])
        # Simple difference of means as approximation
        pos_mean = combined[labels == 1].mean(dim=0)
        neg_mean = combined[labels == 0].mean(dim=0)
        vector = pos_mean - neg_mean
    else:
        raise ValueError(f"Unknown method: {method}")

    return SteeringVector(
        vector=vector,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        layer_idx=layer_idx,
        method=method,
        magnitude=torch.norm(vector).item(),
    )


def compute_multi_layer_steering_vectors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    positive_prompt: str,
    negative_prompt: str,
    layer_indices: Optional[list[int]] = None,
    method: str = "mean_diff",
) -> dict[int, SteeringVector]:
    """
    Compute steering vectors for multiple layers.

    Args:
        model: The model to extract activations from
        tokenizer: The model's tokenizer
        positive_prompt: Prompt representing desired behavior
        negative_prompt: Prompt representing undesired behavior
        layer_indices: Which layers to compute for (None = all)
        method: How to compute difference

    Returns:
        Dictionary mapping layer index to SteeringVector
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    else:
        raise AttributeError("Cannot find layers in model architecture")

    if layer_indices is None:
        layer_indices = list(range(num_layers))

    vectors = {}
    for layer_idx in layer_indices:
        vectors[layer_idx] = compute_contrastive_steering_vector(
            model, tokenizer,
            positive_prompt, negative_prompt,
            layer_idx, method,
        )

    return vectors


class MultiLayerSteering:
    """
    Apply steering vectors across multiple layers simultaneously.

    Allows coordinated steering with different coefficients per layer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        vectors: dict[int, SteeringVector],
        default_coefficient: float = 0.4,
    ) -> None:
        self.model = model
        self.vectors = vectors
        self.coefficients = {idx: default_coefficient for idx in vectors}
        self.hooks: list[tuple[int, SteeringHook, torch.utils.hooks.RemovableHandle]] = []
        self._active = False

    def set_coefficient(self, layer_idx: int, coefficient: float) -> None:
        """Set coefficient for a specific layer."""
        if layer_idx in self.coefficients:
            self.coefficients[layer_idx] = coefficient
            for idx, hook, _ in self.hooks:
                if idx == layer_idx:
                    hook.set_coefficient(coefficient)

    def set_all_coefficients(self, coefficient: float) -> None:
        """Set the same coefficient for all layers."""
        for layer_idx in self.coefficients:
            self.set_coefficient(layer_idx, coefficient)

    def activate(self) -> None:
        """Register all steering hooks."""
        if self._active:
            return

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            raise AttributeError("Cannot find layers in model architecture")

        for layer_idx, vector in self.vectors.items():
            hook = SteeringHook(
                steering_vector=vector.vector,
                coefficient=self.coefficients[layer_idx],
            )
            handle = layers[layer_idx].register_forward_hook(hook)
            self.hooks.append((layer_idx, hook, handle))

        self._active = True

    def deactivate(self) -> None:
        """Remove all steering hooks."""
        for _, _, handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self._active = False

    def __enter__(self) -> "MultiLayerSteering":
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.deactivate()
