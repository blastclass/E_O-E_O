"""
Core EmbeddingExperimentHarness class for Gemma 3 1B embedding manipulation.

This module provides the main harness for conducting embedding experiments including
junk token identification, embedding patching, layer-wise gating, and carrier token
injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


@dataclass
class HookHandle:
    """Wrapper for PyTorch hook handles with metadata."""

    handle: torch.utils.hooks.RemovableHandle
    layer_idx: int
    hook_type: str
    description: str = ""

    def remove(self) -> None:
        """Remove the hook."""
        self.handle.remove()


@dataclass
class EmbeddingSnapshot:
    """Snapshot of embedding state for restoration."""

    token_id: int
    original_vector: Tensor
    modified_vector: Optional[Tensor] = None
    description: str = ""


class EmbeddingExperimentHarness:
    """
    Complete harness for Gemma 3 1B embedding manipulation experiments.

    This harness provides tools for:
    - Identifying junk/undertrained tokens via embedding norm analysis
    - Patching individual token embeddings (affects tied lm_head)
    - Layer-wise token gating through forward hooks
    - Carrier token injection using semantic anchor text

    Attributes:
        model: The loaded Gemma model
        tokenizer: The associated tokenizer
        original_embeddings: Clone of original embedding matrix for restoration
        hooks: List of registered hook handles
        snapshots: Dictionary of embedding snapshots keyed by token_id

    Example:
        >>> harness = EmbeddingExperimentHarness("google/gemma-3-1b-it")
        >>> junk_ids = harness.identify_junk_tokens()
        >>> harness.inject_carrier_token(junk_ids[0], "uncertainty marker")
        >>> # Run inference with modified embeddings
        >>> harness.restore()  # Restore original state
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-1b-it",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize the harness with a Gemma model.

        Args:
            model_id: HuggingFace model identifier
            torch_dtype: Data type for model weights (bfloat16 recommended for Gemma)
            device_map: Device placement strategy ("auto", "cuda", "cpu", etc.)
            load_in_8bit: Enable 8-bit quantization via bitsandbytes
            load_in_4bit: Enable 4-bit quantization via bitsandbytes
        """
        self.model_id = model_id
        self.torch_dtype = torch_dtype

        # Load model with specified configuration
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id, **load_kwargs
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

        # Store original embeddings for restoration
        self._original_embeddings: Tensor = self._get_embedding_weight().clone()

        # Track registered hooks and embedding modifications
        self.hooks: list[HookHandle] = []
        self.snapshots: dict[int, EmbeddingSnapshot] = {}

        # State for hook context (set during forward pass)
        self._current_input_ids: Optional[Tensor] = None

    def _get_embedding_weight(self) -> Tensor:
        """Get the embedding weight tensor, handling different model architectures."""
        # Gemma architecture: model.model.embed_tokens
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens.weight
        # Fallback for other architectures
        if hasattr(self.model, "get_input_embeddings"):
            embed_layer = self.model.get_input_embeddings()
            if hasattr(embed_layer, "weight"):
                return embed_layer.weight
        raise AttributeError(f"Cannot find embedding weight in model architecture: {type(self.model)}")

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layers, handling different model architectures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError(f"Cannot find transformer layers in model architecture: {type(self.model)}")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (hidden size)."""
        return self._get_embedding_weight().shape[1]

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._get_embedding_weight().shape[0]

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return len(self._get_layers())

    @property
    def device(self) -> torch.device:
        """Return the device of the embedding weights."""
        return self._get_embedding_weight().device

    def identify_junk_tokens(
        self,
        threshold_percentile: float = 0.02,
        method: str = "norm",
    ) -> list[int]:
        """
        Return token IDs with lowest embedding norms (likely undertrained).

        Args:
            threshold_percentile: Bottom percentile to consider as junk (0.02 = bottom 2%)
            method: Analysis method - "norm" for L2 norm, "variance" for per-dim variance

        Returns:
            List of token IDs identified as potentially undertrained

        Note:
            Gemma's tied embeddings complicate norm analysis. Tokens with extremely
            low norms relative to the distribution are most likely to be undertrained.
            Cross-reference with unreachable token analysis for higher confidence.
        """
        embeddings = self._get_embedding_weight().data

        if method == "norm":
            scores = torch.norm(embeddings, dim=1)
            threshold = torch.quantile(scores, threshold_percentile)
            candidates = torch.where(scores < threshold)[0].tolist()
        elif method == "variance":
            # Per-token variance across embedding dimensions
            scores = torch.var(embeddings, dim=1)
            threshold = torch.quantile(scores, threshold_percentile)
            candidates = torch.where(scores < threshold)[0].tolist()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'norm' or 'variance'.")

        return candidates

    def find_unreachable_tokens(self, sample_size: Optional[int] = None) -> list[int]:
        """
        Find tokens that cannot be reached via encode-decode roundtrip.

        These are tokens that exist in the vocabulary but are never produced
        by the tokenizer for any input text - guaranteed junk slots.

        Args:
            sample_size: Number of tokens to test (None = all tokens, slow)

        Returns:
            List of unreachable token IDs
        """
        unreachable = []
        test_range = range(self.vocab_size) if sample_size is None else range(min(sample_size, self.vocab_size))

        for token_id in test_range:
            try:
                decoded = self.tokenizer.decode([token_id])
                reencoded = self.tokenizer.encode(decoded, add_special_tokens=False)
                if len(reencoded) != 1 or reencoded[0] != token_id:
                    unreachable.append(token_id)
            except Exception:
                # Decoding failures indicate junk tokens
                unreachable.append(token_id)

        return unreachable

    def get_embedding(self, token_id: int) -> Tensor:
        """
        Get the embedding vector for a specific token.

        Args:
            token_id: The token ID to retrieve

        Returns:
            The embedding vector (shape: [embedding_dim])
        """
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} out of range [0, {self.vocab_size})")
        return self._get_embedding_weight()[token_id].clone()

    def patch_embedding(
        self,
        token_id: int,
        new_vector: Tensor,
        normalize: bool = False,
        snapshot: bool = True,
    ) -> None:
        """
        Replace a single token embedding (affects tied lm_head).

        Args:
            token_id: The token ID to modify
            new_vector: The new embedding vector (must match embedding_dim)
            normalize: Whether to L2-normalize the new vector
            snapshot: Whether to save original for later restoration

        Note:
            In Gemma, input and output embeddings are tied. Modifying embed_tokens.weight
            simultaneously affects the lm_head used for next-token prediction.
        """
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} out of range [0, {self.vocab_size})")

        if new_vector.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Vector dimension {new_vector.shape[-1]} doesn't match embedding dim {self.embedding_dim}"
            )

        # Ensure vector is on correct device and dtype
        new_vector = new_vector.to(device=self.device, dtype=self.torch_dtype)

        if normalize:
            new_vector = new_vector / (torch.norm(new_vector) + 1e-8)

        # Take snapshot before modification
        if snapshot and token_id not in self.snapshots:
            self.snapshots[token_id] = EmbeddingSnapshot(
                token_id=token_id,
                original_vector=self._get_embedding_weight()[token_id].clone(),
            )

        # Update snapshot with new modified vector
        if token_id in self.snapshots:
            self.snapshots[token_id].modified_vector = new_vector.clone()

        # Apply modification
        with torch.no_grad():
            self._get_embedding_weight()[token_id] = new_vector

    def inject_carrier_token(
        self,
        token_id: int,
        semantic_anchor_text: str,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> Tensor:
        """
        Repurpose a junk token as a semantic anchor using text-derived embeddings.

        Args:
            token_id: The token ID to repurpose (should be from junk pool)
            semantic_anchor_text: Text whose meaning should be captured
            pooling: How to combine token embeddings - "mean", "first", "last", "max"
            normalize: Whether to normalize the resulting vector

        Returns:
            The injected carrier vector

        Example:
            >>> harness.inject_carrier_token(junk_id, "this indicates uncertainty")
            >>> # Token junk_id now encodes the semantic concept of uncertainty
        """
        # Tokenize the anchor text
        inputs = self.tokenizer(semantic_anchor_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Get embeddings for anchor tokens
        with torch.no_grad():
            anchor_embeds = self._get_embedding_weight()[input_ids.squeeze()]

        # Apply pooling strategy
        if pooling == "mean":
            carrier_vector = anchor_embeds.mean(dim=0)
        elif pooling == "first":
            carrier_vector = anchor_embeds[0]
        elif pooling == "last":
            carrier_vector = anchor_embeds[-1]
        elif pooling == "max":
            carrier_vector = anchor_embeds.max(dim=0).values
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Inject into target token slot
        self.patch_embedding(token_id, carrier_vector, normalize=normalize)

        return carrier_vector

    def register_layer_gate(
        self,
        layer_idx: int,
        token_ids: list[int],
        gate_value: float = 0.0,
    ) -> HookHandle:
        """
        Suppress specific token contributions at a target layer.

        Args:
            layer_idx: The transformer layer index (0-based)
            token_ids: List of token IDs to gate
            gate_value: Multiplier for gated tokens (0.0 = full suppression)

        Returns:
            HookHandle for later removal

        Note:
            Requires setting self._current_input_ids before forward pass.
            Use the generate_with_context() method for automatic handling.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        token_tensor = torch.tensor(token_ids, device=self.device)

        def gating_hook(module: nn.Module, input: tuple, output: Tensor) -> Tensor:
            if self._current_input_ids is None:
                return output

            # Create mask for tokens to gate
            mask = torch.isin(
                self._current_input_ids,
                token_tensor,
            )

            # Apply gating (output shape: [batch, seq, hidden])
            if mask.any():
                output = output.clone()
                output[:, mask.squeeze(), :] *= gate_value

            return output

        # Register hook on layer output
        layer = self._get_layers()[layer_idx]
        handle = layer.register_forward_hook(gating_hook)

        hook_handle = HookHandle(
            handle=handle,
            layer_idx=layer_idx,
            hook_type="gate",
            description=f"Gate tokens {token_ids[:5]}{'...' if len(token_ids) > 5 else ''} at layer {layer_idx}",
        )
        self.hooks.append(hook_handle)

        return hook_handle

    def register_steering_hook(
        self,
        layer_idx: int,
        steering_vector: Tensor,
        coefficient: float = 0.4,
        token_positions: Optional[list[int]] = None,
    ) -> HookHandle:
        """
        Add a steering vector to activations at a specific layer.

        Args:
            layer_idx: The transformer layer index
            steering_vector: The direction vector to add (shape: [hidden_dim])
            coefficient: Scaling factor for the steering vector
            token_positions: Optional list of sequence positions to steer (None = all)

        Returns:
            HookHandle for later removal

        Reference:
            Activation steering technique from "Steering Language Models" research.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

        steering_vector = steering_vector.to(device=self.device, dtype=self.torch_dtype)

        def steering_hook(module: nn.Module, input: tuple, output: Tensor) -> Tensor:
            modified = output.clone()
            if token_positions is None:
                modified += coefficient * steering_vector
            else:
                for pos in token_positions:
                    if pos < modified.shape[1]:
                        modified[:, pos, :] += coefficient * steering_vector
            return modified

        layer = self._get_layers()[layer_idx]
        handle = layer.register_forward_hook(steering_hook)

        hook_handle = HookHandle(
            handle=handle,
            layer_idx=layer_idx,
            hook_type="steer",
            description=f"Steering at layer {layer_idx} with coef={coefficient}",
        )
        self.hooks.append(hook_handle)

        return hook_handle

    def forward_with_context(
        self,
        input_ids: Tensor,
        **kwargs,
    ) -> tuple:
        """
        Run forward pass with input_ids context available to hooks.

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments passed to model forward

        Returns:
            Model outputs
        """
        self._current_input_ids = input_ids
        try:
            outputs = self.model(input_ids=input_ids, **kwargs)
        finally:
            self._current_input_ids = None
        return outputs

    def generate_with_context(
        self,
        input_ids: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Run generation with input_ids context available to hooks.

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments passed to model.generate

        Returns:
            Generated token IDs
        """
        self._current_input_ids = input_ids
        try:
            outputs = self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            self._current_input_ids = None
        return outputs

    def restore_embedding(self, token_id: int) -> bool:
        """
        Restore a single token embedding to its original state.

        Args:
            token_id: The token ID to restore

        Returns:
            True if restoration was performed, False if no snapshot existed
        """
        if token_id not in self.snapshots:
            return False

        snapshot = self.snapshots[token_id]
        with torch.no_grad():
            self._get_embedding_weight()[token_id] = snapshot.original_vector

        del self.snapshots[token_id]
        return True

    def restore(self) -> None:
        """Restore all original embeddings and remove all hooks."""
        # Restore embeddings
        with torch.no_grad():
            self._get_embedding_weight().copy_(self._original_embeddings)

        # Clear snapshots
        self.snapshots.clear()

        # Remove all hooks
        for hook_handle in self.hooks:
            hook_handle.remove()
        self.hooks.clear()

        # Clear context
        self._current_input_ids = None

    def remove_hook(self, hook_handle: HookHandle) -> bool:
        """
        Remove a specific hook.

        Args:
            hook_handle: The hook handle to remove

        Returns:
            True if hook was found and removed
        """
        if hook_handle in self.hooks:
            hook_handle.remove()
            self.hooks.remove(hook_handle)
            return True
        return False

    def get_embedding_stats(self) -> dict:
        """
        Compute statistics about the embedding matrix.

        Returns:
            Dictionary with embedding statistics
        """
        embeddings = self._get_embedding_weight().data

        norms = torch.norm(embeddings, dim=1)
        variances = torch.var(embeddings, dim=1)

        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "norm_mean": norms.mean().item(),
            "norm_std": norms.std().item(),
            "norm_min": norms.min().item(),
            "norm_max": norms.max().item(),
            "variance_mean": variances.mean().item(),
            "variance_std": variances.std().item(),
            "modified_tokens": list(self.snapshots.keys()),
            "active_hooks": len(self.hooks),
        }

    def __repr__(self) -> str:
        return (
            f"EmbeddingExperimentHarness("
            f"model_id='{self.model_id}', "
            f"vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_layers={self.num_layers}, "
            f"modified_tokens={len(self.snapshots)}, "
            f"active_hooks={len(self.hooks)})"
        )

    def __enter__(self) -> "EmbeddingExperimentHarness":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.restore()
