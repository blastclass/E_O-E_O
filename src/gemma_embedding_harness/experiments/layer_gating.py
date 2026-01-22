"""
Layer-wise Token Gating Experiment.

Tests how gating specific tokens at different transformer layers
affects attention patterns and model outputs.

Protocol:
1. Register forward hooks on specific layers to attenuate token contributions
2. Start by gating junk tokens only in early layers (0-6)
3. Compare against gating in late layers
4. Measure attention entropy changes and output quality
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from tqdm import tqdm

from ..harness import EmbeddingExperimentHarness, HookHandle
from ..token_analysis import find_undertrained_tokens
from ..contribution import (
    analyze_layer_attention,
    compute_attention_entropy,
    LayerAttentionStats,
)


@dataclass
class GatingEffect:
    """Effect of gating on a specific layer."""

    layer_idx: int
    entropy_before: float
    entropy_after: float
    entropy_change: float
    entropy_change_pct: float
    output_similarity: float  # Cosine similarity to ungated output


@dataclass
class GatingExperimentResult:
    """Results from a gating experiment configuration."""

    gated_layers: list[int]
    gate_value: float
    num_tokens_gated: int
    effects: list[GatingEffect]
    overall_entropy_change: float
    output_divergence: float  # Measure of how different outputs are


@dataclass
class LayerGatingAnalysis:
    """Complete analysis of layer-wise gating effects."""

    model_id: str
    vocab_size: int
    num_layers: int
    early_layer_results: GatingExperimentResult
    late_layer_results: GatingExperimentResult
    per_layer_results: list[GatingExperimentResult]
    test_text: str


class LayerGatingExperiment:
    """
    Experiment to analyze layer-wise token gating effects.

    Tests whether gating junk tokens at specific layers affects
    model behavior differently based on layer depth.
    """

    def __init__(
        self,
        harness: EmbeddingExperimentHarness,
        test_text: Optional[str] = None,
    ) -> None:
        """
        Initialize the experiment.

        Args:
            harness: Initialized EmbeddingExperimentHarness
            test_text: Text to use for analysis
        """
        self.harness = harness
        self.test_text = test_text or (
            "The development of artificial intelligence has led to significant "
            "advances in natural language processing, computer vision, and robotics."
        )

    def _get_layer_attention_stats(
        self,
        text: Optional[str] = None,
    ) -> list[LayerAttentionStats]:
        """Get attention statistics for all layers."""
        text = text or self.test_text
        return analyze_layer_attention(
            self.harness.model,
            self.harness.tokenizer,
            text,
        )

    def _compute_output_embedding(
        self,
        text: Optional[str] = None,
    ) -> Tensor:
        """Compute the output hidden state for comparison."""
        text = text or self.test_text
        inputs = self.harness.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.harness.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.harness.model(**inputs, output_hidden_states=True)
            # Get last hidden state, last position
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return last_hidden.squeeze()

    def _cosine_similarity(self, v1: Tensor, v2: Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = v1.flatten().float()
        v2 = v2.flatten().float()
        return torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)
        ).item()

    def run_gating_experiment(
        self,
        layer_indices: list[int],
        token_ids: list[int],
        gate_value: float = 0.0,
    ) -> GatingExperimentResult:
        """
        Run gating experiment for specified layers.

        Args:
            layer_indices: Layers to apply gating
            token_ids: Token IDs to gate
            gate_value: Multiplier (0.0 = full suppression)

        Returns:
            GatingExperimentResult with effects measured
        """
        # Measure baseline
        baseline_stats = self._get_layer_attention_stats()
        baseline_output = self._compute_output_embedding()

        # Register gating hooks
        handles: list[HookHandle] = []
        for layer_idx in layer_indices:
            handle = self.harness.register_layer_gate(
                layer_idx=layer_idx,
                token_ids=token_ids,
                gate_value=gate_value,
            )
            handles.append(handle)

        try:
            # Prepare input context for hooks
            inputs = self.harness.tokenizer(self.test_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.harness.device)

            # Measure with gating
            self.harness._current_input_ids = input_ids
            gated_stats = self._get_layer_attention_stats()
            gated_output = self._compute_output_embedding()
            self.harness._current_input_ids = None

        finally:
            # Remove hooks
            for handle in handles:
                self.harness.remove_hook(handle)

        # Compute effects per layer
        effects = []
        for layer_idx in range(self.harness.num_layers):
            baseline_entropy = baseline_stats[layer_idx].entropy_mean
            gated_entropy = gated_stats[layer_idx].entropy_mean

            change = gated_entropy - baseline_entropy
            change_pct = (change / (baseline_entropy + 1e-8)) * 100

            effects.append(GatingEffect(
                layer_idx=layer_idx,
                entropy_before=baseline_entropy,
                entropy_after=gated_entropy,
                entropy_change=change,
                entropy_change_pct=change_pct,
                output_similarity=self._cosine_similarity(baseline_output, gated_output),
            ))

        overall_change = sum(e.entropy_change for e in effects) / len(effects)
        output_div = 1.0 - self._cosine_similarity(baseline_output, gated_output)

        return GatingExperimentResult(
            gated_layers=layer_indices,
            gate_value=gate_value,
            num_tokens_gated=len(token_ids),
            effects=effects,
            overall_entropy_change=overall_change,
            output_divergence=output_div,
        )

    def run_early_vs_late_comparison(
        self,
        token_ids: Optional[list[int]] = None,
        early_layers: Optional[list[int]] = None,
        late_layers: Optional[list[int]] = None,
        junk_percentile: float = 0.02,
    ) -> tuple[GatingExperimentResult, GatingExperimentResult]:
        """
        Compare gating effects in early vs late layers.

        Args:
            token_ids: Tokens to gate (uses junk tokens if None)
            early_layers: Early layer indices (defaults to 0-6)
            late_layers: Late layer indices (defaults to last 6)
            junk_percentile: Percentile for finding junk tokens

        Returns:
            Tuple of (early_result, late_result)
        """
        num_layers = self.harness.num_layers

        if early_layers is None:
            early_layers = list(range(min(7, num_layers)))
        if late_layers is None:
            late_layers = list(range(max(0, num_layers - 6), num_layers))

        if token_ids is None:
            embeddings = self.harness._get_embedding_weight()
            token_ids = find_undertrained_tokens(embeddings, junk_percentile)

        early_result = self.run_gating_experiment(early_layers, token_ids)
        late_result = self.run_gating_experiment(late_layers, token_ids)

        return early_result, late_result

    def run_per_layer_analysis(
        self,
        token_ids: Optional[list[int]] = None,
        junk_percentile: float = 0.02,
        show_progress: bool = True,
    ) -> list[GatingExperimentResult]:
        """
        Run gating experiment for each layer individually.

        Args:
            token_ids: Tokens to gate (uses junk tokens if None)
            junk_percentile: Percentile for finding junk tokens
            show_progress: Whether to show progress bar

        Returns:
            List of results, one per layer
        """
        if token_ids is None:
            embeddings = self.harness._get_embedding_weight()
            token_ids = find_undertrained_tokens(embeddings, junk_percentile)

        results = []
        iterator = range(self.harness.num_layers)
        if show_progress:
            iterator = tqdm(iterator, desc="Analyzing layers")

        for layer_idx in iterator:
            result = self.run_gating_experiment([layer_idx], token_ids)
            results.append(result)

        return results

    def run_full_analysis(
        self,
        junk_percentile: float = 0.02,
        show_progress: bool = True,
    ) -> LayerGatingAnalysis:
        """
        Run complete layer gating analysis.

        Args:
            junk_percentile: Percentile for junk token identification
            show_progress: Whether to show progress

        Returns:
            LayerGatingAnalysis with all results
        """
        embeddings = self.harness._get_embedding_weight()
        token_ids = find_undertrained_tokens(embeddings, junk_percentile)

        early_result, late_result = self.run_early_vs_late_comparison(
            token_ids=token_ids
        )
        per_layer = self.run_per_layer_analysis(
            token_ids=token_ids,
            show_progress=show_progress,
        )

        return LayerGatingAnalysis(
            model_id=self.harness.model_id,
            vocab_size=self.harness.vocab_size,
            num_layers=self.harness.num_layers,
            early_layer_results=early_result,
            late_layer_results=late_result,
            per_layer_results=per_layer,
            test_text=self.test_text,
        )


def analyze_gating_effects(
    model_id: str = "google/gemma-3-1b-it",
    test_text: Optional[str] = None,
    junk_percentile: float = 0.02,
    output_path: Optional[str] = None,
) -> LayerGatingAnalysis:
    """
    Run complete gating analysis on a model.

    Convenience function that sets up harness and runs analysis.

    Args:
        model_id: HuggingFace model identifier
        test_text: Text to analyze
        junk_percentile: Percentile for junk identification
        output_path: Optional path to save results JSON

    Returns:
        LayerGatingAnalysis with complete results
    """
    harness = EmbeddingExperimentHarness(model_id)
    experiment = LayerGatingExperiment(harness, test_text)
    analysis = experiment.run_full_analysis(junk_percentile)

    if output_path:
        # Convert to serializable format
        def serialize_result(r: GatingExperimentResult) -> dict:
            return {
                "gated_layers": r.gated_layers,
                "gate_value": r.gate_value,
                "tokens_gated": r.num_tokens_gated,
                "overall_entropy_change": r.overall_entropy_change,
                "output_divergence": r.output_divergence,
                "per_layer_effects": [
                    {
                        "layer": e.layer_idx,
                        "entropy_before": e.entropy_before,
                        "entropy_after": e.entropy_after,
                        "change_pct": e.entropy_change_pct,
                    }
                    for e in r.effects
                ],
            }

        output = {
            "model_id": analysis.model_id,
            "vocab_size": analysis.vocab_size,
            "num_layers": analysis.num_layers,
            "test_text": analysis.test_text,
            "early_layers": serialize_result(analysis.early_layer_results),
            "late_layers": serialize_result(analysis.late_layer_results),
            "per_layer": [serialize_result(r) for r in analysis.per_layer_results],
        }
        Path(output_path).write_text(json.dumps(output, indent=2))

    return analysis
