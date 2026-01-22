"""
Junk Token Suppression Experiment.

Tests the hypothesis that suppressing 1-3% of vocabulary (undertrained tokens)
has negligible impact on model performance.

Protocol:
1. Identify bottom 2% embedding norm tokens
2. Systematically zero or clamp their embeddings
3. Measure perplexity on benchmark datasets before/after
4. Report impact per percentage of vocabulary suppressed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..harness import EmbeddingExperimentHarness
from ..token_analysis import find_undertrained_tokens, analyze_embedding_geometry


@dataclass
class SuppressionResult:
    """Results from a single suppression experiment."""

    suppression_percentile: float
    num_tokens_suppressed: int
    baseline_perplexity: float
    suppressed_perplexity: float
    perplexity_change: float
    perplexity_change_pct: float
    token_ids_suppressed: list[int] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Results from running suppression across multiple percentiles."""

    model_id: str
    results: list[SuppressionResult]
    vocab_size: int
    embedding_dim: int
    test_texts: list[str] = field(default_factory=list)


class JunkSuppressionExperiment:
    """
    Experiment to test impact of suppressing undertrained tokens.

    Tests whether zeroing/clamping bottom percentile embeddings
    affects model perplexity on standard text.
    """

    def __init__(
        self,
        harness: EmbeddingExperimentHarness,
        test_texts: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the experiment.

        Args:
            harness: Initialized EmbeddingExperimentHarness
            test_texts: Texts to measure perplexity on (uses defaults if None)
        """
        self.harness = harness
        self.test_texts = test_texts or self._get_default_test_texts()

    def _get_default_test_texts(self) -> list[str]:
        """Return default test texts for perplexity measurement."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "In a world where artificial intelligence continues to advance, "
            "researchers are exploring new ways to understand and control these systems.",
            "The capital of France is Paris, which is known for the Eiffel Tower, "
            "the Louvre Museum, and its rich cultural heritage.",
            "Machine learning models learn patterns from data to make predictions "
            "about new, unseen examples.",
            "Climate change poses significant challenges to ecosystems worldwide, "
            "affecting biodiversity and weather patterns.",
        ]

    def compute_perplexity(
        self,
        texts: Optional[list[str]] = None,
        max_length: int = 512,
    ) -> float:
        """
        Compute average perplexity across test texts.

        Args:
            texts: Texts to evaluate (uses self.test_texts if None)
            max_length: Maximum sequence length

        Returns:
            Average perplexity across texts
        """
        texts = texts or self.test_texts
        total_loss = 0.0
        total_tokens = 0

        self.harness.model.eval()

        for text in texts:
            inputs = self.harness.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            inputs = {k: v.to(self.harness.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.harness.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                num_tokens = inputs["input_ids"].shape[1]
                total_loss += loss * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()

    def suppress_tokens(
        self,
        token_ids: list[int],
        method: str = "zero",
        clamp_value: float = 0.01,
    ) -> None:
        """
        Suppress embeddings for specified tokens.

        Args:
            token_ids: Token IDs to suppress
            method: "zero" to set to zero, "clamp" to set to small value
            clamp_value: Value to use if method is "clamp"
        """
        embeddings = self.harness._get_embedding_weight()

        with torch.no_grad():
            for token_id in token_ids:
                if method == "zero":
                    new_vec = torch.zeros_like(embeddings[token_id])
                elif method == "clamp":
                    new_vec = torch.ones_like(embeddings[token_id]) * clamp_value
                else:
                    raise ValueError(f"Unknown method: {method}")

                self.harness.patch_embedding(token_id, new_vec, snapshot=True)

    def run_single_percentile(
        self,
        percentile: float,
        method: str = "zero",
    ) -> SuppressionResult:
        """
        Run suppression experiment for a single percentile.

        Args:
            percentile: Bottom percentile to suppress (0.02 = 2%)
            method: Suppression method ("zero" or "clamp")

        Returns:
            SuppressionResult with perplexity measurements
        """
        # Measure baseline
        baseline_ppl = self.compute_perplexity()

        # Find tokens to suppress
        embeddings = self.harness._get_embedding_weight()
        junk_tokens = find_undertrained_tokens(embeddings, percentile)

        # Suppress tokens
        self.suppress_tokens(junk_tokens, method=method)

        # Measure suppressed
        suppressed_ppl = self.compute_perplexity()

        # Restore
        self.harness.restore()

        change = suppressed_ppl - baseline_ppl
        change_pct = (change / baseline_ppl) * 100

        return SuppressionResult(
            suppression_percentile=percentile,
            num_tokens_suppressed=len(junk_tokens),
            baseline_perplexity=baseline_ppl,
            suppressed_perplexity=suppressed_ppl,
            perplexity_change=change,
            perplexity_change_pct=change_pct,
            token_ids_suppressed=junk_tokens[:100],  # Store first 100 for reference
        )

    def run_full_experiment(
        self,
        percentiles: Optional[list[float]] = None,
        method: str = "zero",
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """
        Run suppression experiment across multiple percentiles.

        Args:
            percentiles: List of percentiles to test (defaults to range)
            method: Suppression method
            show_progress: Whether to show progress bar

        Returns:
            BenchmarkResult with all results
        """
        if percentiles is None:
            percentiles = [0.01, 0.02, 0.03, 0.05, 0.10]

        results = []
        iterator = percentiles
        if show_progress:
            iterator = tqdm(percentiles, desc="Running suppression experiments")

        for pct in iterator:
            result = self.run_single_percentile(pct, method)
            results.append(result)

        return BenchmarkResult(
            model_id=self.harness.model_id,
            results=results,
            vocab_size=self.harness.vocab_size,
            embedding_dim=self.harness.embedding_dim,
            test_texts=self.test_texts,
        )


def run_suppression_benchmark(
    model_id: str = "google/gemma-3-1b-it",
    percentiles: Optional[list[float]] = None,
    test_texts: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkResult:
    """
    Run complete junk token suppression benchmark.

    Convenience function that sets up harness and runs experiment.

    Args:
        model_id: HuggingFace model identifier
        percentiles: Percentiles to test
        test_texts: Custom test texts
        output_path: Optional path to save results JSON

    Returns:
        BenchmarkResult with all measurements
    """
    harness = EmbeddingExperimentHarness(model_id)
    experiment = JunkSuppressionExperiment(harness, test_texts)
    results = experiment.run_full_experiment(percentiles)

    if output_path:
        # Convert to serializable format
        output = {
            "model_id": results.model_id,
            "vocab_size": results.vocab_size,
            "embedding_dim": results.embedding_dim,
            "test_texts": results.test_texts,
            "results": [
                {
                    "percentile": r.suppression_percentile,
                    "tokens_suppressed": r.num_tokens_suppressed,
                    "baseline_ppl": r.baseline_perplexity,
                    "suppressed_ppl": r.suppressed_perplexity,
                    "change": r.perplexity_change,
                    "change_pct": r.perplexity_change_pct,
                }
                for r in results.results
            ],
        }
        Path(output_path).write_text(json.dumps(output, indent=2))

    return results
