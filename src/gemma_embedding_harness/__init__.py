"""
Gemma Embedding Harness - Experimental toolkit for Gemma 3 1B embedding manipulation.

This package provides tools for:
- Identifying undertrained/junk tokens via embedding geometry analysis
- Hook-based injection and layer-wise token gating
- Activation steering and carrier token injection
- Per-token contribution analysis across transformer layers

Example usage:
    >>> from gemma_embedding_harness import EmbeddingExperimentHarness
    >>> harness = EmbeddingExperimentHarness("google/gemma-3-1b-it")
    >>> junk_tokens = harness.identify_junk_tokens(threshold_percentile=0.02)
    >>> harness.inject_carrier_token(junk_tokens[0], "begin structured reasoning")
"""

from .harness import EmbeddingExperimentHarness
from .token_analysis import (
    compute_embedding_norms,
    find_undertrained_tokens,
    find_unreachable_tokens,
    analyze_embedding_geometry,
)
from .steering import (
    compute_contrastive_steering_vector,
    SteeringHook,
    GatingHook,
)
from .contribution import (
    compute_token_contribution,
    compute_attention_entropy,
    identify_attention_sinks,
)

__version__ = "0.1.0"
__all__ = [
    "EmbeddingExperimentHarness",
    "compute_embedding_norms",
    "find_undertrained_tokens",
    "find_unreachable_tokens",
    "analyze_embedding_geometry",
    "compute_contrastive_steering_vector",
    "SteeringHook",
    "GatingHook",
    "compute_token_contribution",
    "compute_attention_entropy",
    "identify_attention_sinks",
]
