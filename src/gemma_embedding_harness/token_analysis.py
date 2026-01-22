"""
Token analysis utilities for identifying undertrained and junk tokens.

This module implements techniques from "Fishing for Magikarp" (EMNLP 2024) for
identifying undertrained tokens via embedding geometry analysis. Key insight:
embedding L2 norms correlate with training frequency across ten orders of magnitude.

For Gemma's tied embeddings, we use multiple complementary approaches:
- Embedding norm analysis (bottom percentile = likely undertrained)
- Cosine distance from known untrained reference tokens
- Encode-decode roundtrip failures (unreachable tokens)
- Unicode category analysis (guaranteed junk ranges)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


@dataclass
class TokenAnalysis:
    """Analysis results for a single token."""

    token_id: int
    token_str: str
    embedding_norm: float
    embedding_variance: float
    is_unreachable: bool
    unicode_category: Optional[str]
    confidence_score: float  # 0-1, higher = more likely junk


@dataclass
class EmbeddingGeometry:
    """Statistics about embedding space geometry."""

    norms: Tensor
    variances: Tensor
    mean_norm: float
    std_norm: float
    percentile_1: float
    percentile_5: float
    percentile_95: float
    percentile_99: float


def compute_embedding_norms(
    embeddings: Tensor,
    normalize_by_dim: bool = False,
) -> Tensor:
    """
    Compute L2 norms for all embeddings.

    Args:
        embeddings: Embedding matrix [vocab_size, embedding_dim]
        normalize_by_dim: Whether to normalize by sqrt(dim) for comparability

    Returns:
        Tensor of norms [vocab_size]
    """
    norms = torch.norm(embeddings, dim=1)
    if normalize_by_dim:
        norms = norms / (embeddings.shape[1] ** 0.5)
    return norms


def compute_embedding_variances(embeddings: Tensor) -> Tensor:
    """
    Compute per-token variance across embedding dimensions.

    Low variance tokens may indicate undertrained or pathological embeddings.

    Args:
        embeddings: Embedding matrix [vocab_size, embedding_dim]

    Returns:
        Tensor of variances [vocab_size]
    """
    return torch.var(embeddings, dim=1)


def analyze_embedding_geometry(embeddings: Tensor) -> EmbeddingGeometry:
    """
    Compute comprehensive statistics about embedding space geometry.

    Args:
        embeddings: Embedding matrix [vocab_size, embedding_dim]

    Returns:
        EmbeddingGeometry with distribution statistics
    """
    norms = compute_embedding_norms(embeddings)
    variances = compute_embedding_variances(embeddings)

    return EmbeddingGeometry(
        norms=norms,
        variances=variances,
        mean_norm=norms.mean().item(),
        std_norm=norms.std().item(),
        percentile_1=torch.quantile(norms, 0.01).item(),
        percentile_5=torch.quantile(norms, 0.05).item(),
        percentile_95=torch.quantile(norms, 0.95).item(),
        percentile_99=torch.quantile(norms, 0.99).item(),
    )


def find_undertrained_tokens(
    embeddings: Tensor,
    threshold_percentile: float = 0.02,
    method: str = "norm",
) -> list[int]:
    """
    Identify tokens with abnormally low embedding norms/variance.

    Based on "Fishing for Magikarp" finding that embedding norms correlate
    with training frequency. Bottom percentile tokens are likely undertrained.

    Args:
        embeddings: Embedding matrix [vocab_size, embedding_dim]
        threshold_percentile: Bottom percentile to flag (0.02 = 2%)
        method: "norm" for L2 norm, "variance" for dimension variance

    Returns:
        List of token IDs likely undertrained
    """
    if method == "norm":
        scores = compute_embedding_norms(embeddings)
    elif method == "variance":
        scores = compute_embedding_variances(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    threshold = torch.quantile(scores, threshold_percentile)
    return torch.where(scores < threshold)[0].tolist()


def find_unreachable_tokens(
    tokenizer: PreTrainedTokenizer,
    vocab_size: Optional[int] = None,
    show_progress: bool = True,
) -> list[int]:
    """
    Find tokens that fail encode-decode roundtrip.

    These tokens exist in vocabulary but are never produced by the tokenizer
    for any input - guaranteed to be unused/junk slots.

    Args:
        tokenizer: The model's tokenizer
        vocab_size: Vocabulary size (inferred if None)
        show_progress: Whether to show progress bar

    Returns:
        List of unreachable token IDs
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    unreachable = []
    iterator = range(vocab_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Checking token reachability")

    for token_id in iterator:
        try:
            decoded = tokenizer.decode([token_id])
            reencoded = tokenizer.encode(decoded, add_special_tokens=False)

            # Token is unreachable if roundtrip doesn't recover it
            if len(reencoded) != 1 or reencoded[0] != token_id:
                unreachable.append(token_id)
        except Exception:
            # Decoding failures indicate problematic tokens
            unreachable.append(token_id)

    return unreachable


def find_unicode_junk_tokens(
    tokenizer: PreTrainedTokenizer,
    vocab_size: Optional[int] = None,
) -> dict[str, list[int]]:
    """
    Identify tokens in known junk Unicode ranges.

    Returns categorized lists of tokens in:
    - Invalid UTF-8 byte ranges (0xF5-0xFF)
    - Private Use Area (U+E000-U+F8FF)
    - Control characters (C0/C1 controls)
    - Surrogate pairs (U+D800-U+DFFF)
    - Noncharacters (U+FDD0-U+FDEF, U+FFFE, U+FFFF)

    Args:
        tokenizer: The model's tokenizer
        vocab_size: Vocabulary size (inferred if None)

    Returns:
        Dictionary mapping category names to token ID lists
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    categories = {
        "invalid_utf8": [],
        "private_use": [],
        "control_chars": [],
        "surrogates": [],
        "noncharacters": [],
        "other_unusual": [],
    }

    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id])

            for char in decoded:
                code = ord(char)
                cat = unicodedata.category(char)

                # Check various junk categories
                if 0xF5 <= code <= 0xFF:
                    categories["invalid_utf8"].append(token_id)
                    break
                elif 0xE000 <= code <= 0xF8FF:
                    categories["private_use"].append(token_id)
                    break
                elif cat in ("Cc", "Cf") and code not in (0x09, 0x0A, 0x0D):
                    # Control chars except tab, newline, carriage return
                    categories["control_chars"].append(token_id)
                    break
                elif 0xD800 <= code <= 0xDFFF:
                    categories["surrogates"].append(token_id)
                    break
                elif (0xFDD0 <= code <= 0xFDEF) or code in (0xFFFE, 0xFFFF):
                    categories["noncharacters"].append(token_id)
                    break
                elif cat in ("Co", "Cs", "Cn"):
                    # Other private use, surrogates, unassigned
                    categories["other_unusual"].append(token_id)
                    break
        except Exception:
            categories["other_unusual"].append(token_id)

    return categories


def compute_cosine_distances_from_reference(
    embeddings: Tensor,
    reference_ids: list[int],
) -> Tensor:
    """
    Compute cosine distance from known untrained reference tokens.

    For tied embeddings where norm analysis is complicated, measuring
    distance from known junk tokens helps identify similar patterns.

    Args:
        embeddings: Embedding matrix [vocab_size, embedding_dim]
        reference_ids: Token IDs known to be untrained (e.g., from Unicode junk)

    Returns:
        Mean cosine distance from reference tokens [vocab_size]
    """
    if not reference_ids:
        raise ValueError("Need at least one reference token ID")

    # Normalize all embeddings
    normed = embeddings / (torch.norm(embeddings, dim=1, keepdim=True) + 1e-8)

    # Get reference embeddings
    ref_embeds = normed[reference_ids]

    # Compute mean cosine similarity to references
    # High similarity = close to junk tokens
    similarities = torch.mm(normed, ref_embeds.T)  # [vocab, num_refs]
    mean_similarity = similarities.mean(dim=1)

    # Return as distance (1 - similarity)
    return 1.0 - mean_similarity


def comprehensive_junk_analysis(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    norm_threshold: float = 0.02,
    show_progress: bool = True,
) -> dict:
    """
    Perform comprehensive analysis to identify all junk tokens.

    Combines multiple methods:
    1. Embedding norm analysis (bottom percentile)
    2. Encode-decode roundtrip failures
    3. Unicode category analysis
    4. Cosine distance from known junk references

    Args:
        model: The loaded model
        tokenizer: The model's tokenizer
        norm_threshold: Bottom percentile for norm analysis
        show_progress: Whether to show progress bars

    Returns:
        Dictionary with analysis results and token categorizations
    """
    # Get embeddings
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embeddings = model.model.embed_tokens.weight.data
    else:
        embeddings = model.get_input_embeddings().weight.data

    vocab_size = embeddings.shape[0]

    # Run analyses
    geometry = analyze_embedding_geometry(embeddings)
    low_norm = find_undertrained_tokens(embeddings, norm_threshold, method="norm")
    low_var = find_undertrained_tokens(embeddings, norm_threshold, method="variance")
    unreachable = find_unreachable_tokens(tokenizer, vocab_size, show_progress)
    unicode_junk = find_unicode_junk_tokens(tokenizer, vocab_size)

    # Combine Unicode junk categories for reference
    all_unicode_junk = []
    for cat_tokens in unicode_junk.values():
        all_unicode_junk.extend(cat_tokens)
    all_unicode_junk = list(set(all_unicode_junk))

    # Compute cosine distance from Unicode junk if we have references
    if all_unicode_junk:
        cosine_distances = compute_cosine_distances_from_reference(
            embeddings, all_unicode_junk[:100]  # Use subset for efficiency
        )
        close_to_junk_threshold = torch.quantile(cosine_distances, 0.05)
        close_to_junk = torch.where(cosine_distances < close_to_junk_threshold)[0].tolist()
    else:
        cosine_distances = None
        close_to_junk = []

    # Find high-confidence junk (appears in multiple analyses)
    from collections import Counter
    all_candidates = low_norm + low_var + unreachable + all_unicode_junk + close_to_junk
    counts = Counter(all_candidates)
    high_confidence = [tid for tid, count in counts.items() if count >= 3]
    medium_confidence = [tid for tid, count in counts.items() if count == 2]

    return {
        "geometry": geometry,
        "by_method": {
            "low_norm": low_norm,
            "low_variance": low_var,
            "unreachable": unreachable,
            "unicode_junk": unicode_junk,
            "close_to_junk_reference": close_to_junk,
        },
        "combined": {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "all_candidates": list(set(all_candidates)),
        },
        "stats": {
            "vocab_size": vocab_size,
            "total_candidates": len(set(all_candidates)),
            "high_confidence_count": len(high_confidence),
            "unreachable_count": len(unreachable),
        },
    }


def get_safe_junk_tokens(
    tokenizer: PreTrainedTokenizer,
    embeddings: Tensor,
    count: int = 20,
    require_unreachable: bool = True,
) -> list[int]:
    """
    Get a list of safe-to-repurpose junk tokens.

    Prioritizes tokens that are:
    1. Unreachable via tokenizer (won't appear in normal text)
    2. Have low embedding norms (likely undertrained)
    3. Are in known junk Unicode ranges

    Args:
        tokenizer: The model's tokenizer
        embeddings: Embedding matrix
        count: Number of tokens to return
        require_unreachable: Only return unreachable tokens

    Returns:
        List of token IDs safe for repurposing
    """
    vocab_size = embeddings.shape[0]

    # Find unreachable tokens
    unreachable = set(find_unreachable_tokens(tokenizer, vocab_size, show_progress=False))

    # Find low-norm tokens
    norms = compute_embedding_norms(embeddings)
    low_norm_threshold = torch.quantile(norms, 0.05)
    low_norm = set(torch.where(norms < low_norm_threshold)[0].tolist())

    if require_unreachable:
        # Intersection: both unreachable and low-norm
        candidates = list(unreachable & low_norm)
        if len(candidates) < count:
            # Fall back to just unreachable
            candidates = list(unreachable)
    else:
        # Union: either unreachable or low-norm
        candidates = list(unreachable | low_norm)

    # Sort by norm (lowest first = most likely junk)
    candidates.sort(key=lambda tid: norms[tid].item())

    return candidates[:count]
