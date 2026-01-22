"""
Carrier Wave Injection Experiment.

Tests whether junk tokens can be repurposed as semantic anchors that
the model responds to as intended signals.

Protocol:
1. Select 10-20 junk tokens from unreachable/undertrained pool
2. Inject semantic anchors using mean-pooled embeddings from concept phrases
3. Test whether models respond to these repurposed tokens as intended
4. Measure via targeted prompts that include the carrier tokens
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from tqdm import tqdm

from ..harness import EmbeddingExperimentHarness
from ..token_analysis import get_safe_junk_tokens, find_unreachable_tokens


@dataclass
class CarrierToken:
    """A junk token repurposed as a semantic carrier."""

    token_id: int
    original_str: str
    semantic_anchor: str
    carrier_vector: Tensor
    placeholder: str  # What to use in prompts to reference this token


@dataclass
class CarrierResponse:
    """Model response to a carrier token in context."""

    carrier_id: int
    prompt: str
    expected_behavior: str
    generated_text: str
    alignment_score: float  # 0-1, how well response aligns with expected
    logprob_change: float  # Change in logprobs for expected tokens


@dataclass
class CarrierInjectionResult:
    """Results from testing a set of carrier tokens."""

    carriers: list[CarrierToken]
    responses: list[CarrierResponse]
    overall_alignment: float
    successful_carriers: int
    total_carriers: int


class CarrierInjectionExperiment:
    """
    Experiment to test semantic carrier token injection.

    Tests whether repurposing junk tokens with semantic embeddings
    allows using them as conceptual markers in prompts.
    """

    # Default semantic anchors to test
    DEFAULT_ANCHORS = [
        ("uncertainty", "this indicates uncertainty and doubt"),
        ("reasoning", "begin structured logical reasoning"),
        ("conclusion", "now provide the final conclusion"),
        ("example", "here is a concrete example"),
        ("caution", "proceed with caution and care"),
        ("creativity", "be creative and imaginative"),
        ("precision", "be precise and exact"),
        ("summary", "provide a brief summary"),
        ("question", "this is asking a question"),
        ("emphasis", "this point is very important"),
    ]

    def __init__(
        self,
        harness: EmbeddingExperimentHarness,
        semantic_anchors: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        """
        Initialize the experiment.

        Args:
            harness: Initialized EmbeddingExperimentHarness
            semantic_anchors: List of (name, anchor_text) tuples
        """
        self.harness = harness
        self.semantic_anchors = semantic_anchors or self.DEFAULT_ANCHORS
        self.carriers: list[CarrierToken] = []

    def setup_carriers(
        self,
        num_carriers: Optional[int] = None,
        junk_token_ids: Optional[list[int]] = None,
    ) -> list[CarrierToken]:
        """
        Set up carrier tokens by injecting semantic anchors.

        Args:
            num_carriers: Number of carriers to create (defaults to anchor count)
            junk_token_ids: Specific junk tokens to use (finds if None)

        Returns:
            List of created CarrierToken objects
        """
        if num_carriers is None:
            num_carriers = len(self.semantic_anchors)

        # Get junk tokens if not provided
        if junk_token_ids is None:
            embeddings = self.harness._get_embedding_weight()
            junk_token_ids = get_safe_junk_tokens(
                self.harness.tokenizer,
                embeddings,
                count=num_carriers,
            )

        if len(junk_token_ids) < num_carriers:
            num_carriers = len(junk_token_ids)

        self.carriers = []
        for i in range(min(num_carriers, len(self.semantic_anchors))):
            token_id = junk_token_ids[i]
            name, anchor_text = self.semantic_anchors[i]

            # Get original token string
            try:
                original_str = self.harness.tokenizer.decode([token_id])
            except Exception:
                original_str = f"<token_{token_id}>"

            # Inject carrier
            carrier_vec = self.harness.inject_carrier_token(
                token_id=token_id,
                semantic_anchor_text=anchor_text,
                pooling="mean",
                normalize=True,
            )

            # Create a placeholder string to use in prompts
            # We'll use the actual token by ID
            placeholder = f"[{name.upper()}]"

            self.carriers.append(CarrierToken(
                token_id=token_id,
                original_str=original_str,
                semantic_anchor=anchor_text,
                carrier_vector=carrier_vec,
                placeholder=placeholder,
            ))

        return self.carriers

    def _inject_carrier_in_prompt(
        self,
        prompt_template: str,
        carrier: CarrierToken,
    ) -> tuple[str, Tensor]:
        """
        Create a prompt with carrier token inserted.

        Returns the prompt string and input_ids with carrier injected.
        """
        # Tokenize template with placeholder
        placeholder = carrier.placeholder
        if placeholder not in prompt_template:
            # Append carrier at end if no placeholder
            prompt_template = f"{prompt_template} {placeholder}"

        # Split around placeholder
        parts = prompt_template.split(placeholder)

        # Tokenize parts and insert carrier token
        all_ids = []
        for i, part in enumerate(parts):
            if part:
                ids = self.harness.tokenizer.encode(part, add_special_tokens=(i == 0))
                all_ids.extend(ids)
            if i < len(parts) - 1:
                # Insert carrier token
                all_ids.append(carrier.token_id)

        input_ids = torch.tensor([all_ids], device=self.harness.device)
        full_prompt = self.harness.tokenizer.decode(all_ids)

        return full_prompt, input_ids

    def test_carrier_response(
        self,
        carrier: CarrierToken,
        prompt_template: str,
        expected_keywords: list[str],
        max_new_tokens: int = 50,
    ) -> CarrierResponse:
        """
        Test how the model responds to a carrier token in context.

        Args:
            carrier: CarrierToken to test
            prompt_template: Prompt with [PLACEHOLDER] for carrier
            expected_keywords: Keywords expected in response
            max_new_tokens: Max tokens to generate

        Returns:
            CarrierResponse with alignment analysis
        """
        full_prompt, input_ids = self._inject_carrier_in_prompt(
            prompt_template, carrier
        )

        # Generate response
        with torch.no_grad():
            outputs = self.harness.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.harness.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.harness.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Compute alignment score based on keyword presence
        generated_lower = generated_text.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in generated_lower)
        alignment = matches / len(expected_keywords) if expected_keywords else 0.0

        # Compute logprob change (simplified - compare with/without carrier)
        # For now, use alignment as proxy
        logprob_change = alignment - 0.5  # Centered around neutral

        return CarrierResponse(
            carrier_id=carrier.token_id,
            prompt=full_prompt,
            expected_behavior=carrier.semantic_anchor,
            generated_text=generated_text,
            alignment_score=alignment,
            logprob_change=logprob_change,
        )

    def run_carrier_tests(
        self,
        test_prompts: Optional[dict[str, tuple[str, list[str]]]] = None,
        show_progress: bool = True,
    ) -> CarrierInjectionResult:
        """
        Run tests for all carrier tokens.

        Args:
            test_prompts: Dict mapping anchor name to (prompt_template, expected_keywords)
            show_progress: Whether to show progress bar

        Returns:
            CarrierInjectionResult with all responses
        """
        if not self.carriers:
            self.setup_carriers()

        # Default test prompts for each anchor type
        if test_prompts is None:
            test_prompts = {
                "uncertainty": (
                    "The answer to this question [UNCERTAINTY] is",
                    ["maybe", "perhaps", "might", "could", "uncertain", "not sure"],
                ),
                "reasoning": (
                    "Let's solve this problem. [REASONING] First",
                    ["step", "because", "therefore", "since", "consider", "analyze"],
                ),
                "conclusion": (
                    "After considering all factors, [CONCLUSION]",
                    ["therefore", "thus", "conclude", "final", "summary", "result"],
                ),
                "example": (
                    "To illustrate this concept, [EXAMPLE]",
                    ["for instance", "such as", "like", "example", "consider"],
                ),
                "caution": (
                    "When approaching this task, [CAUTION]",
                    ["careful", "caution", "warning", "note", "important", "risk"],
                ),
                "creativity": (
                    "Imagine a world where [CREATIVITY]",
                    ["imagine", "creative", "unique", "novel", "innovative"],
                ),
                "precision": (
                    "The exact measurement [PRECISION] is",
                    ["exactly", "precisely", "specific", "accurate", "exact"],
                ),
                "summary": (
                    "The main points are: [SUMMARY]",
                    ["briefly", "summary", "key", "main", "overview", "essential"],
                ),
                "question": (
                    "I need to understand [QUESTION]",
                    ["what", "how", "why", "when", "which", "?"],
                ),
                "emphasis": (
                    "This point [EMPHASIS] matters because",
                    ["important", "crucial", "critical", "essential", "significant"],
                ),
            }

        responses = []
        iterator = self.carriers
        if show_progress:
            iterator = tqdm(self.carriers, desc="Testing carriers")

        for carrier in iterator:
            # Find matching test prompt
            anchor_name = carrier.semantic_anchor.split()[0]  # Get first word
            for name, (prompt_template, keywords) in test_prompts.items():
                if name in carrier.semantic_anchor.lower() or anchor_name in name:
                    response = self.test_carrier_response(
                        carrier, prompt_template, keywords
                    )
                    responses.append(response)
                    break
            else:
                # Use generic test if no match
                response = self.test_carrier_response(
                    carrier,
                    f"The meaning of this {carrier.placeholder} is",
                    carrier.semantic_anchor.split()[:3],
                )
                responses.append(response)

        # Compute overall metrics
        successful = sum(1 for r in responses if r.alignment_score > 0.3)
        overall = sum(r.alignment_score for r in responses) / len(responses) if responses else 0

        return CarrierInjectionResult(
            carriers=self.carriers,
            responses=responses,
            overall_alignment=overall,
            successful_carriers=successful,
            total_carriers=len(self.carriers),
        )

    def cleanup(self) -> None:
        """Restore original embeddings and clear carriers."""
        self.harness.restore()
        self.carriers.clear()


def test_carrier_responses(
    model_id: str = "google/gemma-3-1b-it",
    num_carriers: int = 10,
    semantic_anchors: Optional[list[tuple[str, str]]] = None,
    output_path: Optional[str] = None,
) -> CarrierInjectionResult:
    """
    Run complete carrier injection test.

    Convenience function that sets up harness and runs experiment.

    Args:
        model_id: HuggingFace model identifier
        num_carriers: Number of carrier tokens to test
        semantic_anchors: Custom semantic anchors
        output_path: Optional path to save results JSON

    Returns:
        CarrierInjectionResult with test results
    """
    harness = EmbeddingExperimentHarness(model_id)
    experiment = CarrierInjectionExperiment(harness, semantic_anchors)

    try:
        experiment.setup_carriers(num_carriers)
        results = experiment.run_carrier_tests()

        if output_path:
            output = {
                "model_id": model_id,
                "num_carriers": results.total_carriers,
                "successful": results.successful_carriers,
                "overall_alignment": results.overall_alignment,
                "carriers": [
                    {
                        "token_id": c.token_id,
                        "original": c.original_str,
                        "anchor": c.semantic_anchor,
                    }
                    for c in results.carriers
                ],
                "responses": [
                    {
                        "carrier_id": r.carrier_id,
                        "prompt": r.prompt,
                        "generated": r.generated_text,
                        "alignment": r.alignment_score,
                    }
                    for r in results.responses
                ],
            }
            Path(output_path).write_text(json.dumps(output, indent=2))

        return results

    finally:
        experiment.cleanup()
