"""
Experimental protocols for Gemma embedding manipulation.

This module provides ready-to-use experiments implementing the three
main hypotheses from the embedding manipulation research:

1. Junk Token Suppression: Test whether suppressing 1-3% of vocabulary
   has negligible impact on model performance.

2. Layer-wise Gating: Investigate how gating specific tokens at different
   layers affects attention entropy and model outputs.

3. Carrier Wave Injection: Repurpose junk tokens as semantic anchors
   and test whether models respond to these signals.
"""

from .junk_suppression import (
    JunkSuppressionExperiment,
    run_suppression_benchmark,
)
from .layer_gating import (
    LayerGatingExperiment,
    analyze_gating_effects,
)
from .carrier_injection import (
    CarrierInjectionExperiment,
    test_carrier_responses,
)

__all__ = [
    "JunkSuppressionExperiment",
    "run_suppression_benchmark",
    "LayerGatingExperiment",
    "analyze_gating_effects",
    "CarrierInjectionExperiment",
    "test_carrier_responses",
]
