"""COCF-SS-DCA: Causal-Counterfactual Compute Field with Semantic-Tube Anchoring.

A *plug-in* acceleration layer for video diffusion transformers (DiT). It does not
generate video on its own; instead it wraps a frozen backbone (HunyuanVideo, Wan2.1,
...) and dynamically allocates compute per cross-frame *semantic tube* so that
perceptually unimportant regions are cheaply approximated while causally important
regions keep full fidelity.

The package is organised into loosely-coupled subsystems, each mapping onto one
section of the design document:

    common      infrastructure: types, config, registry, memory helpers
    backbones   backbone-agnostic adapters (the multi-model compatibility layer)
    tubes       STA  - semantic tube anchoring (§4)
    lcocf       L-COCF - lightweight counterfactual causal compute field (§3)
    raec        RAEC - revocable anchoring & error certificates (§5)
    cmsc        CMSC - cross-modal semantic conservation (§6)
    scheduler   dynamic compute-budget scheduling & action allocation (§7.3, §2.2)
    engine      end-to-end accelerated inference loop (§7.2)
    core        the top-level accelerator that wires everything together
    data        training-data pipeline incl. counterfactual teacher generation (§7.1.1)
    training    three-stage training system (§7.1)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
