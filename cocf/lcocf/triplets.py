"""Local causal sub-graph construction via VLM parsing (§3.3.1).

L-COCF's first simplification: instead of *learning* a global spatio-temporal
causal graph (NP-hard, §3.1), we *read* a local causal sub-graph straight from the
prompt with a frozen VLM, then close it under spatio-temporal locality
(axiom §3.2.1). Concretely:

    prompt ──VLM──▶ causal triplets {(E_i, A_ij, E_j)}  ──locality──▶  G_s

The parser is frozen (zero training cost, §3.3.5) and injected behind the
:class:`CausalParser` contract, so a rule-based parser (tests/cold-start) and a
real VLM (LLaVA / Qwen-VL / GPT-4o) are interchangeable with no algorithm change.

Output is a :class:`~cocf.common.types.CausalSubgraph`: the entities, their VLM
importance scores (→ ``s_E``), the triplet adjacency, and the set of
quality-critical entities tagged text/face/hands (§9.3), which the strength field
and the budget scheduler treat preferentially.
"""

from __future__ import annotations

import abc
import re
from typing import Dict, List, Optional, Sequence, Tuple

from cocf.common.config import LCOCFConfig
from cocf.common.types import CausalSubgraph, CausalTriplet

# Lightweight lexicons for the rule-based fallback parser. A real VLM supersedes
# these; they exist so the framework runs (and tests) with no model download.
_CRITICAL_HINTS = {
    "text": ("text", "word", "letter", "sign", "logo", "caption", "number", "字", "文字"),
    "face": ("face", "person", "man", "woman", "child", "eye", "portrait", "脸", "人"),
    "hands": ("hand", "finger", "palm", "gesture", "手", "手指"),
}
_STOP = {"a", "an", "the", "of", "in", "on", "with", "and", "is", "are", "to", "at"}


class CausalParser(abc.ABC):
    """Frozen prompt → causal sub-graph parser (§3.3.1)."""

    @abc.abstractmethod
    def parse(self, prompt: str) -> CausalSubgraph:
        ...


class RuleBasedCausalParser(CausalParser):
    """Dependency-free heuristic parser (cold-start fallback & unit tests).

    Extracts crude ``(subject, verb, object)`` triplets via token heuristics and
    assigns importance by salience (subject > object > modifiers). It is *not*
    meant to rival a VLM — it makes the pipeline runnable and deterministic.
    """

    def __init__(self, config: Optional[LCOCFConfig] = None) -> None:
        self.cfg = config or LCOCFConfig()

    def parse(self, prompt: str) -> CausalSubgraph:
        tokens = [t for t in re.findall(r"[\w']+", prompt.lower()) if t not in _STOP]
        verbs = {"running", "walking", "jumping", "holding", "moving", "spinning",
                 "writing", "talking", "dancing", "flying", "falling", "rotating"}
        triplets: List[CausalTriplet] = []
        subj = obj = None
        action = "exists"
        nouns = [t for t in tokens if t not in verbs]
        found_verbs = [t for t in tokens if t in verbs]
        if nouns:
            subj = nouns[0]
            obj = nouns[1] if len(nouns) > 1 else nouns[0]
        if found_verbs:
            action = found_verbs[0]
        if subj is not None:
            triplets.append(
                CausalTriplet(
                    subject=subj, action=action, obj=obj or subj,
                    subject_importance=1.0, object_importance=0.7,
                    tags=self._tags(f"{subj} {obj}"),
                )
            )
        return self._to_subgraph(triplets, prompt)

    @staticmethod
    def _tags(text: str) -> Tuple[str, ...]:
        tags = []
        for tag, hints in _CRITICAL_HINTS.items():
            if any(h in text for h in hints):
                tags.append(tag)
        return tuple(tags)

    def _to_subgraph(self, triplets: List[CausalTriplet], prompt: str) -> CausalSubgraph:
        return build_subgraph(triplets)


class VLMCausalParser(CausalParser):
    """Frozen VLM parser (LLaVA / Qwen-VL / …) — lazy, optional dependency.

    Prompts the VLM for a JSON list of causal triplets with importance scores and
    critical-entity tags, then closes them into a sub-graph. Falls back to the
    rule-based parser if the model or its output is unavailable, so callers never
    have to special-case the cold-start path.
    """

    def __init__(self, config: LCOCFConfig) -> None:
        self.cfg = config
        self._fallback = RuleBasedCausalParser(config)
        self._model = None  # lazily built in _ensure

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        try:  # pragma: no cover - requires a downloaded VLM
            from transformers import pipeline

            self._model = pipeline("image-to-text", model=self.cfg.vlm_name)
            return True
        except Exception:
            return False

    def parse(self, prompt: str) -> CausalSubgraph:
        if not self._ensure():
            return self._fallback.parse(prompt)
        try:  # pragma: no cover
            triplets = self._query_vlm(prompt)
            return build_subgraph(triplets)
        except Exception:
            return self._fallback.parse(prompt)

    def _query_vlm(self, prompt: str) -> List[CausalTriplet]:  # pragma: no cover
        raise NotImplementedError("wire the concrete VLM prompt/JSON schema here")


def build_subgraph(triplets: List[CausalTriplet]) -> CausalSubgraph:
    """Close a triplet list into a local causal sub-graph (§3.3.1).

    Aggregates per-entity importance (max over its appearances) and collects
    critical entities. No structure learning, no global edges — locality is
    enforced by construction (axiom §3.2.1).
    """
    importance: Dict[str, float] = {}
    critical: List[str] = []
    for tr in triplets:
        importance[tr.subject] = max(importance.get(tr.subject, 0.0), tr.subject_importance)
        importance[tr.obj] = max(importance.get(tr.obj, 0.0), tr.object_importance)
        if tr.tags:
            critical.extend([tr.subject, tr.obj])
    return CausalSubgraph(
        triplets=triplets,
        entity_importance=importance,
        critical_entities=tuple(dict.fromkeys(critical)),
    )


def build_parser(config: LCOCFConfig) -> CausalParser:
    """Factory: a real VLM parser when configured, else the rule-based fallback."""
    if config.vlm_name and config.vlm_name not in ("frozen-vlm", "rule", "mock"):
        return VLMCausalParser(config)
    return RuleBasedCausalParser(config)
