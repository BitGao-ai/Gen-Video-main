"""Local causal sub-graph construction via parsing."""

from __future__ import annotations

import abc
import re
from typing import Dict, List, Optional, Sequence, Tuple

from cocf.common.config import LCOCFConfig
from cocf.common.types import CausalSubgraph, CausalTriplet

_CRITICAL_HINTS = {
    "text": ("text", "word", "letter", "sign", "logo", "caption", "number", "字", "文字"),
    "face": ("face", "person", "man", "woman", "child", "eye", "portrait", "脸", "人"),
    "hands": ("hand", "finger", "palm", "gesture", "手", "手指"),
}
_STOP = {"a", "an", "the", "of", "in", "on", "with", "and", "is", "are", "to", "at"}


class CausalParser(abc.ABC):
    """Frozen prompt to causal sub-graph parser."""

    @abc.abstractmethod
    def parse(self, prompt: str) -> CausalSubgraph:
        """Parse prompt into causal sub-graph."""
        ...


class RuleBasedCausalParser(CausalParser):
    """Heuristic parser fallback for tests."""

    def __init__(self, config: Optional[LCOCFConfig] = None) -> None:
        """Store config."""
        self.cfg = config or LCOCFConfig()

    def parse(self, prompt: str) -> CausalSubgraph:
        """Parse prompt with token heuristics."""
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
        """Critical entity tags in text."""
        tags = []
        for tag, hints in _CRITICAL_HINTS.items():
            if any(h in text for h in hints):
                tags.append(tag)
        return tuple(tags)

    def _to_subgraph(self, triplets: List[CausalTriplet], prompt: str) -> CausalSubgraph:
        """Wrap triplets into sub-graph."""
        return build_subgraph(triplets)


class VLMCausalParser(CausalParser):
    """Frozen VLM parser with rule fallback."""

    def __init__(self, config: LCOCFConfig) -> None:
        """Store config and fallback parser."""
        self.cfg = config
        self._fallback = RuleBasedCausalParser(config)
        self._model = None

    def _ensure(self) -> bool:
        """Load VLM if available."""
        if self._model is not None:
            return True
        try:  # pragma: no cover - requires a downloaded VLM
            from transformers import pipeline

            self._model = pipeline("image-to-text", model=self.cfg.vlm_name)
            return True
        except Exception:
            return False

    def parse(self, prompt: str) -> CausalSubgraph:
        """Parse prompt via VLM or fallback."""
        if not self._ensure():
            return self._fallback.parse(prompt)
        try:  # pragma: no cover
            triplets = self._query_vlm(prompt)
            return build_subgraph(triplets)
        except Exception:
            return self._fallback.parse(prompt)

    def _query_vlm(self, prompt: str) -> List[CausalTriplet]:  # pragma: no cover
        """Query VLM for triplets."""
        raise NotImplementedError("wire the concrete VLM prompt/JSON schema here")


def build_subgraph(triplets: List[CausalTriplet]) -> CausalSubgraph:
    """Close triplet list into sub-graph."""
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
    """Build VLM or rule-based parser."""
    if config.vlm_name and config.vlm_name not in ("frozen-vlm", "rule", "mock"):
        return VLMCausalParser(config)
    return RuleBasedCausalParser(config)
