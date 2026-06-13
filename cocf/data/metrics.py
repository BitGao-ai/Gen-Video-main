"""Video quality-metric extraction — the perception backend for damage & CMSC.

A :class:`~cocf.lcocf.damage.MetricExtractor` turns a decoded video into the
compact :class:`~cocf.lcocf.damage.VideoFeatures` bundle (DINO identity, CLIP
appearance, RAFT motion, OCR fidelity) that two subsystems consume:

    * the L-COCF teacher labels (§7.1.1) — damage = degradation of these features
      in the counterfactual video vs the full-compute reference;
    * the CMSC conservation loss (§6.3.2) — deviation of these features between the
      full and accelerated videos.

Both consumers compare *two* :class:`VideoFeatures`, so the only hard requirement
on an extractor is **determinism and self-consistency**: the same video must map
to the same features, and the projections must be identical across the videos
being compared. That is exactly what lets a cheap mock stand in for the real
DINOv2/CLIP/RAFT/OCR stack in tests and CPU demos.

This module provides:

    MockMetricExtractor   deterministic, content-dependent, dependency-free — the
                          features react to freezing/blurring (what skip actions do)
                          so counterfactual damage is non-trivially positive.
    ModelMetricExtractor  dependency-injected real backend: you supply (or lazily
                          build) the DINOv2/CLIP/RAFT/OCR callables; the feature
                          assembly is shared.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from cocf.lcocf.damage import MetricExtractor, VideoFeatures

Tensor = torch.Tensor

# Prompt cues that mean the scene contains rendered text → OCR fidelity matters.
_TEXT_CUES = ("text", "word", "letter", "sign", "logo", "caption", "number",
              "title", "subtitle", "字", "文字", "标题")


def _seed_from_str(s: str, salt: int = 0) -> int:
    """Deterministic 31-bit seed from a string (stable across processes/runs)."""
    h = hashlib.sha1(f"{salt}:{s}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _pool_frames(video: Tensor, grid: int = 8) -> Tensor:
    """``[F, 3, H, W] → [F, 3*grid*grid]`` low-res appearance descriptor (in [0,1])."""
    if video.dim() != 4:
        raise ValueError(f"expected video [F,3,H,W], got shape {tuple(video.shape)}")
    v = video.float().clamp(0.0, 1.0)
    pooled = F.adaptive_avg_pool2d(v, (grid, grid))  # [F,3,g,g]
    return pooled.reshape(pooled.shape[0], -1)        # [F, 3*g*g]


def _high_freq_energy(video: Tensor) -> Tensor:
    """Per-frame high-frequency energy ``[F]`` (a sharpness / text-legibility proxy).

    Skip actions (freeze/interpolate) blur high-frequency detail, so a drop here is
    the signal behind the mock's OCR-fidelity degradation.
    """
    v = video.float().mean(1, keepdim=True)  # [F,1,H,W] luminance
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    k = k.view(1, 1, 3, 3).to(v.dtype)
    lap = F.conv2d(v, k, padding=1)
    return lap.abs().flatten(1).mean(1)  # [F]


class MockMetricExtractor(MetricExtractor):
    """Deterministic, content-dependent stand-in for the DINO/CLIP/RAFT/OCR stack.

    Features are fixed linear projections of a low-res frame descriptor, so they are
    reproducible and *react to video content*: a frozen or interpolated frame yields
    a near-duplicate descriptor (low flicker / low flow), a blurred frame loses
    high-frequency energy (lower OCR). That makes the counterfactual damage signal
    meaningful end-to-end on CPU without any model download.
    """

    def __init__(self, d_dino: int = 64, d_clip: int = 64, grid: int = 8,
                 seed: int = 1234) -> None:
        self.d_dino = d_dino
        self.d_clip = d_clip
        self.grid = grid
        feat_dim = 3 * grid * grid
        g = torch.Generator().manual_seed(seed)
        # Fixed projection matrices (the "frozen perception model" weights).
        self._w_dino = torch.randn(feat_dim, d_dino, generator=g) / (feat_dim ** 0.5)
        self._w_clip = torch.randn(feat_dim, d_clip, generator=g) / (feat_dim ** 0.5)
        self._text_basis = torch.randn(d_clip, generator=g)

    def extract(self, video: Tensor, prompt: str) -> VideoFeatures:
        desc = _pool_frames(video, self.grid)            # [F, feat_dim]
        dino = desc @ self._w_dino                        # [F, d_dino]
        clip = desc @ self._w_clip                        # [F, d_clip]

        # CLIPScore: cosine of the mean appearance to a prompt-conditioned direction.
        prompt_dir = self._prompt_direction(prompt)       # [d_clip]
        clip_mean = F.normalize(clip.mean(0), dim=-1)
        clip_text_score = float((clip_mean @ prompt_dir).clamp(-1, 1) * 0.5 + 0.5)

        # RAFT motion proxy: appearance change magnitude between consecutive frames.
        if desc.shape[0] >= 2:
            flow_mag = (desc[1:] - desc[:-1]).abs().mean(-1)  # [F-1]
        else:
            flow_mag = torch.zeros(0)

        ocr = self._ocr_fidelity(video, prompt)
        return VideoFeatures(
            dino_per_frame=dino.float(),
            clip_per_frame=clip.float(),
            clip_text_score=clip_text_score,
            flow_mag_per_pair=flow_mag.float(),
            ocr_accuracy=ocr,
        )

    # -- pieces ---------------------------------------------------------- #

    def _prompt_direction(self, prompt: str) -> Tensor:
        g = torch.Generator().manual_seed(_seed_from_str(prompt))
        v = torch.randn(self.d_clip, generator=g) + 0.3 * self._text_basis
        return F.normalize(v, dim=-1)

    def _ocr_fidelity(self, video: Tensor, prompt: str) -> float:
        """1.0 when no text is implied; else a sharpness-derived legibility score."""
        if not any(cue in prompt.lower() for cue in _TEXT_CUES):
            return 1.0
        energy = _high_freq_energy(video).mean()
        # Map sharpness to [0,1] with a soft saturating curve; blurred → lower OCR.
        return float(torch.tanh(8.0 * energy).clamp(0.0, 1.0))


class ModelMetricExtractor(MetricExtractor):
    """Real perception backend, assembled from injected feature callables.

    Each callable is optional and dependency-injected, so this class wires the
    *assembly* (the :class:`VideoFeatures` contract) without hard-coding any model.
    Supply your own, or use :meth:`from_pretrained` to lazily build the standard
    DINOv2 + CLIP + RAFT + OCR stack.

    Parameters
    ----------
    dino_fn(video)->[F,d]        per-frame subject/identity features (DINOv2)
    clip_fn(video)->[F,d]        per-frame appearance features (CLIP image encoder)
    clip_text_fn(video,prompt)->float   CLIPScore in [0,1]
    flow_fn(video)->[F-1]        per-pair mean RAFT flow magnitude
    ocr_fn(video,prompt)->float  OCR fidelity in [0,1] (1.0 if no text)
    """

    def __init__(
        self,
        dino_fn: Callable[[Tensor], Tensor],
        clip_fn: Callable[[Tensor], Tensor],
        clip_text_fn: Callable[[Tensor, str], float],
        flow_fn: Callable[[Tensor], Tensor],
        ocr_fn: Optional[Callable[[Tensor, str], float]] = None,
    ) -> None:
        self.dino_fn = dino_fn
        self.clip_fn = clip_fn
        self.clip_text_fn = clip_text_fn
        self.flow_fn = flow_fn
        self.ocr_fn = ocr_fn

    @torch.no_grad()
    def extract(self, video: Tensor, prompt: str) -> VideoFeatures:
        return VideoFeatures(
            dino_per_frame=self.dino_fn(video).float().cpu(),
            clip_per_frame=self.clip_fn(video).float().cpu(),
            clip_text_score=float(self.clip_text_fn(video, prompt)),
            flow_mag_per_pair=self.flow_fn(video).float().cpu(),
            ocr_accuracy=float(self.ocr_fn(video, prompt)) if self.ocr_fn else 1.0,
        )

    # ------------------------------------------------------------------ #
    # optional: build the standard stack lazily (requires the deps installed)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        *,
        dino_name: str = "facebook/dinov2-base",
        clip_name: str = "openai/clip-vit-base-patch32",
        enable_ocr: bool = False,
    ) -> "ModelMetricExtractor":  # pragma: no cover - needs model downloads
        """Wire DINOv2 + CLIP + torchvision-RAFT (+ optional OCR) into callables.

        Imported lazily so this module stays import-clean without the heavy deps.
        Wrap-up only — the projections/normalisation that matter for *comparison*
        are the model defaults, applied identically to both videos being compared.
        """
        import torch as _t
        from transformers import (  # type: ignore
            AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor,
        )

        dino = AutoModel.from_pretrained(dino_name).to(device).eval()
        dino_proc = AutoImageProcessor.from_pretrained(dino_name)
        clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
        clip_proc = CLIPProcessor.from_pretrained(clip_name)

        def _prep(video: Tensor, proc) -> Tensor:
            # video [F,3,H,W] in [0,1] → processor pixel values on device
            imgs = [f for f in (video.clamp(0, 1))]
            return proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)

        def dino_fn(video: Tensor) -> Tensor:
            out = dino(_prep(video, dino_proc)).last_hidden_state  # [F,T,d]
            return out.mean(1)  # CLS-pooled identity per frame

        def clip_fn(video: Tensor) -> Tensor:
            return clip.get_image_features(_prep(video, clip_proc))

        def clip_text_fn(video: Tensor, prompt: str) -> float:
            img = F.normalize(clip_fn(video).mean(0, keepdim=True), dim=-1)
            txt_in = clip_proc(text=[prompt], return_tensors="pt",
                               padding=True).to(device)
            txt = F.normalize(clip.get_text_features(**txt_in), dim=-1)
            return float((img @ txt.T).clamp(-1, 1).item() * 0.5 + 0.5)

        try:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

            def flow_fn(video: Tensor) -> Tensor:
                v = (video.clamp(0, 1) * 2 - 1).to(device)
                a, b = v[:-1], v[1:]
                if a.shape[0] == 0:
                    return _t.zeros(0)
                flow = raft(a, b)[-1]  # [F-1,2,H,W]
                return flow.flatten(1).norm(dim=1) / flow.shape[-1]
        except Exception:
            def flow_fn(video: Tensor) -> Tensor:
                d = _pool_frames(video.to(device))
                return (d[1:] - d[:-1]).abs().mean(-1) if d.shape[0] >= 2 else _t.zeros(0)

        ocr_fn = None
        if enable_ocr:
            import easyocr  # type: ignore
            reader = easyocr.Reader(["en"])

            def ocr_fn(video: Tensor, prompt: str) -> float:  # noqa: E306
                if not any(cue in prompt.lower() for cue in _TEXT_CUES):
                    return 1.0
                frame = (video[len(video) // 2].permute(1, 2, 0) * 255).byte().cpu().numpy()
                dets = reader.readtext(frame)
                conf = sum(d[2] for d in dets) / max(1, len(dets))
                return float(min(max(conf, 0.0), 1.0))

        return cls(dino_fn, clip_fn, clip_text_fn, flow_fn, ocr_fn)
