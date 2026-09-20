"""Video quality-metric extraction: the perception backend for damage and CMSC."""

from __future__ import annotations

import hashlib
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from cocf.common.hf_clip import clip_image_embed, clip_text_embed, clip_text_inputs
from cocf.common.logging import get_logger
from cocf.common.memory import freeze, normal_mode
from cocf.common.raft import RAFT_MIN_EDGE, load_raft, raft_pad
from cocf.lcocf.damage import MetricExtractor, VideoFeatures, crop_to_tube

Tensor = torch.Tensor
_log = get_logger(__name__)

# Prompt cues implying rendered text in the scene (OCR fidelity matters).
_TEXT_CUES = ("text", "word", "letter", "sign", "logo", "caption", "number",
              "title", "subtitle", "字", "文字", "标题")

DEFAULT_FRAME_CHUNK = 4  # frame pairs per RAFT forward (bounds the correlation volume)
DEFAULT_VIT_CHUNK = 16  # frames per DINOv2/CLIP forward
DEFAULT_FLOW_MAX_EDGE = 448  # longest edge RAFT runs at
_TEXT_CACHE_MAX = 32  # prompt embeddings cached per text tower


def _chunks(total: int, size: int):
    """Yield ``[lo, hi)`` windows of at most ``size`` items covering ``range(total)``."""
    step = max(1, int(size))
    for lo in range(0, total, step):
        yield lo, min(lo + step, total)


def _seed_from_str(s: str, salt: int = 0) -> int:
    """Deterministic 31-bit seed from a string (stable across processes/runs)."""
    h = hashlib.sha1(f"{salt}:{s}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _pool_frames(video: Tensor, grid: int = 8) -> Tensor:
    """``[F, 3, H, W] -> [F, 3*grid*grid]`` low-res appearance descriptor (in [0,1])."""
    if video.dim() != 4:
        raise ValueError(f"expected video [F,3,H,W], got shape {tuple(video.shape)}")
    v = video.float().clamp(0.0, 1.0)
    pooled = F.adaptive_avg_pool2d(v, (grid, grid))
    return pooled.reshape(pooled.shape[0], -1)


def _high_freq_energy(video: Tensor) -> Tensor:
    """Per-frame high-frequency energy ``[F]`` (a sharpness / text-legibility proxy)."""
    v = video.float().mean(1, keepdim=True)
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                     device=v.device, dtype=v.dtype)
    k = k.view(1, 1, 3, 3)
    lap = F.conv2d(v, k, padding=1)
    return lap.abs().flatten(1).mean(1)


class MockMetricExtractor(MetricExtractor):
    """Deterministic, content-dependent stand-in for the DINO/CLIP/RAFT/OCR stack."""

    def __init__(self, d_dino: int = 64, d_clip: int = 64, grid: int = 8,
                 seed: int = 1234) -> None:
        self.d_dino = d_dino
        self.d_clip = d_clip
        self.grid = grid
        feat_dim = 3 * grid * grid
        g = torch.Generator().manual_seed(seed)
        self._w_dino = torch.randn(feat_dim, d_dino, generator=g) / (feat_dim ** 0.5)
        self._w_clip = torch.randn(feat_dim, d_clip, generator=g) / (feat_dim ** 0.5)
        self._text_basis = torch.randn(d_clip, generator=g)

    def extract(
        self, video: Tensor, prompt: str, *,
        differentiable: bool = False,
        tube_masks: Optional[Dict[int, Tensor]] = None,
        offload: bool = True,
    ) -> VideoFeatures:
        device = video.device  # follow the video's device (projections live on CPU)
        desc = _pool_frames(video, self.grid)
        dino = desc @ self._w_dino.to(device)
        clip = desc @ self._w_clip.to(device)

        prompt_dir = self._prompt_direction(prompt).to(device)
        clip_mean = F.normalize(clip.mean(0), dim=-1)
        clip_text_score = float(
            ((clip_mean @ prompt_dir).clamp(-1, 1) * 0.5 + 0.5).detach()
        )

        if desc.shape[0] >= 2:
            flow_mag = (desc[1:] - desc[:-1]).abs().mean(-1)  # motion proxy
        else:
            flow_mag = torch.zeros(0, device=video.device)

        ocr = self._ocr_fidelity(video, prompt)
        tube_dino = {
            tid: (_pool_frames(crop_to_tube(video, m), self.grid)
                  @ self._w_dino.to(device)).float()
            for tid, m in (tube_masks or {}).items()
        }
        return VideoFeatures(
            dino_per_frame=dino.float(),
            clip_per_frame=clip.float(),
            clip_text_score=clip_text_score,
            flow_mag_per_pair=flow_mag.float(),
            ocr_accuracy=ocr,
            tube_dino=tube_dino,
        )

    def _prompt_direction(self, prompt: str) -> Tensor:
        g = torch.Generator().manual_seed(_seed_from_str(prompt))
        v = torch.randn(self.d_clip, generator=g) + 0.3 * self._text_basis
        return F.normalize(v, dim=-1)

    def _ocr_fidelity(self, video: Tensor, prompt: str) -> float:
        """1.0 when no text is implied; else a sharpness-derived legibility score."""
        if not any(cue in prompt.lower() for cue in _TEXT_CUES):
            return 1.0
        energy = _high_freq_energy(video).mean()
        return float(torch.tanh(8.0 * energy).clamp(0.0, 1.0))


class ModelMetricExtractor(MetricExtractor):
    """Real perception backend assembled from injected per-feature callables.

    ``clip_text_fn`` scores from the already-computed per-frame CLIP features,
    not from the video.
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

    def extract(
        self, video: Tensor, prompt: str, *,
        differentiable: bool = False,
        tube_masks: Optional[Dict[int, Tensor]] = None,
        offload: bool = True,
    ) -> VideoFeatures:
        # differentiable=True keeps the graph and the input device for Stage C.
        grad_ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_ctx:
            dino = self.dino_fn(video).float()
            clip = self.clip_fn(video).float()
            flow = self.flow_fn(video).float()
            clip_text_score = float(self.clip_text_fn(clip, prompt))
            tube_dino = {
                tid: self.dino_fn(crop_to_tube(video, m)).float()
                for tid, m in (tube_masks or {}).items()
            }
            if offload and not differentiable:
                dino, clip, flow = dino.cpu(), clip.cpu(), flow.cpu()
                tube_dino = {tid: v.cpu() for tid, v in tube_dino.items()}
            return VideoFeatures(
                dino_per_frame=dino,
                clip_per_frame=clip,
                clip_text_score=clip_text_score,
                flow_mag_per_pair=flow,
                ocr_accuracy=float(self.ocr_fn(video, prompt)) if self.ocr_fn else 1.0,
                tube_dino=tube_dino,
            )

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        *,
        dino_name: str = "facebook/dinov2-base",
        clip_name: str = "openai/clip-vit-base-patch32",
        enable_ocr: bool = False,
        frame_chunk: int = DEFAULT_FRAME_CHUNK,
        vit_chunk: int = DEFAULT_VIT_CHUNK,
        flow_max_edge: int = DEFAULT_FLOW_MAX_EDGE,
        share_from: Optional[object] = None,
        dtype: Optional["torch.dtype"] = None,
        raft_weights: Optional[str] = None,
        require_flow: bool = False,
    ) -> "ModelMetricExtractor":  # pragma: no cover - needs model downloads
        """Build the standard DINOv2 + CLIP + RAFT (+ optional OCR) stack.

        ``share_from`` reuses an existing ModelPerception's DINOv2/CLIP weights.
        """
        import torch as _t
        import torch.nn.functional as _F
        from transformers import (  # type: ignore
            AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor,
        )

        shared = getattr(share_from, "models", None) or {}
        dino = freeze(shared.get("dino") or AutoModel.from_pretrained(dino_name).to(device))
        dino_proc = shared.get("dino_proc") or AutoImageProcessor.from_pretrained(dino_name)
        clip = freeze(shared.get("clip") or CLIPModel.from_pretrained(clip_name).to(device))
        clip_proc = shared.get("clip_proc") or CLIPProcessor.from_pretrained(clip_name)
        if shared:
            _log.info("ModelMetricExtractor: reusing the perception backend's "
                      "DINOv2/CLIP weights (no second copy loaded)")
        elif dtype is not None:
            dino = dino.to(dtype)  # only narrow weights we own
            clip = clip.to(dtype)
        dino_dtype = next(dino.parameters()).dtype
        clip_dtype = next(getattr(clip, "vision_model", clip).parameters()).dtype

        def _prep(video: Tensor, proc, enc_dtype: torch.dtype) -> Tensor:
            """``[F,3,H,W]`` in [0,1] -> the encoder's pixel values, on device.

            Resize/normalise in torch (not the HF processor) to keep the autograd
            graph intact.
            """
            v = video.clamp(0, 1).to(device)
            size = getattr(proc, "size", None) or getattr(
                getattr(proc, "image_processor", None), "size", None) or {}
            edge = size.get("shortest_edge") or size.get("height") or 224
            v = _F.interpolate(v, size=(int(edge), int(edge)),
                               mode="bicubic", align_corners=False).clamp(0, 1)
            ip = getattr(proc, "image_processor", proc)
            mean = _t.tensor(getattr(ip, "image_mean", [0.5, 0.5, 0.5]), device=device)
            std = _t.tensor(getattr(ip, "image_std", [0.5, 0.5, 0.5]), device=device)
            return ((v - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)).to(enc_dtype)

        def _per_frame(video: Tensor, fn) -> Tensor:
            """Apply a per-frame encoder over ``video`` in chunks."""
            f = video.shape[0]
            if f == 0:
                return _t.zeros(0, device=device)
            size = frame_chunk if _t.is_grad_enabled() else vit_chunk
            return _t.cat([fn(video[lo:hi]) for lo, hi in _chunks(f, size)], 0)

        def dino_fn(video: Tensor) -> Tensor:
            def _run(chunk: Tensor) -> Tensor:
                out = dino(_prep(chunk, dino_proc, dino_dtype)).last_hidden_state
                return out.mean(1)  # token-mean identity per frame
            return _per_frame(video, _run)

        def clip_fn(video: Tensor) -> Tensor:
            return _per_frame(video, lambda c: clip_image_embed(clip, _prep(c, clip_proc, clip_dtype)))

        text_cache: Dict[str, Tensor] = {}

        def _text_embed(prompt: str) -> Tensor:
            """Unit-norm fp32 prompt embedding, memoised across a clip's rollouts."""
            hit = text_cache.get(prompt)
            if hit is not None:
                return hit
            with normal_mode(), _t.no_grad():  # plain tensors, safe to cache across contexts
                txt_in = clip_text_inputs(clip, clip_proc, [prompt], device=device)
                txt = F.normalize(clip_text_embed(clip, **txt_in).float(), dim=-1)
            if len(text_cache) >= _TEXT_CACHE_MAX:
                text_cache.pop(next(iter(text_cache)))
            text_cache[prompt] = txt
            return txt

        def clip_text_fn(clip_feats: Tensor, prompt: str) -> float:
            img = F.normalize(clip_feats.to(device).float().mean(0, keepdim=True), dim=-1)
            return float((img @ _text_embed(prompt).T).clamp(-1, 1).item() * 0.5 + 0.5)

        raft = load_raft(device, variant="small", weights_path=raft_weights,
                         required=require_flow)
        if raft is not None:
            raft_dtype = next(raft.parameters()).dtype  # cast frames, not the module

            def _raft_input(video: Tensor) -> Tensor:
                """``[F,3,H,W]`` in [0,1] -> RAFT's [-1,1] input, edge-capped, on device."""
                v = (video.clamp(0, 1) * 2 - 1).to(device)
                h, w = v.shape[-2:]
                edge = max(h, w)
                if flow_max_edge and edge > flow_max_edge:
                    scale = flow_max_edge / edge
                    size = (max(RAFT_MIN_EDGE, int(h * scale) // 8 * 8),
                            max(RAFT_MIN_EDGE, int(w * scale) // 8 * 8))
                    v = _F.interpolate(v, size=size, mode="bilinear", align_corners=False)
                return raft_pad(v.to(raft_dtype))

            def flow_fn(video: Tensor) -> Tensor:
                v = _raft_input(video)
                a, b = v[:-1], v[1:]
                if a.shape[0] == 0:
                    return _t.zeros(0, device=device)
                mags = []
                for lo, hi in _chunks(a.shape[0], frame_chunk):
                    flow = raft(a[lo:hi], b[lo:hi])[-1]
                    mags.append(flow.norm(dim=1).flatten(1).mean(1))
                    del flow
                return _t.cat(mags, 0)
        else:
            def flow_fn(video: Tensor) -> Tensor:
                d = _pool_frames(video.to(device))
                return (d[1:] - d[:-1]).abs().mean(-1) if d.shape[0] >= 2 else _t.zeros(0, device=device)

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
