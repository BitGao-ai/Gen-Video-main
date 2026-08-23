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
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from cocf.common.hf_clip import clip_image_embed, clip_text_embed, clip_text_inputs
from cocf.common.logging import get_logger
from cocf.common.memory import freeze
from cocf.lcocf.damage import MetricExtractor, VideoFeatures, crop_to_tube

Tensor = torch.Tensor
_log = get_logger(__name__)

# Prompt cues that mean the scene contains rendered text → OCR fidelity matters.
_TEXT_CUES = ("text", "word", "letter", "sign", "logo", "caption", "number",
              "title", "subtitle", "字", "文字", "标题")

# Frames (or frame pairs) pushed through a metric backbone in one forward. Sized
# for RAFT, whose per-pair correlation volume is ``(H/8 · W/8)²`` floats — ~156 MB
# at 480×832 — so 4 pairs peak around 0.6 GB instead of the ~7 GB a whole 49-frame
# clip would need in a single batch. Reduced from 8 for 40 GB cards.
DEFAULT_FRAME_CHUNK = 4


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
    # Build the Laplacian kernel on the video's device (not just its dtype) so a GPU
    # render does not hit a CPU-weight × CUDA-input conv2d mismatch.
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                     device=v.device, dtype=v.dtype)
    k = k.view(1, 1, 3, 3)
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

    def extract(
        self, video: Tensor, prompt: str, *,
        differentiable: bool = False,
        tube_masks: Optional[Dict[int, Tensor]] = None,
        offload: bool = True,
    ) -> VideoFeatures:
        # The fixed projection matrices live on CPU; follow the video's device so a
        # GPU render (Stage C runs the full pipeline on GPU) does not hit a CPU×GPU
        # matmul. The mock is already grad-transparent, so ``differentiable`` only needs
        # to govern device here (no ``no_grad`` to lift).
        device = video.device
        desc = _pool_frames(video, self.grid)             # [F, feat_dim]
        dino = desc @ self._w_dino.to(device)             # [F, d_dino]
        clip = desc @ self._w_clip.to(device)             # [F, d_clip]

        # CLIPScore: cosine of the mean appearance to a prompt-conditioned direction.
        # ``detach`` before the scalar cast: this field is a plain float on
        # VideoFeatures, so it leaves the graph regardless — and once Stage C's decode
        # became differentiable, the implicit cast started warning on every call.
        prompt_dir = self._prompt_direction(prompt).to(device)   # [d_clip]
        clip_mean = F.normalize(clip.mean(0), dim=-1)
        clip_text_score = float(
            ((clip_mean @ prompt_dir).clamp(-1, 1) * 0.5 + 0.5).detach()
        )

        # RAFT motion proxy: appearance change magnitude between consecutive frames.
        if desc.shape[0] >= 2:
            flow_mag = (desc[1:] - desc[:-1]).abs().mean(-1)  # [F-1]
        else:
            flow_mag = torch.zeros(0, device=video.device)

        ocr = self._ocr_fidelity(video, prompt)
        # Per-tube identity features: the same projection applied to the tube's own
        # frames/region, so a tube-group counterfactual is scored where it intervened
        # rather than on the whole clip (§7.1.1).
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
    clip_text_fn(clip_feats,prompt)->float  CLIPScore in [0,1] from the *already
                                 computed* per-frame image features (not the video —
                                 taking the video made it re-encode every frame)
    flow_fn(video)->[F-1]        per-pair mean RAFT flow magnitude
    ocr_fn(video,prompt)->float  OCR fidelity in [0,1] (1.0 if no text)
    """

    def __init__(
        self,
        dino_fn: Callable[[Tensor], Tensor],
        clip_fn: Callable[[Tensor], Tensor],
        clip_text_fn: Callable[[Tensor, str], float],  # (clip_feats, prompt)
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
        # Label/metric path: no grad, and — when the caller will consume the features
        # off-device — offload to CPU (cheap, features are detached references).
        # Stage-C accelerated branch (differentiable=True): keep the graph so the §6.3.2
        # quality loss reaches the render, and keep the input device so it composes with
        # the on-device pixel loss without a CPU×GPU mismatch.
        grad_ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_ctx:
            dino = self.dino_fn(video).float()
            clip = self.clip_fn(video).float()
            flow = self.flow_fn(video).float()
            # Scored from ``clip`` rather than from ``video``: the callable used to
            # take the clip and call the image encoder again, so every extract ran
            # CLIP over all frames twice (§P2-7). Computed here, before the CPU
            # offload below, so the text tower still sees on-device features.
            clip_text_score = float(self.clip_text_fn(clip, prompt))
            # One extra identity pass per requested tube — the price of a label that
            # is actually about that tube (§7.1.1); callers pass only what they score.
            tube_dino = {
                tid: self.dino_fn(crop_to_tube(video, m)).float()
                for tid, m in (tube_masks or {}).items()
            }
            # Offload only when asked *and* when there is no graph to keep on-device.
            # This used to key off ``differentiable`` alone, which silently broke every
            # consumer that wants the no-grad path but compares the result on the GPU —
            # Stage C's §6.3.2 reference observation being the one that matters (see the
            # ``offload`` contract on :meth:`MetricExtractor.extract`).
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
        frame_chunk: int = DEFAULT_FRAME_CHUNK,
        share_from: Optional[object] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> "ModelMetricExtractor":  # pragma: no cover - needs model downloads
        """Wire DINOv2 + CLIP + torchvision-RAFT (+ optional OCR) into callables.

        ``share_from`` takes an already-built :class:`~cocf.tubes.model_perception.
        ModelPerception` and reuses its DINOv2/CLIP weights instead of loading a
        second copy — ~1 GB of duplicate residency when Stage A builds both under
        ``--real-models`` (§P2-7). The two use the models identically (frozen, eval,
        default projections), so sharing changes no output.

        Imported lazily so this module stays import-clean without the heavy deps.
        Wrap-up only — the projections/normalisation that matter for *comparison*
        are the model defaults, applied identically to both videos being compared.

        ``frame_chunk`` bounds how many frames (or frame *pairs*, for flow) go
        through a backbone in one forward. It exists for VRAM, not throughput: RAFT
        materialises an all-pairs correlation volume of
        ``B × (H/8 · W/8)²`` floats, which for a whole 49-frame 480×832 clip is a
        single ~7 GiB tensor — larger than anything else Stage A allocates. All the
        models run in ``eval`` (so BatchNorm uses running stats) and every reduction
        here is per-frame, so chunking is numerically transparent.
        """
        import torch as _t
        import torch.nn.functional as _F
        from transformers import (  # type: ignore
            AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor,
        )

        shared = getattr(share_from, "models", None) or {}
        # Frozen, not merely ``.eval()`` — see the note in
        # :func:`cocf.tubes.model_perception.ModelPerception.from_pretrained`. This
        # extractor's *differentiable* branch (Stage C's accelerated render) is exactly
        # where an unfrozen tower costs both retained activations and a permanent fp32
        # gradient buffer. ``freeze`` is applied to shared modules too: it is idempotent
        # and the sharing path must not be the one that leaves them trainable.
        dino = freeze(shared.get("dino") or AutoModel.from_pretrained(dino_name).to(device))
        dino_proc = shared.get("dino_proc") or AutoImageProcessor.from_pretrained(dino_name)
        clip = freeze(shared.get("clip") or CLIPModel.from_pretrained(clip_name).to(device))
        clip_proc = shared.get("clip_proc") or CLIPProcessor.from_pretrained(clip_name)
        if shared:
            _log.info("ModelMetricExtractor: reusing the perception backend's "
                      "DINOv2/CLIP weights (no second copy loaded)")
        elif dtype is not None:
            # Only narrow weights we own: a shared backend already applied its own
            # dtype, and re-casting it in place would silently change the perception
            # provider's precision from under it.
            dino = dino.to(dtype)
            clip = clip.to(dtype)
        # The dtype ``_prep`` must produce, resolved **per tower**. DINOv2 and CLIP
        # are narrowed together above, but ``share_from`` hands them over as two
        # independently-built modules, so nothing guarantees they agree — and feeding
        # one tower the other's precision is the same rejection RAFT hits below
        # ("Input type (c10::BFloat16) and bias type (float) should be the same").
        # Read off the module rather than from ``dtype`` so the shared-backend path
        # is covered too, and for CLIP off the *vision* tower specifically: that is
        # the submodule these pixels reach (via ``clip_image_embed``), whereas
        # ``next(clip.parameters())`` reports whichever submodule CLIPModel happens
        # to register first. Falling back to ``clip`` matches ``clip_image_embed``,
        # which trusts ``get_image_features`` when there is no ``vision_model``.
        dino_dtype = next(dino.parameters()).dtype
        clip_dtype = next(getattr(clip, "vision_model", clip).parameters()).dtype

        def _prep(video: Tensor, proc, enc_dtype: torch.dtype) -> Tensor:
            """``[F,3,H,W]`` in [0,1] → the encoder's pixel values, on device.

            Resize + normalise **in torch**, using the processor's own constants,
            instead of handing tensors to the HF processor: that path converts to
            numpy/PIL internally, which severs the autograd graph. The §6.3.2 Stage-C
            semantic loss asks this extractor for a differentiable branch, and it was
            silently getting a detached one — the loss looked healthy and trained
            nothing through the render.
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
            # Cast last, and with ``.to`` rather than a dtype-typed literal, so the
            # normalisation itself still happens in the wider of the two dtypes and the
            # §6.3.2 autograd path through this branch stays intact.
            return ((v - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)).to(enc_dtype)

        def _per_frame(video: Tensor, fn) -> Tensor:
            """Apply a per-frame encoder over ``video`` in ``frame_chunk`` slices."""
            f = video.shape[0]
            if f == 0:
                return _t.zeros(0, device=device)
            return _t.cat([fn(video[lo:hi]) for lo, hi in _chunks(f, frame_chunk)], 0)

        def dino_fn(video: Tensor) -> Tensor:
            def _run(chunk: Tensor) -> Tensor:
                out = dino(_prep(chunk, dino_proc, dino_dtype)).last_hidden_state  # [f,T,d]
                return out.mean(1)  # CLS-pooled identity per frame
            return _per_frame(video, _run)

        def clip_fn(video: Tensor) -> Tensor:
            # clip_image_embed, not clip.get_image_features: the latter returns the
            # vision tower's token sequence [f,50,768] on newer transformers instead of
            # the pooled, projected [f,d_clip] embedding CLIP similarity is defined on
            # (see cocf.common.hf_clip). Stays differentiable for the §6.3.2 loss.
            return _per_frame(video, lambda c: clip_image_embed(clip, _prep(c, clip_proc, clip_dtype)))

        def clip_text_fn(clip_feats: Tensor, prompt: str) -> float:
            # Both operands to fp32 before the cosine. ``extract`` hands us image
            # features it has already ``.float()``-ed (§P2-7 stopped re-encoding the
            # video here), while the text tower still answers in the *weights'*
            # dtype — under ``--perception-dtype bfloat16`` that pair is a matmul
            # torch rejects outright:
            #     RuntimeError: expected mat1 and mat2 to have the same dtype,
            #                   but got: float != c10::BFloat16
            # Casting the two vectors, not the tower, keeps the weights in bf16
            # where the speed is; a [1,d_clip] dot in fp32 costs nothing measurable
            # (same seam as the RAFT cast below).
            img = F.normalize(clip_feats.to(device).float().mean(0, keepdim=True), dim=-1)
            # clip_text_inputs, not a bare clip_proc(...): the text tower has a
            # 77-token position table and rejects anything longer, so an untruncated
            # OpenVid-1M caption crashed Stage A on its first clip (see hf_clip).
            txt_in = clip_text_inputs(clip, clip_proc, [prompt], device=device)
            txt = F.normalize(clip_text_embed(clip, **txt_in).float(), dim=-1)
            return float((img @ txt.T).clamp(-1, 1).item() * 0.5 + 0.5)

        try:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            raft = freeze(raft_small(weights=Raft_Small_Weights.DEFAULT).to(device))
            # Videos decoded by a bf16/fp16 backbone must be cast to RAFT's own
            # weight dtype, or conv2d rejects the pair outright ("Input type
            # (c10::BFloat16) and bias type (float) should be the same"). Cast the
            # frames, not the module, so the caller's precision never leaks in.
            raft_dtype = next(raft.parameters()).dtype

            def flow_fn(video: Tensor) -> Tensor:
                v = (video.clamp(0, 1) * 2 - 1).to(device=device, dtype=raft_dtype)
                a, b = v[:-1], v[1:]
                if a.shape[0] == 0:
                    return _t.zeros(0, device=device)
                # Chunked over frame *pairs*: the correlation volume is the single
                # largest allocation in the whole Stage-A pass (see the docstring).
                mags = []
                for lo, hi in _chunks(a.shape[0], frame_chunk):
                    flow = raft(a[lo:hi], b[lo:hi])[-1]  # [n,2,H,W]
                    mags.append(flow.flatten(1).norm(dim=1) / flow.shape[-1])
                    del flow
                return _t.cat(mags, 0)
        except Exception:
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
