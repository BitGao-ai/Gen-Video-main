"""Real perception backend — SAM + DINOv2 + CLIP + RAFT behind the STA contract.

:class:`~cocf.tubes.regions.PerceptionProvider` is the single seam the whole STA
subsystem depends on (§4.3.1). :class:`~cocf.tubes.mock_perception.MockPerception`
implements it with synthetic blobs so the pipeline is CPU/dependency-free in tests;
this module implements the *same five methods* with the real checkpoints, so a
production Stage-A run (``scripts/data/generate_counterfactual_data.py --real-models``)
segments the *actual* semantic objects the teacher must intervene on — without which
the counterfactual damage labels describe a fake object set.

Mirroring :class:`~cocf.data.metrics.ModelMetricExtractor`, the class is built two
ways:

    * **dependency-injected** ``__init__`` — takes plain callables, so it is unit-
      testable with fakes and this module imports with only ``torch`` present (the
      heavy ``transformers``/``torchvision`` deps are *never* imported at module
      load, keeping the local mock stack importable on a box without them);
    * :meth:`from_pretrained` — lazily wires the standard
      SAM(mask-generation) + DINOv2 + CLIP + RAFT stack into those callables.

The five methods only ever feed **cosine** comparisons downstream (region affinity
in :mod:`cocf.tubes.affinity`, identity confidence in :mod:`cocf.tubes.state`) or a
*constructor-adaptive* projection (CMSC ``visual_dim`` is probed from ``d_clip`` in
:meth:`cocf.core.accelerator.Accelerator._probe_visual_dim`), so the real feature
widths (DINOv2-base ⇒ ``d_id=768``, CLIP ViT-B/32 ⇒ ``d_clip=512``) drop in with no
downstream dimension change versus the mock's 64.

All outputs follow the *input frame's* device (the device convention: a provider's
masks/features live where its frame lives), so the whole tube path stays on one
device even though the models are pinned to ``device`` at load time.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from cocf.tubes.regions import PerceptionProvider

Tensor = torch.Tensor


def raft_to_framework_flow(flow: Tensor) -> Tensor:
    """torchvision RAFT ``(dx, dy)`` → this framework's ``(dy, dx)`` channel order.

    torchvision returns horizontal displacement in channel 0; every consumer here
    reads channel 0 as the *vertical* one (see
    :meth:`cocf.tubes.regions.PerceptionProvider.optical_flow`). Handing the raw
    output through therefore transposed every warp: occlusion, motion phase, the
    affinity flow kernel and the warped-mask IoU were all computed from a
    displacement rotated 90°, and on a non-square frame the down-sampling in
    :meth:`TubeBuilder._downsample_flow` scaled each axis by the wrong extent on top
    of that. Silent under the mock, which drives both channels identically (§P1-11).
    """
    if flow.shape[0] != 2:
        raise ValueError(f"expected a [2, H, W] flow field, got {tuple(flow.shape)}")
    return flow.flip(0)


Tensor = torch.Tensor

# Type aliases for the injected callables (documentation only).
SegmentFn = Callable[[Tensor], Tensor]              # frame[3,Hp,Wp] -> masks[R,Hp,Wp] bool
FeatureFn = Callable[[Tensor, Tensor], Tensor]      # (frame, mask) -> [d]
TextFn = Callable[[str], Tensor]                    # prompt -> [d_clip]
FlowFn = Callable[[Tensor, Tensor], Tensor]         # (frame_a, frame_b) -> [2,Hp,Wp]


class ModelPerception(PerceptionProvider):
    """Real SAM/DINOv2/CLIP/RAFT perception, assembled from injected callables.

    Each callable is dependency-injected so the *assembly* (the
    :class:`PerceptionProvider` contract, the empty-mask guards, the device
    discipline) is testable without any model download; :meth:`from_pretrained`
    supplies the standard stack.

    Parameters
    ----------
    segment_fn(frame[3,Hp,Wp] in [0,1]) -> masks[R,Hp,Wp] bool
        Instance masks at the frame's pixel resolution (SAM automatic masks).
    identity_fn(frame, mask) -> [d_id]
        Pooled DINOv2 identity embedding of the masked region crop.
    clip_image_fn(frame, mask) -> [d_clip]
        CLIP visual embedding of the masked region crop. Shared by
        :meth:`clip_feature` and :meth:`clip_score` (same source ⇒ no train/serve skew).
    clip_text_fn(prompt) -> [d_clip]
        CLIP text embedding of the prompt (for the semantic filter score).
    flow_fn(frame_a, frame_b) -> [2,Hp,Wp]
        Dense RAFT optical flow mapping ``frame_a`` pixels to ``frame_b``.
    d_id, d_clip
        Feature widths, exposed as attributes so consumers that must size a buffer
        (``tube_clip_embed``) or a projection (CMSC ``visual_dim``) can read them.
    """

    def __init__(
        self,
        segment_fn: SegmentFn,
        identity_fn: FeatureFn,
        clip_image_fn: FeatureFn,
        clip_text_fn: TextFn,
        flow_fn: FlowFn,
        *,
        d_id: int,
        d_clip: int,
        device: str | torch.device = "cpu",
        batch_fn: Optional[Callable] = None,
    ) -> None:
        self._segment_fn = segment_fn
        self._identity_fn = identity_fn
        self._clip_image_fn = clip_image_fn
        self._clip_text_fn = clip_text_fn
        self._flow_fn = flow_fn
        # (frame, [mask]) -> ([identity], [clip]); enables the batched path below.
        self._batch_fn = batch_fn
        self.d_id = int(d_id)
        self.d_clip = int(d_clip)
        self.device = torch.device(device)

    # -- the five PerceptionProvider methods ---------------------------- #

    def segment(self, frame: Tensor) -> Tensor:
        """RGB frame ``[3, Hp, Wp]`` → instance masks ``[R, Hp, Wp]`` (bool)."""
        return self._segment_fn(frame)

    def identity_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """Pooled DINOv2 identity embedding ``[d_id]`` of the masked region.

        An empty mask (no pixels) has no crop to embed, so it degrades to a zero
        vector — cosine identity with anything else is then 0.5, i.e. "unknown",
        which the matcher gates out rather than mis-linking.
        """
        if not self._nonempty(mask):
            return torch.zeros(self.d_id, device=frame.device)
        return self._identity_fn(frame, mask).detach().float().to(frame.device)

    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        """CLIP image-text match in ``[0, 1]`` for the region vs the prompt.

        ``cosine(clip_img, clip_txt)·0.5 + 0.5`` — the same ``[-1,1]→[0,1]`` map
        :class:`~cocf.data.metrics.ModelMetricExtractor` uses for CLIPScore. Feeds
        the §4.3.1 semantic filter (drop regions below ``TubeConfig.min_clip_score``).
        """
        if not self._nonempty(mask):
            return 0.0
        img = F.normalize(self._clip_image_fn(frame, mask).float().flatten(), dim=0)
        txt = self._clip_text_fn(prompt).float().flatten().to(img.device)
        txt = F.normalize(txt, dim=0)
        return float((img @ txt).clamp(-1.0, 1.0) * 0.5 + 0.5)

    def clip_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """CLIP visual embedding ``[d_clip]`` of the region (for text-tube alignment)."""
        if not self._nonempty(mask):
            return torch.zeros(self.d_clip, device=frame.device)
        return self._clip_image_fn(frame, mask).detach().float().to(frame.device)

    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """RAFT flow ``[2, Hp, Wp]`` mapping ``frame_a`` pixels to ``frame_b``."""
        return self._flow_fn(frame_a, frame_b).to(frame_a.device)

    def region_features(self, frame: Tensor, masks):
        """All of a frame's regions in **one** DINOv2 + one CLIP forward (§P2-8).

        Uses the batched callables when :meth:`from_pretrained` supplied them; falls
        back to the per-region default otherwise, so a hand-injected instance keeps
        working. Empty masks are excluded from the batch and filled with zeros, which
        is what the per-region methods do.
        """
        masks = list(masks)
        if self._batch_fn is None or not masks:
            return super().region_features(frame, masks)
        keep = [i for i, m in enumerate(masks) if self._nonempty(m)]
        ident = [torch.zeros(self.d_id, device=frame.device) for _ in masks]
        textf = [torch.zeros(self.d_clip, device=frame.device) for _ in masks]
        if not keep:
            return ident, textf
        d_feats, c_feats = self._batch_fn(frame, [masks[i] for i in keep])
        for j, i in enumerate(keep):
            ident[i] = d_feats[j].detach().float().to(frame.device)
            textf[i] = c_feats[j].detach().float().to(frame.device)
        return ident, textf

    # -- helpers -------------------------------------------------------- #

    @staticmethod
    def _nonempty(mask: Tensor) -> bool:
        """True when the mask selects at least one pixel (guards degenerate crops)."""
        return bool(mask.any())

    # ------------------------------------------------------------------ #
    # build the standard SAM + DINOv2 + CLIP + RAFT stack lazily
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        *,
        sam_model: str = "facebook/sam-vit-base",
        dino_name: str = "facebook/dinov2-base",
        clip_name: str = "openai/clip-vit-base-patch32",
        points_per_side: int = 16,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        max_masks: int = 24,
    ) -> "ModelPerception":  # pragma: no cover - needs model downloads
        """Wire SAM(mask-generation) + DINOv2 + CLIP + RAFT into the five callables.

        Imported lazily (like :meth:`ModelMetricExtractor.from_pretrained`) so this
        module stays import-clean on a box without ``transformers``/``torchvision``.
        ``d_id``/``d_clip`` are read from the loaded models, so swapping checkpoints
        (e.g. ``sam-vit-huge``, ``dinov2-large``) needs no other change.

        Any model name may instead be a **local directory** for an air-gapped server.
        ``points_per_side`` and ``max_masks`` are the throughput knobs — SAM runs once
        per decoded frame and is the dominant cost of a real Stage-A pass.
        """
        import numpy as _np
        import torch as _t
        import torch.nn.functional as _F
        from PIL import Image
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            CLIPModel,
            CLIPProcessor,
            pipeline,
        )

        mask_gen = pipeline("mask-generation", model=sam_model, device=device)
        dino = AutoModel.from_pretrained(dino_name).to(device).eval()
        dino_proc = AutoImageProcessor.from_pretrained(dino_name)
        clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
        clip_proc = CLIPProcessor.from_pretrained(clip_name)
        d_id = int(dino.config.hidden_size)
        d_clip = int(clip.config.projection_dim)

        def _to_pil(frame: Tensor) -> "Image.Image":
            arr = (frame.detach().clamp(0, 1).permute(1, 2, 0) * 255).to(_t.uint8).cpu().numpy()
            return Image.fromarray(arr)

        def _masked_crop(frame: Tensor, mask: Tensor) -> Optional[Tensor]:
            """Tight bbox crop of ``frame`` with the region's background zeroed."""
            ys, xs = _t.nonzero(mask, as_tuple=True)
            if ys.numel() == 0:
                return None
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = frame[:, y0:y1, x0:x1]
            m = mask[y0:y1, x0:x1].to(crop.dtype)
            return crop * m  # background → 0 so the encoder sees only the object

        def segment_fn(frame: Tensor) -> Tensor:
            hp, wp = frame.shape[-2], frame.shape[-1]
            out = mask_gen(
                _to_pil(frame),
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            masks = out.get("masks") or []
            if not masks:
                return _t.zeros((0, hp, wp), dtype=_t.bool, device=frame.device)
            stacked = _t.from_numpy(_np.stack([_np.asarray(m, dtype=bool) for m in masks]))
            # Keep the largest ``max_masks`` regions: tiny ones are dropped by
            # RegionExtractor's area filter anyway, and this bounds downstream cost.
            areas = stacked.flatten(1).sum(dim=1)
            order = _t.argsort(areas, descending=True)[:max_masks]
            return stacked[order].to(frame.device)

        def identity_fn(frame: Tensor, mask: Tensor) -> Tensor:
            crop = _masked_crop(frame, mask)
            if crop is None or crop.numel() == 0:
                return _t.zeros(d_id, device=frame.device)
            px = dino_proc(images=[_to_pil(crop)], return_tensors="pt")["pixel_values"].to(device)
            with _t.no_grad():
                feat = dino(px).last_hidden_state.mean(dim=1)[0]  # CLS+patch mean-pool → [d_id]
            return feat

        def clip_image_fn(frame: Tensor, mask: Tensor) -> Tensor:
            crop = _masked_crop(frame, mask)
            if crop is None or crop.numel() == 0:
                return _t.zeros(d_clip, device=frame.device)
            px = clip_proc(images=[_to_pil(crop)], return_tensors="pt")["pixel_values"].to(device)
            with _t.no_grad():
                feat = clip.get_image_features(px)[0]  # [d_clip]
            return feat

        def batch_fn(frame: Tensor, masks):
            """All regions of one frame in a single DINOv2 + CLIP forward (§P2-8)."""
            crops = [_masked_crop(frame, m) for m in masks]
            pil = [_to_pil(c) for c in crops if c is not None and c.numel()]
            if not pil:
                z_d = [_t.zeros(d_id, device=frame.device)] * len(masks)
                z_c = [_t.zeros(d_clip, device=frame.device)] * len(masks)
                return z_d, z_c
            d_px = dino_proc(images=pil, return_tensors="pt")["pixel_values"].to(device)
            c_px = clip_proc(images=pil, return_tensors="pt")["pixel_values"].to(device)
            with _t.no_grad():
                d_feat = dino(d_px).last_hidden_state.mean(dim=1)   # [n, d_id]
                c_feat = clip.get_image_features(c_px)              # [n, d_clip]
            return list(d_feat), list(c_feat)

        def clip_text_fn(prompt: str) -> Tensor:
            tin = clip_proc(
                text=[prompt or " "], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with _t.no_grad():
                return clip.get_text_features(**tin)[0]  # [d_clip]

        try:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

            def flow_fn(frame_a: Tensor, frame_b: Tensor) -> Tensor:
                out_dev = frame_a.device

                def _pad8(x: Tensor):
                    _, h, w = x.shape
                    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
                    # RAFT wants [N,3,H,W] in [-1,1], H/W divisible by 8.
                    xp = _F.pad((x[None].to(device) * 2 - 1), (0, pw, 0, ph), mode="reflect")
                    return xp, h, w

                ap, h, w = _pad8(frame_a)
                bp, _, _ = _pad8(frame_b)
                with _t.no_grad():
                    fl = raft(ap, bp)[-1][0]  # [2, H+pad, W+pad] in RAFT's (dx, dy)
                return raft_to_framework_flow(fl[:, :h, :w]).to(out_dev)
        except Exception:  # torchvision RAFT unavailable → zero flow (motion_phase=0)
            def flow_fn(frame_a: Tensor, frame_b: Tensor) -> Tensor:
                _, h, w = frame_a.shape
                return _t.zeros(2, h, w, device=frame_a.device)

        self = cls(
            segment_fn, identity_fn, clip_image_fn, clip_text_fn, flow_fn,
            d_id=d_id, d_clip=d_clip, device=device, batch_fn=batch_fn,
        )
        # Publish the loaded backbones so a co-resident consumer can reuse them.
        # DINOv2 + CLIP is ~1 GB, and Stage A builds *both* this and the metric
        # extractor when --real-models is given, which held two identical copies on
        # the card (§P2-7). See ``ModelMetricExtractor.from_pretrained(share_from=…)``.
        self.models = {"dino": dino, "dino_proc": dino_proc,
                       "clip": clip, "clip_proc": clip_proc}
        return self
