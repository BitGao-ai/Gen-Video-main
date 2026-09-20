"""Real SAM plus DINOv2 plus CLIP plus RAFT perception backend."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocf.common.hf_clip import clip_image_embed, clip_text_embed, clip_text_inputs
from cocf.common.logging import get_logger
from cocf.common.memory import free_memory, freeze
from cocf.common.raft import load_raft, raft_pad
from cocf.tubes.regions import PerceptionProvider

Tensor = torch.Tensor
_log = get_logger(__name__)


def raft_to_framework_flow(flow: Tensor) -> Tensor:
    """Convert RAFT flow to framework channel order."""
    if flow.shape[0] != 2:
        raise ValueError(f"expected a [2, H, W] flow field, got {tuple(flow.shape)}")
    return flow.flip(0)


def _sam_dtype_kwargs(pipeline_fn, dtype) -> dict:
    """Dtype kwarg name for this transformers version."""
    if dtype is None:
        return {}
    import inspect

    try:
        params = inspect.signature(pipeline_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        params = {}
    return {"dtype" if "dtype" in params else "torch_dtype": dtype}


def _fp32_mask_postprocess(original: Callable) -> Callable:
    """Wrap mask postprocess in fp32."""

    def wrapped(all_masks, all_scores, all_boxes, *args, **kwargs):
        if torch.is_tensor(all_scores):
            all_scores = all_scores.float()
        if torch.is_tensor(all_boxes):
            all_boxes = all_boxes.float()
        return original(all_masks, all_scores, all_boxes, *args, **kwargs)

    return wrapped


def _patch_sam_mask_postprocess(mask_gen) -> bool:
    """Patch SAM mask postprocess to fp32."""
    for owner in (getattr(mask_gen, "image_processor", None),
                  getattr(getattr(mask_gen, "processor", None), "image_processor", None)):
        original = getattr(owner, "post_process_for_mask_generation", None)
        if original is None:
            continue
        owner.post_process_for_mask_generation = _fp32_mask_postprocess(original)
        return True
    return False


def _build_mask_generator(sam_model: str, device, dtype):  # pragma: no cover - needs weights
    """Build SAM mask-generation pipeline."""
    from transformers import pipeline

    mask_gen = pipeline("mask-generation", model=sam_model, device=device,
                        **_sam_dtype_kwargs(pipeline, dtype))
    sam = getattr(mask_gen, "model", None)
    if isinstance(sam, nn.Module):
        freeze(sam)
    if dtype is not None and not _patch_sam_mask_postprocess(mask_gen):
        _log.warning(
            "could not reach SamImageProcessor.post_process_for_mask_generation on this "
            "transformers; half-precision SAM may fail in NMS (the load-time probe will "
            "catch it and fall back to fp32)"
        )
    return mask_gen


def _sam_probe_frame() -> Tensor:
    """Probe frame yielding SAM masks."""
    frame = torch.full((3, 128, 128), 0.08)
    frame[:, :, 64:] = 0.92
    frame[:, 40:88, 40:88] = 0.5
    return frame


Tensor = torch.Tensor

SegmentFn = Callable[[Tensor], Tensor]
FeatureFn = Callable[[Tensor, Tensor], Tensor]
TextFn = Callable[[str], Tensor]
FlowFn = Callable[[Tensor, Tensor], Tensor]


class ModelPerception(PerceptionProvider):
    """Real perception from injected SAM/DINOv2/CLIP/RAFT callables."""

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
        clip_grad_fn: Optional[FeatureFn] = None,
    ) -> None:
        """Store injected perception callables."""
        self._segment_fn = segment_fn
        self._identity_fn = identity_fn
        self._clip_image_fn = clip_image_fn
        self._clip_grad_fn = clip_grad_fn
        self._clip_text_fn = clip_text_fn
        self._flow_fn = flow_fn
        self._batch_fn = batch_fn
        self.d_id = int(d_id)
        self.d_clip = int(d_clip)
        self.device = torch.device(device)

    def segment(self, frame: Tensor) -> Tensor:
        """Segment frame into instance masks."""
        return self._segment_fn(frame)

    def identity_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """Identity embedding of masked region."""
        if not self._nonempty(mask):
            return torch.zeros(self.d_id, device=frame.device)
        return self._as_vector(
            self._identity_fn(frame, mask), self.d_id, frame.device, "identity_fn"
        )

    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        """CLIP match score for region vs prompt."""
        if not self._nonempty(mask):
            return 0.0
        img = self._as_vector(
            self._clip_image_fn(frame, mask), self.d_clip, frame.device, "clip_image_fn"
        )
        txt = self._as_vector(
            self._clip_text_fn(prompt), self.d_clip, img.device, "clip_text_fn"
        )
        img, txt = F.normalize(img, dim=0), F.normalize(txt, dim=0)
        return float((img @ txt).clamp(-1.0, 1.0) * 0.5 + 0.5)

    def clip_feature(self, frame: Tensor, mask: Tensor) -> Tensor:
        """CLIP visual embedding of region."""
        if not self._nonempty(mask):
            return torch.zeros(self.d_clip, device=frame.device)
        return self._as_vector(
            self._clip_image_fn(frame, mask), self.d_clip, frame.device, "clip_image_fn"
        )

    def clip_feature_grad(self, frame: Tensor, mask: Tensor) -> Tensor:
        """CLIP embedding retaining input gradients."""
        if self._clip_grad_fn is None:
            raise RuntimeError("Differentiable tube CLIP requires clip_grad_fn")
        return self._as_vector(self._clip_grad_fn(frame, mask), self.d_clip,
                               frame.device, "clip_grad_fn", detach=False)

    def text_feature(self, prompt: str) -> Tensor:
        """CLIP text embedding of prompt."""
        return self._clip_text_fn(prompt).detach().float()

    def optical_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """Optical flow from frame_a to frame_b."""
        return self._flow_fn(frame_a, frame_b).to(frame_a.device)

    def region_features(self, frame: Tensor, masks):
        """Batched identity and CLIP features for frame."""
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
            ident[i] = self._as_vector(d_feats[j], self.d_id, frame.device, "batch_fn identity")
            textf[i] = self._as_vector(c_feats[j], self.d_clip, frame.device, "batch_fn clip")
        return ident, textf

    def _as_vector(self, feat: Tensor, width: int, device, what: str, *, detach: bool = True) -> Tensor:
        """Coerce callable output to 1-D feature vector."""
        feat = (feat.detach() if detach else feat).float().to(device)
        if feat.ndim > 1 and feat.shape[0] == 1:
            feat = feat[0]
        if feat.ndim != 1 or feat.shape[0] != width:
            raise ValueError(
                f"{what} must return a [{width}] vector, got {tuple(feat.shape)}. "
                "A 2-D shape here is a token sequence (e.g. [50, 768] vision / "
                "[77, 512] text), i.e. the pooled + projected embedding was never "
                "taken — build the CLIP callables via cocf.common.hf_clip."
            )
        return feat

    @staticmethod
    def _nonempty(mask: Tensor) -> bool:
        """True if mask has any pixel."""
        return bool(mask.any())

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cuda",
        *,
        sam_model: str = "facebook/sam-vit-base",
        dino_name: str = "facebook/dinov2-base",
        clip_name: str = "openai/clip-vit-base-patch32",
        points_per_crop: int = 16,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        max_masks: int = 24,
        dtype: Optional[torch.dtype] = None,
        raft_weights: Optional[str] = None,
        require_flow: bool = False,
    ) -> "ModelPerception":  # pragma: no cover - needs model downloads
        """Build standard SAM plus DINOv2 plus CLIP plus RAFT stack."""
        import numpy as _np
        import torch as _t
        from PIL import Image
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            CLIPModel,
            CLIPProcessor,
        )

        mask_gen = _build_mask_generator(sam_model, device, dtype)
        dino = freeze(AutoModel.from_pretrained(dino_name).to(device))
        dino_proc = AutoImageProcessor.from_pretrained(dino_name)
        clip = freeze(CLIPModel.from_pretrained(clip_name).to(device))
        clip_proc = CLIPProcessor.from_pretrained(clip_name)
        if dtype is not None:
            dino = dino.to(dtype)
            clip = clip.to(dtype)
        d_id = int(dino.config.hidden_size)
        d_clip = int(clip.config.projection_dim)
        dino_dtype = next(dino.parameters()).dtype
        clip_dtype = next(getattr(clip, "vision_model", clip).parameters()).dtype

        def _to_pil(frame: Tensor) -> "Image.Image":
            """Convert frame tensor to PIL image."""
            arr = (frame.detach().clamp(0, 1).permute(1, 2, 0) * 255).to(_t.uint8).cpu().numpy()
            return Image.fromarray(arr)

        def _masked_crop(frame: Tensor, mask: Tensor) -> Optional[Tensor]:
            """Crop frame to mask bbox with background zeroed."""
            ys, xs = _t.nonzero(mask, as_tuple=True)
            if ys.numel() == 0:
                return None
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = frame[:, y0:y1, x0:x1]
            m = mask[y0:y1, x0:x1].to(crop.dtype)
            return crop * m

        def segment_fn(frame: Tensor) -> Tensor:
            """Segment frame into masks."""
            hp, wp = frame.shape[-2], frame.shape[-1]
            out = mask_gen(
                _to_pil(frame),
                points_per_crop=points_per_crop,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            masks = out.get("masks") or []
            if not masks:
                return _t.zeros((0, hp, wp), dtype=_t.bool, device=frame.device)
            stacked = _t.from_numpy(_np.stack([_np.asarray(m, dtype=bool) for m in masks]))
            areas = stacked.flatten(1).sum(dim=1)
            order = _t.argsort(areas, descending=True)[:max_masks]
            return stacked[order].to(frame.device)

        def _probe_segmentation() -> int:
            """Probe mask count on test frame."""
            return int(segment_fn(_sam_probe_frame()).shape[0])

        try:
            n_probe = _probe_segmentation()
        except Exception as exc:
            if dtype is None:
                raise
            _log.warning(
                "SAM mask generation failed at dtype=%s (%s: %s); reloading SAM in "
                "float32 — slower per frame, but dtype-safe", dtype, type(exc).__name__, exc,
            )
            mask_gen = None
            free_memory()
            mask_gen = _build_mask_generator(sam_model, device, None)
            n_probe = _probe_segmentation()
        if n_probe:
            _log.info("SAM mask path verified at load (%d masks on the probe frame)", n_probe)
        else:
            _log.warning(
                "SAM returned no masks for the load-time probe frame, so the mask "
                "post-processing path is unverified — a dtype/NMS failure would only "
                "surface on a real clip. Check pred_iou_thresh/stability_score_thresh."
            )

        def identity_fn(frame: Tensor, mask: Tensor) -> Tensor:
            """Identity embedding for region crop."""
            crop = _masked_crop(frame, mask)
            if crop is None or crop.numel() == 0:
                return _t.zeros(d_id, device=frame.device)
            px = dino_proc(images=[_to_pil(crop)], return_tensors="pt")["pixel_values"].to(
                device=device, dtype=dino_dtype)
            with _t.no_grad():
                feat = dino(px).last_hidden_state.mean(dim=1)[0]
            return feat

        def clip_image_fn(frame: Tensor, mask: Tensor) -> Tensor:
            """CLIP image embedding for region crop."""
            with _t.no_grad():
                return clip_grad_fn(frame, mask)

        def clip_grad_fn(frame: Tensor, mask: Tensor) -> Tensor:
            """Differentiable CLIP embedding for region crop."""
            crop = _masked_crop(frame, mask)
            if crop is None or crop.numel() == 0:
                return _t.zeros(d_clip, device=frame.device)
            ip = clip_proc.image_processor
            size = ip.size
            edge = int(size["shortest_edge"] if isinstance(size, dict) else size)
            h, w = crop.shape[-2:]
            resized = (edge, int(edge * w / h)) if h <= w else (int(edge * h / w), edge)
            px = _t.nn.functional.interpolate(crop[None].to(device=device, dtype=_t.float32),
                size=resized, mode="bicubic", align_corners=False, antialias=True)
            ch, cw = int(ip.crop_size["height"]), int(ip.crop_size["width"])
            pad_h, pad_w = max(0, ch - resized[0]), max(0, cw - resized[1])
            px = _t.nn.functional.pad(px, (pad_w // 2, pad_w - pad_w // 2,
                                           pad_h // 2, pad_h - pad_h // 2))
            top, left = (px.shape[-2] - ch) // 2, (px.shape[-1] - cw) // 2
            px = px[:, :, top:top + ch, left:left + cw]
            mean = px.new_tensor(ip.image_mean).view(1, 3, 1, 1)
            std = px.new_tensor(ip.image_std).view(1, 3, 1, 1)
            return clip_image_embed(clip, ((px - mean) / std).to(clip_dtype))[0]

        def batch_fn(frame: Tensor, masks):
            """Batched region embeddings for one frame."""
            crops = [_masked_crop(frame, m) for m in masks]
            pil = [_to_pil(c) for c in crops if c is not None and c.numel()]
            if not pil:
                z_d = [_t.zeros(d_id, device=frame.device)] * len(masks)
                z_c = [_t.zeros(d_clip, device=frame.device)] * len(masks)
                return z_d, z_c
            d_px = dino_proc(images=pil, return_tensors="pt")["pixel_values"].to(
                device=device, dtype=dino_dtype)
            c_px = clip_proc(images=pil, return_tensors="pt")["pixel_values"].to(
                device=device, dtype=clip_dtype)
            with _t.no_grad():
                d_feat = dino(d_px).last_hidden_state.mean(dim=1)
                c_feat = clip_image_embed(clip, c_px)
            return list(d_feat), list(c_feat)

        _text_cache: "OrderedDict[str, Tensor]" = OrderedDict()

        def clip_text_fn(prompt: str) -> Tensor:
            """Cached CLIP text embedding."""
            hit = _text_cache.get(prompt)
            if hit is not None:
                _text_cache.move_to_end(prompt)
                return hit
            tin = clip_text_inputs(clip, clip_proc, [prompt], device=device)
            with _t.no_grad():
                emb = clip_text_embed(clip, **tin)[0]
            _text_cache[prompt] = emb
            if len(_text_cache) > 4:
                _text_cache.popitem(last=False)
            return emb

        raft = load_raft(device, variant="large", weights_path=raft_weights,
                         required=require_flow)
        if raft is not None:
            raft_dtype = next(raft.parameters()).dtype

            def flow_fn(frame_a: Tensor, frame_b: Tensor) -> Tensor:
                """RAFT flow for frame pair."""
                out_dev = frame_a.device

                def _pad8(x: Tensor):
                    """Pad frame for RAFT."""
                    _, h, w = x.shape
                    x = x[None].to(device=device, dtype=raft_dtype)
                    return raft_pad(x * 2 - 1), h, w

                ap, h, w = _pad8(frame_a)
                bp, _, _ = _pad8(frame_b)
                with _t.no_grad():
                    fl = raft(ap, bp)[-1][0]
                return raft_to_framework_flow(fl[:, :h, :w]).to(device=out_dev, dtype=_t.float32)
        else:
            def flow_fn(frame_a: Tensor, frame_b: Tensor) -> Tensor:
                """Zero flow fallback."""
                _, h, w = frame_a.shape
                return _t.zeros(2, h, w, device=frame_a.device)

        self = cls(
            segment_fn, identity_fn, clip_image_fn, clip_text_fn, flow_fn,
            d_id=d_id, d_clip=d_clip, device=device, batch_fn=batch_fn,
            clip_grad_fn=clip_grad_fn,
        )
        self.models = {"dino": dino, "dino_proc": dino_proc,
                       "clip": clip, "clip_proc": clip_proc}
        return self
