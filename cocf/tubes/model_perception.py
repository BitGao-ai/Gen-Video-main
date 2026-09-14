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


# --------------------------------------------------------------------------- #
# SAM (mask-generation) plumbing — half-precision safety
# --------------------------------------------------------------------------- #


def _sam_dtype_kwargs(pipeline_fn, dtype) -> dict:
    """``{"dtype": …}`` or ``{"torch_dtype": …}`` — whichever this transformers wants.

    ``transformers`` 4.56 renamed the pipeline's weight-dtype argument and now emits
    ```torch_dtype` is deprecated! Use `dtype` instead!`` for the old spelling,
    while releases before it only accept ``torch_dtype``. Read the parameter name off
    the function instead of pinning a version, so both sides of the rename work.
    """
    if dtype is None:
        return {}
    import inspect

    try:
        params = inspect.signature(pipeline_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        params = {}
    return {"dtype" if "dtype" in params else "torch_dtype": dtype}


def _fp32_mask_postprocess(original: Callable) -> Callable:
    """``post_process_for_mask_generation`` with its NMS operands cast to fp32."""

    def wrapped(all_masks, all_scores, all_boxes, *args, **kwargs):
        if torch.is_tensor(all_scores):
            all_scores = all_scores.float()
        if torch.is_tensor(all_boxes):
            all_boxes = all_boxes.float()
        return original(all_masks, all_scores, all_boxes, *args, **kwargs)

    return wrapped


def _patch_sam_mask_postprocess(mask_gen) -> bool:
    """Force SAM's automatic-mask post-processing into fp32 on ``mask_gen``.

    ``MaskGenerationPipeline.postprocess`` hands the per-crop masks, the model's own
    IoU scores and the predicted boxes to
    ``SamImageProcessor.post_process_for_mask_generation``, which NMS-merges them
    with::

        batched_nms(boxes=all_boxes.float(), scores=all_scores, …)

    — boxes cast to fp32, scores left in the *model's* dtype. Under
    ``--perception-dtype bfloat16`` that is a dtype mismatch torchvision rejects
    outright::

        RuntimeError: dets should have the same type as scores

    and it only fires once a frame yields at least one surviving mask (torchvision
    short-circuits empty inputs before the dtype check), i.e. never at load and never
    on a black test frame — it took down a 2600-clip Stage-A run on its 12th clip.
    Casting both operands at this seam keeps the *weights* in bf16, which is where the
    speed is; NMS over ≤a few hundred boxes in fp32 costs nothing measurable.

    Returns whether the patch was applied (False ⇒ a transformers whose seam moved;
    :meth:`ModelPerception.from_pretrained` then falls back to fp32 weights).
    """
    for owner in (getattr(mask_gen, "image_processor", None),
                  getattr(getattr(mask_gen, "processor", None), "image_processor", None)):
        original = getattr(owner, "post_process_for_mask_generation", None)
        if original is None:
            continue
        # Instance attribute: shadows the class method for this pipeline only, so a
        # co-resident SamImageProcessor elsewhere in the process is untouched.
        owner.post_process_for_mask_generation = _fp32_mask_postprocess(original)
        return True
    return False


def _build_mask_generator(sam_model: str, device, dtype):  # pragma: no cover - needs weights
    """The ``mask-generation`` pipeline for ``sam_model``, dtype-safe (see above)."""
    from transformers import pipeline

    mask_gen = pipeline("mask-generation", model=sam_model, device=device,
                        **_sam_dtype_kwargs(pipeline, dtype))
    # SAM only ever runs under ``no_grad`` (tube segmentation on a preview decode), so
    # this costs nothing today — but it is the same class of latent waste as the DINO/
    # CLIP/RAFT freezes above, and leaving one tower trainable is how the next caller
    # reintroduces a 94M-parameter gradient buffer by accident.
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
    """A tiny high-contrast frame SAM reliably returns *surviving* masks for.

    The mask path's dtype trap is only reachable when NMS actually runs, so the
    load-time probe needs a frame that clears ``pred_iou_thresh`` /
    ``stability_score_thresh`` — flat noise or a black frame gets filtered to nothing
    and would pass the probe vacuously.
    """
    frame = torch.full((3, 128, 128), 0.08)
    frame[:, :, 64:] = 0.92          # hard vertical split
    frame[:, 40:88, 40:88] = 0.5     # a centred square straddling it
    return frame


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
        clip_grad_fn: Optional[FeatureFn] = None,
    ) -> None:
        self._segment_fn = segment_fn
        self._identity_fn = identity_fn
        self._clip_image_fn = clip_image_fn
        self._clip_grad_fn = clip_grad_fn
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
        return self._as_vector(
            self._identity_fn(frame, mask), self.d_id, frame.device, "identity_fn"
        )

    def clip_score(self, frame: Tensor, mask: Tensor, prompt: str) -> float:
        """CLIP image-text match in ``[0, 1]`` for the region vs the prompt.

        ``cosine(clip_img, clip_txt)·0.5 + 0.5`` — the same ``[-1,1]→[0,1]`` map
        :class:`~cocf.data.metrics.ModelMetricExtractor` uses for CLIPScore. Feeds
        the §4.3.1 semantic filter (drop regions below ``TubeConfig.min_clip_score``).

        Both operands are validated to be the pooled ``[d_clip]`` embedding first: the
        cosine is only meaningful in CLIP's *joint* space, so an un-projected tower
        output must not reach the dot product (see :meth:`_as_vector`).
        """
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
        """CLIP visual embedding ``[d_clip]`` of the region (for text-tube alignment)."""
        if not self._nonempty(mask):
            return torch.zeros(self.d_clip, device=frame.device)
        return self._as_vector(
            self._clip_image_fn(frame, mask), self.d_clip, frame.device, "clip_image_fn"
        )

    def clip_feature_grad(self, frame: Tensor, mask: Tensor) -> Tensor:
        """Frozen CLIP forward retaining gradients to the input pixels."""
        if self._clip_grad_fn is None:
            raise RuntimeError("Differentiable tube CLIP requires clip_grad_fn")
        return self._as_vector(self._clip_grad_fn(frame, mask), self.d_clip,
                               frame.device, "clip_grad_fn", detach=False)

    def text_feature(self, prompt: str) -> Tensor:
        """CLIP text embedding ``[d_clip]`` of the prompt (memoised by the callable)."""
        return self._clip_text_fn(prompt).detach().float()

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
            ident[i] = self._as_vector(d_feats[j], self.d_id, frame.device, "batch_fn identity")
            textf[i] = self._as_vector(c_feats[j], self.d_clip, frame.device, "batch_fn clip")
        return ident, textf

    # -- helpers -------------------------------------------------------- #

    def _as_vector(self, feat: Tensor, width: int, device, what: str, *, detach: bool = True) -> Tensor:
        """Coerce a callable's output to the contracted 1-D ``[width]`` feature vector.

        A leading batch axis of 1 is unwrapped (``[1, d]`` → ``[d]``); anything else is
        a contract violation and is rejected **here**, naming the offending shape,
        instead of being ``flatten()``-ed into a longer vector that only fails much
        later as an unreadable size error deep in the tube path. That is precisely how
        a ``transformers`` upgrade reached this code: ``get_image_features`` /
        ``get_text_features`` started returning the towers' *token sequences* rather
        than the pooled, projected embeddings, and the first symptom was

            RuntimeError: inconsistent tensor size, expected tensor [38400] and
            src [39424]

        from :meth:`clip_score` — 50·768 vision tokens against 77·512 text tokens, four
        call frames below the actual mistake. The same silent widening would otherwise
        have written un-projected 768-d "CLIP embeds" into every Stage-A
        ``tube_visual_embed`` (:func:`cocf.lcocf.data.tube_clip_embed`) and into CMSC's
        probed ``visual_dim``. See :mod:`cocf.common.hf_clip` for the fix at the source.
        """
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
        points_per_crop: int = 16,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        max_masks: int = 24,
        dtype: Optional[torch.dtype] = None,
        raft_weights: Optional[str] = None,
        require_flow: bool = False,
    ) -> "ModelPerception":  # pragma: no cover - needs model downloads
        """Wire SAM(mask-generation) + DINOv2 + CLIP + RAFT into the five callables.

        Imported lazily (like :meth:`ModelMetricExtractor.from_pretrained`) so this
        module stays import-clean on a box without ``transformers``/``torchvision``.
        ``d_id``/``d_clip`` are read from the loaded models, so swapping checkpoints
        (e.g. ``sam-vit-huge``, ``dinov2-large``) needs no other change.

        Any model name may instead be a **local directory** for an air-gapped server.
        ``points_per_crop`` and ``max_masks`` are the throughput knobs — SAM runs once
        per decoded frame and is the dominant cost of a real Stage-A pass. The name
        matches the HuggingFace ``mask-generation`` pipeline's parameter deliberately:
        the original ``segment_anything`` repo calls it ``points_per_side``, and
        :meth:`MaskGenerationPipeline._sanitize_parameters` **silently drops** kwargs it
        does not recognise — so passing the segment-anything spelling left the default
        ``points_per_crop=32`` in force and ran 1024 point prompts per frame instead of
        the intended 256, at 4× the time and 4× the mask-postprocessing transient.

        ``dtype`` narrows DINOv2/CLIP/SAM (RAFT is deliberately left in fp32 — its
        all-pairs correlation volume is numerically fragile at half precision). ``None``
        keeps the checkpoints' own dtype, which is the historical behaviour. SAM's
        *weights* narrow, but its mask post-processing is pinned back to fp32
        (:func:`_patch_sam_mask_postprocess`) and the whole mask path is probed here at
        load — half precision otherwise dies inside torchvision NMS on the first
        textured frame, not at load.

        ``raft_weights`` points RAFT at a local checkpoint (an offline host cannot
        fetch the torchvision ``DEFAULT`` weights); ``require_flow`` makes an
        unavailable RAFT a hard failure rather than a silent zero-flow fallback — see
        :func:`cocf.common.raft.load_raft`.
        """
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
        # ``freeze``, not just ``.eval()``: Stage C runs CLIP (and, through the metric
        # extractor, DINOv2/RAFT) on the *accelerated* branch with the autograd graph
        # on, so a tower whose parameters still say ``requires_grad=True`` makes
        # autograd (a) retain each layer's input for a **weight** gradient nothing will
        # ever use, and (b) materialise a full fp32 ``.grad`` for every parameter on
        # the first backward — ~0.97 GiB across DINOv2-base + CLIP + RAFT, which no
        # optimiser touches and ``zero_grad`` never clears (Stage C's optimiser only
        # knows the 7M plugin/LoRA params). The gradient the §6.3.2 loss actually needs
        # is w.r.t. the *input pixels*, and that path is unaffected by freezing.
        dino = freeze(AutoModel.from_pretrained(dino_name).to(device))
        dino_proc = AutoImageProcessor.from_pretrained(dino_name)
        clip = freeze(CLIPModel.from_pretrained(clip_name).to(device))
        clip_proc = CLIPProcessor.from_pretrained(clip_name)
        if dtype is not None:
            dino = dino.to(dtype)
            clip = clip.to(dtype)
        d_id = int(dino.config.hidden_size)
        d_clip = int(clip.config.projection_dim)
        # Pixel values must match the towers' weight dtype; ``_as_vector`` casts every
        # feature back to fp32 on the way out, so nothing downstream sees half precision.
        # Resolved **per tower, off the module** rather than as ``dtype or float32``:
        # ``dtype=None`` keeps each checkpoint's own precision (see above), which is not
        # necessarily fp32, and DINOv2/CLIP are separate modules that a future loader
        # need not narrow together. Reading the weights is correct under every path.
        # For CLIP that means the *vision* tower — the submodule these pixels reach via
        # ``clip_image_embed`` — not ``next(clip.parameters())``, which reports whichever
        # submodule CLIPModel happens to register first.
        dino_dtype = next(dino.parameters()).dtype
        clip_dtype = next(getattr(clip, "vision_model", clip).parameters()).dtype

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
                points_per_crop=points_per_crop,
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

        # Prove the mask path end-to-end *at load* rather than discovering a dtype or
        # checkpoint problem an hour into a shard: one SAM forward over a 128 px
        # synthetic frame chosen to actually yield masks, so the run reaches the NMS
        # merge that half precision breaks (see ``_patch_sam_mask_postprocess``). If it
        # still fails, drop SAM to fp32 — slower per frame, but with no dtype seam left
        # — instead of letting the shard die on its first textured clip.
        def _probe_segmentation() -> int:
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
            mask_gen = None          # release the half-precision copy before reloading
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
            crop = _masked_crop(frame, mask)
            if crop is None or crop.numel() == 0:
                return _t.zeros(d_id, device=frame.device)
            px = dino_proc(images=[_to_pil(crop)], return_tensors="pt")["pixel_values"].to(
                device=device, dtype=dino_dtype)
            with _t.no_grad():
                feat = dino(px).last_hidden_state.mean(dim=1)[0]  # CLS+patch mean-pool → [d_id]
            return feat

        def clip_image_fn(frame: Tensor, mask: Tensor) -> Tensor:
            with _t.no_grad():
                return clip_grad_fn(frame, mask)

        def clip_grad_fn(frame: Tensor, mask: Tensor) -> Tensor:
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
            """All regions of one frame in a single DINOv2 + CLIP forward (§P2-8)."""
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
                d_feat = dino(d_px).last_hidden_state.mean(dim=1)   # [n, d_id]
                c_feat = clip_image_embed(clip, c_px)               # [n, d_clip]
            return list(d_feat), list(c_feat)

        # One clip = one prompt, but ``RegionExtractor.extract_frame`` calls
        # ``clip_score`` once per *region* (regions.py) and each call re-encodes the
        # caption — ~300 redundant text-tower forwards per clip, each preceded by a
        # host sync. The cache is keyed by prompt and holds a handful of entries
        # because the access pattern is one hot key with rare changes.
        _text_cache: "OrderedDict[str, Tensor]" = OrderedDict()

        def clip_text_fn(prompt: str) -> Tensor:
            hit = _text_cache.get(prompt)
            if hit is not None:
                _text_cache.move_to_end(prompt)
                return hit
            tin = clip_text_inputs(clip, clip_proc, [prompt], device=device)
            with _t.no_grad():
                emb = clip_text_embed(clip, **tin)[0]  # [d_clip]
            _text_cache[prompt] = emb
            if len(_text_cache) > 4:
                _text_cache.popitem(last=False)
            return emb

        raft = load_raft(device, variant="large", weights_path=raft_weights,
                         required=require_flow)
        if raft is not None:
            # RAFT's weights stay in whatever dtype the checkpoint loads as (fp32).
            # The frames handed to us come from the teacher's VAE decode and are
            # therefore often bf16/fp16, which conv2d rejects outright:
            #   RuntimeError: Input type (c10::BFloat16) and bias type (float)
            #   should be the same
            # Feed RAFT its own dtype rather than casting the module, so a caller
            # running an autocast pipeline never dictates the flow net's precision.
            raft_dtype = next(raft.parameters()).dtype

            def flow_fn(frame_a: Tensor, frame_b: Tensor) -> Tensor:
                out_dev = frame_a.device

                def _pad8(x: Tensor):
                    _, h, w = x.shape
                    # RAFT wants [N,3,H,W] in [-1,1], H/W divisible by 8 *and* at
                    # least RAFT_MIN_EDGE (below that its correlation pyramid
                    # raises); bottom/right only, so the [:h, :w] slice below
                    # recovers the caller's field.
                    x = x[None].to(device=device, dtype=raft_dtype)
                    return raft_pad(x * 2 - 1), h, w

                ap, h, w = _pad8(frame_a)
                bp, _, _ = _pad8(frame_b)
                with _t.no_grad():
                    fl = raft(ap, bp)[-1][0]  # [2, H+pad, W+pad] in RAFT's (dx, dy)
                return raft_to_framework_flow(fl[:, :h, :w]).to(device=out_dev, dtype=_t.float32)
        else:
            def flow_fn(frame_a: Tensor, frame_b: Tensor) -> Tensor:
                _, h, w = frame_a.shape
                return _t.zeros(2, h, w, device=frame_a.device)

        self = cls(
            segment_fn, identity_fn, clip_image_fn, clip_text_fn, flow_fn,
            d_id=d_id, d_clip=d_clip, device=device, batch_fn=batch_fn,
            clip_grad_fn=clip_grad_fn,
        )
        # Publish the loaded backbones so a co-resident consumer can reuse them.
        # DINOv2 + CLIP is ~1 GB, and Stage A builds *both* this and the metric
        # extractor when --real-models is given, which held two identical copies on
        # the card (§P2-7). See ``ModelMetricExtractor.from_pretrained(share_from=…)``.
        self.models = {"dino": dino, "dino_proc": dino_proc,
                       "clip": clip, "clip_proc": clip_proc}
        return self
