"""Typed, hierarchical configuration for COCF-SS-DCA.

Every hyperparameter that appears in the design document lives here exactly once,
annotated with the section it comes from. Sub-configs mirror the subsystem layout
so a component only ever receives the slice of config it needs (keeping coupling
low — e.g. the affinity module takes an :class:`AffinityConfig`, not the world).

Configs are plain ``dataclasses`` (no third-party dependency required). They can be
built from nested ``dict`` (``Config.from_dict``) or a YAML file (``Config.load``)
and serialized back (``to_dict``), so experiment configs stay human-readable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# STA — semantic tube anchoring (§4)
# --------------------------------------------------------------------------- #


@dataclass
class AffinityConfig:
    """Cross-frame region affinity weights ``Aff(i, j)`` (§4.3.1)."""

    w_id: float = 0.40  # identity (DINOv2) similarity
    w_flow: float = 0.30  # optical-flow warped-mask consistency
    w_iou: float = 0.15  # warped-mask IoU
    w_txt: float = 0.10  # text-alignment similarity
    w_pos: float = 0.05  # positional proximity
    sigma_p: float = 16.0  # positional kernel bandwidth (latent px)
    flow_scale: float = 1.0  # divisor inside the flow exp(-||·||) kernel


@dataclass
class TubeConfig:
    """Semantic-tube construction & maintenance (§4.3.1, §4.3.2)."""

    affinity: AffinityConfig = field(default_factory=AffinityConfig)
    min_region_ratio: float = 0.001  # drop regions < 0.1% of total pixels
    min_clip_score: float = 0.20  # drop low-semantic regions (CLIP match < 0.2)
    max_tube_len: int = 16  # split tubes longer than 16 frames
    affinity_match_threshold: float = 0.30  # Hungarian gating threshold
    identity_unstable_threshold: float = 0.50  # I_k < 0.5 ⇒ force FULL
    identity_ema: float = 0.90  # EMA factor for the running identity feature
    # Tube-level action-smoothing loss (§4.3.2)
    lambda_temporal: float = 0.20
    lambda_boundary: float = 0.10


# --------------------------------------------------------------------------- #
# L-COCF — lightweight counterfactual causal compute field (§3)
# --------------------------------------------------------------------------- #


@dataclass
class StrengthConfig:
    """Causal-strength model ``s = α·s_E + β·s_A + γ·s_T`` and tier thresholds (§3.3.2/3)."""

    alpha_init: float = 1.0  # learnable weight on entity strength s_E
    beta_init: float = 1.0  # learnable weight on action strength s_A
    gamma_init: float = 1.0  # learnable weight on temporal-transition strength s_T
    theta1: float = 0.66  # s > θ1 ⇒ HIGH tier (FULL)
    theta2: float = 0.33  # θ2 < s ≤ θ1 ⇒ MID (LOW FREQ); else LOW (INTERP/ANCHOR)
    normalize_strength: bool = True  # squash s to [0,1] before thresholding


@dataclass
class PredictorConfig:
    """Counterfactual damage predictor ``H_φ`` head (§3.3, used by §5.3.1)."""

    state_dim: int = 7  # TUBE_STATE_DIM
    context_dim: int = 64  # extra context features (strength, budget, step embed…)
    hidden_dim: int = 128
    num_layers: int = 3
    num_actions: int = 4
    dropout: float = 0.0
    # predict log-variance for numerical stability; σ = exp(0.5·logger)
    predict_log_variance: bool = True


@dataclass
class CounterfactualConfig:
    """Local single-hop counterfactual verification (§3.3.4)."""

    theta_sT: float = 0.50  # trigger CF check only at temporal mutation points s_T>θ
    eta: float = 0.05  # residual threshold Δ<η ⇒ causal-omission ⇒ repair
    max_checks_per_step: int = 4  # T_jump · S budget cap (keep it cheap)
    repair_net_dim: int = 128  # lightweight residual-repair sub-net width


@dataclass
class LCOCFConfig:
    strength: StrengthConfig = field(default_factory=StrengthConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    # spatio-temporal locality neighborhood (axiom §3.2.1): time ±tau, space radius r
    tau: int = 2
    spatial_radius: int = 2
    vlm_name: str = "frozen-vlm"  # causal-triplet parser, frozen (§3.3.5)


# --------------------------------------------------------------------------- #
# RAEC — revocable anchoring & error certificates (§5)
# --------------------------------------------------------------------------- #


@dataclass
class CertificateConfig:
    """Error-certificate ``E_cert`` coefficients & training (§5.3.1)."""

    kappa: float = 1.96  # uncertainty multiplier (95% one-sided)
    lambda_res: float = 0.10  # residual-to-anchor term
    lambda_bnd: float = 0.05  # boundary-gradient term
    lambda_age: float = 0.01  # anchor-age term
    lambda_cmsc: float = 0.20  # local CMSC term
    # certificate training loss (§5.3.1)
    alpha_cert: float = 0.10  # penalty weight for exceeding τ_safe
    tau_safe: float = 0.20


@dataclass
class TriggerConfig:
    """Risk trigger & local repair (§5.3.2)."""

    tau_low: float = 0.40
    tau_high: float = 0.80
    force_full_steps: int = 2  # q: steps to force FULL after a rollback
    sigma_bnd: float = 4.0  # boundary soft-mask bandwidth


# --------------------------------------------------------------------------- #
# CMSC — cross-modal semantic conservation (§6)
# --------------------------------------------------------------------------- #


@dataclass
class CMSCConfig:
    """Multidimensional semantic-conservation loss weights (§6.3.2)."""

    lambda_align: float = 0.30  # text-tube alignment conservation
    lambda_id: float = 0.20  # identity (DINO) conservation
    lambda_motion: float = 0.20  # motion (RAFT) conservation
    lambda_spatial: float = 0.15  # spatial-relation conservation
    lambda_ocr: float = 0.10  # OCR / text conservation
    lambda_bnd: float = 0.05  # boundary conservation
    temperature: float = 0.07  # τ in the text-tube alignment softmax
    align_dim: int = 256  # projection dim W: R^{d_c × d_v}


# --------------------------------------------------------------------------- #
# Scheduler — budget & allocation (§2.2, §7.3)
# --------------------------------------------------------------------------- #


@dataclass
class BudgetConfig:
    """Dynamic per-step compute budget ``B_t`` (§7.3)."""

    b_min: float = 0.30  # min fraction of full compute
    b_max: float = 1.00  # max fraction of full compute
    eta_scene: float = 0.10  # weight on scene complexity
    eta_uncertainty: float = 0.15  # weight on mean damage uncertainty
    eta_interaction: float = 0.10  # weight on tube interaction density
    # q(t): U-shaped time weight — high early (structure) & late (detail) (§7.3)
    q_early_boost: float = 0.30
    q_late_boost: float = 0.40
    q_mid_floor: float = 0.10


@dataclass
class AllocatorConfig:
    """Budget-constrained action allocation (§2.2)."""

    # per-action relative cost multipliers C(a, |g_k|) ∝ |g_k| (§2.2)
    action_cost: Tuple[float, float, float, float] = (1.0, 0.45, 0.15, 0.0)
    risk_threshold: float = 0.80  # τ_r hard risk constraint (== TriggerConfig.tau_high)
    greedy_fallback: bool = True  # use greedy knapsack if LP solver unavailable


# --------------------------------------------------------------------------- #
# Engine — accelerated inference loop (§7.2)
# --------------------------------------------------------------------------- #


@dataclass
class EngineConfig:
    """Knobs of the accelerated denoising loop (§7.2)."""

    num_inference_steps: int = 30
    # Tube construction is expensive (SAM); build once at this step then refresh
    # state cheaply. ``tube_build_step`` counts from the *start* (t=T is step 0).
    tube_build_step: int = 1  # build after 1 warm-up FULL step so structure exists
    tube_refresh_every: int = 0  # re-segment every N steps (0 = never re-segment)
    warmup_full_steps: int = 2  # first steps run dense FULL (cold-start stability)
    lowfreq_stride: int = 2  # LOW FREQ spatial stride (2 ⇒ ~1/4 tokens computed)
    measure_residual: bool = True  # measure skip residuals for the certificate
    cf_check_enabled: bool = True  # run §3.3.4 single-hop counterfactual checks
    decode_preview_for_tubes: bool = True  # decode a preview to segment when no GT
    use_dynamic_budget: bool = True  # else spend b_max every step
    risk_control_enabled: bool = True  # enable RAEC trigger/repair at inference


# --------------------------------------------------------------------------- #
# Backbone & memory (multi-model compat + memory savings: user reqs #1, #2)
# --------------------------------------------------------------------------- #


@dataclass
class BackboneConfig:
    """Which backbone to wrap and where its weights live (§9.1)."""

    name: str = "mock"  # registry key: "hunyuanvideo" | "wan21" | "mock"
    model_path: Optional[str] = None
    dtype: str = "bfloat16"  # compute dtype for the frozen backbone
    device: str = "cuda"
    # backbone-specific knobs passed straight through to the adapter
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Training memory-saving switches (user requirement #1, realised in §7.1)."""

    amp_dtype: str = "bfloat16"  # autocast dtype; "none" disables AMP
    gradient_checkpointing: bool = True  # checkpoint trainable blocks
    grad_accum_steps: int = 4  # accumulate to shrink effective batch memory
    offload_backbone_to_cpu: bool = False  # keep frozen backbone on CPU until needed
    offload_optimizer_state: bool = False  # 8-bit / paged optimiser state
    cache_latents: bool = True  # pre-encode videos → latents on disk (no VAE in RAM)
    cache_teacher_labels: bool = True  # Stage-A labels precomputed offline
    empty_cache_every: int = 0  # call torch.cuda.empty_cache every N steps (0=never)
    max_grad_norm: float = 1.0


# --------------------------------------------------------------------------- #
# Data — video/caption reading, latent caching & teacher generation (§7.1)
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Video+caption dataset reading & latent caching (§7.1, user reqs #1/#4).

    Frame sampling / resolution bucketing mirror the HunyuanVideo & Wan2.1 data
    pipelines (a ``4k+1`` frame count for the 4× causal-temporal VAE, fixed
    resolution buckets, ``[-1, 1]`` normalization). The latent/text cache is the
    dominant *training* memory saving (user requirement #1): the VAE and text
    encoder run once offline so neither occupies VRAM during plugin training.
    """

    data_root: str = ""
    meta_file: str = ""  # jsonl/csv manifest with {video, caption[, scene]}
    # OpenVid-1M layout: clips resolve to ``{data_root}/{video_subdir}/{video}``
    # and all 202 zip parts extract into a single flat folder (§1.1).
    video_subdir: str = "video"
    # Root of the processed six-level store ``LCOCF_OpenVid1M_Processed`` (§3).
    processed_root: str = "LCOCF_OpenVid1M_Processed"
    # --- frame sampling (HunyuanVideo/Wan2.1 convention) ---
    num_frames: int = 49  # 4k+1 for the 4× causal-temporal VAE (k=12)
    frame_interval: int = 1  # temporal stride when sampling source frames
    fps: int = 16
    # --- resolution bucketing: candidate (frames, H, W) the loader snaps to ---
    resolution_buckets: Tuple[Tuple[int, int, int], ...] = (
        (49, 480, 832),
        (49, 720, 1280),
    )
    height: int = 480  # default bucket when aspect-ratio routing is off
    width: int = 832
    normalize_to_unit: bool = True  # videos returned in [-1, 1] (VAE convention)
    # --- caching (the dominant training memory saving, user requirement #1) ---
    cache_dir: str = "cache/latents"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 1234


@dataclass
class FilterConfig:
    """Four-level quality-filter thresholds (§2).

    The filter runs primarily on the OpenVid metadata columns (resolution, seconds,
    aesthetic / motion scores, caption) so it is cheap and decode-free; the optional
    blur/watermark gates are applied only when a perception hook supplies the signal
    (§2.1 Table 0). Defaults mirror the document's stated cut-offs verbatim.
    """

    # L1 — basic hard filter (§2.1, Table 0)
    min_resolution: int = 512  # drop < 512×512; HD (1080p) force-kept
    min_duration_s: float = 2.0
    max_duration_s: float = 15.0
    preferred_min_duration_s: float = 3.0  # 3–10s is the mainstream window
    preferred_max_duration_s: float = 10.0
    black_frame_max_frac: float = 0.30  # > 30% black/garbled ⇒ drop
    watermark_max_frac: float = 0.20  # > 20% watermark/logo/mosaic ⇒ drop
    blur_laplacian_min: float = 0.0  # Laplacian-variance floor (0 ⇒ gate disabled)
    # L2 — semantic filter (§2.2)
    min_caption_words: int = 5
    min_clip_align: float = 0.25
    aesthetic_drop_frac: float = 0.20  # drop the bottom 20% aesthetic
    dedup_sim_threshold: float = 0.95  # caption/fingerprint near-duplicate cut
    # L3 — task-fitness filter (§2.3)
    drop_static: bool = True  # drop pure-static / no-motion / no-semantic-change
    static_motion_max: float = 0.02  # motion score below ⇒ treated as static
    complex_min_frac: float = 0.30  # complex scenes ≥ 30% of the kept set
    # L4 — final sampling (§2.4)
    target_samples: int = 180_000  # 15–20万 high-quality clips
    hd_min_frac: float = 0.60  # OpenVidHD ≥ 60% of the final set
    val_frac: float = 0.10  # 10% validation, split by video_id (no leakage)
    test_hard_frac: float = 0.05  # hard-sample test list (multi/occlusion/text/fast)


@dataclass
class TeacherConfig:
    """Stage-A offline counterfactual teacher-data generation (§7.1.1).

    Encodes the §7.1.1 sampling protocol and its four cost-reduction tricks:
    tube-*group* intervention (one rollout per tube, not per token), full
    counterfactual rollouts only at *representative* timesteps with adjacent-step
    label interpolation, proxy metrics instead of human preference (the injected
    :class:`MetricExtractor`), and scene-balanced prompt sampling.
    """

    out_dir: str = "cache/teacher"
    num_inference_steps: int = 30
    seeds_per_prompt: int = 3  # ≥3 seeds per prompt (§7.1.1)
    # representative ``step_frac = t/T`` values that get a *full* CF rollout; the
    # in-between steps are filled by label interpolation (§7.1.1 cost trick).
    representative_step_fracs: Tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.1)
    interpolate_adjacent_steps: bool = True
    # skip actions probed per (tube, step) — FULL (=0) is the zero-damage reference.
    probe_actions: Tuple[int, ...] = (1, 2, 3)  # LOW FREQ, INTERP, ANCHOR
    max_tubes_per_prompt: int = 8  # cap tube-group interventions per prompt
    samples_per_video: int = 30  # cap on (tube,step,action) labels per video (§1.5)
    scene_balanced: bool = True  # balance static/dynamic/text/face/multi/occlusion
    use_preview_decode_for_tubes: bool = True  # segment a preview decode of z_t
    shard_size: int = 256  # records per on-disk shard


# --------------------------------------------------------------------------- #
# Training (§7.1)
# --------------------------------------------------------------------------- #


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 200
    max_steps: int = 20000
    use_8bit_adam: bool = False


@dataclass
class TrainingConfig:
    """Stage-B/C loss weights and optimisation (§7.1.2, §7.1.3)."""

    # total loss weights: L = L_cocf + λ_sta·L_tube + λ_cert·L_cert + λ_cmsc·L_cmsc + λ_cost·L_budget
    lambda_sta: float = 1.0
    lambda_cert: float = 1.0
    lambda_cmsc: float = 1.0
    lambda_cost: float = 0.10
    optim: OptimConfig = field(default_factory=OptimConfig)
    # Stage-C LoRA fine-tune of the last few DiT blocks (optional, §7.1.3)
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_target_last_n_blocks: int = 4
    log_every: int = 20
    ckpt_every: int = 1000
    # Stage-B validation & early stopping (§4.1): evaluate degradation-prediction
    # MAE, certificate-violation rate, budget-hit rate and action smoothness.
    val_every_epochs: int = 1
    early_stop_patience: int = 3  # epochs without val improvement before stopping


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """Root configuration object wiring every subsystem together."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    tube: TubeConfig = field(default_factory=TubeConfig)
    lcocf: LCOCFConfig = field(default_factory=LCOCFConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    cmsc: CMSCConfig = field(default_factory=CMSCConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 1234

    # -- (de)serialisation ------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return _build_dataclass(cls, data or {})

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load from a YAML (preferred) or JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            data = json.loads(text)
        return cls.from_dict(data or {})

    def save(self, path: str) -> None:
        import json

        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as fh:
            try:
                import yaml  # type: ignore

                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
            except ImportError:
                json.dump(data, fh, indent=2, ensure_ascii=False)


def _build_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively construct a (possibly nested) dataclass from a plain dict.

    Unknown keys are ignored with no error so configs stay forward-compatible;
    nested dataclasses and tuples are reconstructed by type. ``get_type_hints``
    resolves the string annotations produced by ``from __future__ import
    annotations`` back into real types.
    """
    import typing

    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    try:
        resolved = typing.get_type_hints(cls)
    except Exception:  # pragma: no cover - fall back to raw (string) annotations
        resolved = {f.name: f.type for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = resolved.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build_dataclass(ftype, value)  # type: ignore[arg-type]
        elif isinstance(value, list):
            kwargs[f.name] = tuple(value) if "Tuple" in str(ftype) else value
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[call-arg]
