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
    # -- head bias initialisation: the cold-start calibration of E_cert ------- #
    # An untrained predictor must not certify every action as catastrophic. With
    # zero biases, softplus(0)=0.693 and exp(0)=1.0 give μ₀≈0.70 and σ₀≈1.0, so
    # E_cert = μ + κσ ≈ 0.70 + 1.96 ≈ 2.7 — 3.4× the τ_high=0.80 trigger. Every tube
    # was rolled back and pinned to FULL on the *second* step, permanently disabling
    # the accelerator on the out-of-the-box config (no checkpoint = README quick
    # start). These set the head biases so the cold start begins *below* τ_low:
    #   μ₀ = mu_init, σ₀ = exp(0.5·log_var_init)  ⇒  E_cert₀ ≈ 0.02 + 1.96·0.135 ≈ 0.29
    # They only affect initialisation; a Stage-B checkpoint overwrites both heads.
    # Force μ[FULL] = 0. FULL is the zero-damage reference the labels (§1.5) and the
    # allocator's benefit/cost arithmetic are both defined against; letting the head
    # emit a free positive value there breaks that premise (§P1-12).
    pin_full_zero: bool = True
    mu_init: float = 0.02  # μ at init (small but non-zero: damage is unknown, not nil)
    log_var_init: float = -4.0  # ⇒ σ₀ ≈ 0.135; κ·σ₀ ≈ 0.26 instead of 1.96


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
    # temporal locality neighbourhood (axiom §3.2.1): ±tau steps
    tau: int = 2
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
    # -- anchor-library admission (deliberately *not* τ_low) ------------------ #
    # τ_low is the trigger boundary "below this, do nothing"; reusing it as the
    # anchor-write gate coupled two unrelated policies and produced a deadlock:
    # a mis-calibrated certificate sat above τ_low, so nothing was ever anchored,
    # so rollback() had no anchor to restore and degenerated into a no-op — while
    # the force-FULL pin it set still fired every step. Anchoring is *cheap and
    # reversible* (a snapshot), so it is admitted on a looser gate.
    tau_anchor: float = 0.60  # E_cert ≤ this ⇒ snapshot the tube as a safe anchor
    # Always snapshot a tube the first time it is computed at acceptable risk, even
    # if its certificate exceeds τ_anchor: an imperfect rollback target beats none.
    seed_anchor_on_first_compute: bool = True


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

    # Per-action relative cost multipliers C(a, |g_k|) ∝ |g_k| (§2.2): the fraction of
    # a tube's tokens the action actually recomputes. These must agree with what the
    # transition executor really does, or the knapsack optimises a fiction — the old
    # (1.0, 0.45, 0.15, 0.0) priced LOWFREQ at 0.45 while a stride-2 lattice computes
    # 1/4 of the tokens, and INTERP at 0.15 though it runs no denoiser at all, so the
    # solver believed a downgrade bought savings that never materialised (§P1-6).
    # The LOWFREQ entry is *derived* from ``EngineConfig.lowfreq_stride`` at
    # construction (1/stride²); the value here is the stride-2 default.
    action_cost: Tuple[float, float, float, float] = (1.0, 0.25, 0.0, 0.0)
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
    lowfreq_stride: int = 2  # LOW FREQ spatial stride (2 ⇒ ~1/4 tokens computed)
    # Opt-in throughput heuristic, **off by default**. On a backbone without
    # token-sparse attention a mask this sparse costs a full dense forward anyway, so
    # the transition executor can promote it to a whole-step cache reuse — a real
    # saving. The catch is that a promoted step silently downgrades any LOWFREQ tube
    # to "reuse cached ε", and there is then no computed reference to measure that
    # tube's skip residual against, so RAEC's certificate sees δ=0 and cannot price
    # the extra error (§5.3.1). Enable it only when throughput matters more than the
    # risk layer's coverage. A step where *every* tube skips is still skipped for free
    # without this — that path needs no heuristic and keeps the certificate honest.
    dense_step_skip_below: float = 0.0
    # Recompute the tokens no tube covers every N steps (0 = never). The background
    # is ~3/4 of the grid and is outside every RAEC guarantee — certificates, rollback
    # and repair are all tube-scoped — so leaving it on the warm-up step's ε for the
    # whole trajectory is an unmonitored quality loss (§P1-8). On a dense-only backbone
    # this costs nothing: the forward runs at full price whatever the mask says.
    background_refresh_every: int = 4
    measure_residual: bool = True  # measure skip residuals for the certificate
    cf_check_enabled: bool = True  # run §3.3.4 single-hop counterfactual checks
    use_dynamic_budget: bool = True  # else spend b_max every step
    risk_control_enabled: bool = True  # enable RAEC trigger/repair at inference
    # Stage-C truncated BPTT (§4.2), in the classic segment sense: the graph is cut
    # every ``grad_window_steps`` *computed* denoising steps (steps whose denoiser
    # actually ran — skipped steps add no activations and carry no gradient). Counting
    # computed steps rather than wall-clock ones is what keeps the LoRA branch
    # trainable on an accelerated trajectory: a run that computes 2 of 30 steps must
    # not have its graph cut simply because those 2 were early.
    # Note this is a periodic reset, not a sliding window — at the final decode the
    # graph holds between 1 and ``grad_window_steps`` computed steps depending on
    # where the last cut landed. Peak activation memory is what the bound buys.
    # 0 = full BPTT (real backbones will OOM). Ignored at inference (decode_grad=False).
    grad_window_steps: int = 4


# --------------------------------------------------------------------------- #
# Backbone & memory (multi-model compat + memory savings: user reqs #1, #2)
# --------------------------------------------------------------------------- #


@dataclass
class BackboneConfig:
    """Which backbone to wrap and where its weights live (§9.1)."""

    name: str = "mock"  # registry key: "wan22" (primary) | "wan21" | "hunyuanvideo" | "mock"
    model_path: Optional[str] = None
    dtype: str = "bfloat16"  # compute dtype for the frozen backbone
    device: str = "cuda"
    # -- VRAM residency policy for the frozen stack (user requirement #1) ----- #
    # A real backbone keeps every component resident for the whole run, which on
    # Wan2.2-A14B is ~67 GB (2×14B experts + a 5.5B umT5) — the entire budget of an
    # 80 GB card, leaving nothing for the VAE decode / metric activations Stage A
    # needs. These switches park the components that are idle most of the time.
    # They default **on**: an OOM tens of minutes into a run is far more expensive
    # than the transfers, so the safe setting is the default and the CLI's --no-*
    # flags buy the speed back on cards with room to spare.
    offload_text_encoder: bool = True  # park the text encoder on CPU between prompts (saves ~11 GB)
    offload_idle_expert: bool = True   # keep only the active MoE expert resident (saves ~28 GB)
    vae_tiling: bool = True            # tiled/sliced VAE encode+decode (bounded peak)
    # Tile edge (output pixels) when vae_tiling is on. The decoder's peak transient
    # scales with tile_size². 128 bounds a 480×832 decode at ~0.4 GB (vs 1.4 GB at
    # 256, vs 7.7 GiB untiled) — necessary headroom on 40 GB cards.
    vae_tile_size: int = 128
    # backbone-specific knobs passed straight through to the adapter
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Training memory-saving switches (user requirement #1, realised in §7.1)."""

    amp_dtype: str = "bfloat16"  # autocast dtype; "none" disables AMP
    gradient_checkpointing: bool = True  # checkpoint trainable blocks
    offload_backbone_to_cpu: bool = False  # keep frozen backbone on CPU until needed
    cache_latents: bool = True  # pre-encode videos → latents on disk (no VAE in RAM)
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
    num_inference_steps: int = 20  # reduced from 30 for throughput (40GB-card friendly)
    seeds_per_prompt: int = 2  # reduced from 3: 2 seeds still give variance, 3× faster
    # ``step_frac = t/T`` values that get a full CF rollout. Every generated sample
    # sits on one of these, so this list *is* the timestep coverage the predictor
    # learns from. 3 points give early/mid/late coverage at ~40% fewer rollouts than
    # 5. (§7.1.1 also proposes interpolating labels for the in-between steps; that is
    # not implemented — the generator samples only at these steps.)
    representative_step_fracs: Tuple[float, ...] = (0.8, 0.5, 0.2)
    # skip actions probed per (tube, step) — FULL (=0) is the zero-damage reference.
    probe_actions: Tuple[int, ...] = (1, 2, 3)  # LOW FREQ, INTERP, ANCHOR
    max_tubes_per_prompt: int = 4  # reduced from 8: halves rollout count per clip
    samples_per_video: int = 15  # reduced from 30: matches 3 steps × 4 tubes × ~1 action
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
