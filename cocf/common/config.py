"""Typed hierarchical configuration for COCF-SS-DCA."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


@dataclass
class AffinityConfig:
    """Cross-frame region affinity weights."""

    w_id: float = 0.40
    w_flow: float = 0.30
    w_iou: float = 0.15
    w_txt: float = 0.10
    w_pos: float = 0.05
    sigma_p: float = 16.0
    flow_scale: float = 1.0


@dataclass
class TubeConfig:
    """Semantic-tube construction config."""

    affinity: AffinityConfig = field(default_factory=AffinityConfig)
    min_region_ratio: float = 0.001
    min_clip_score: float = 0.20
    max_tube_len: int = 16
    affinity_match_threshold: float = 0.30
    identity_unstable_threshold: float = 0.50
    lambda_temporal: float = 0.20
    lambda_boundary: float = 0.10


@dataclass
class StrengthConfig:
    """Causal-strength model config."""

    alpha_init: float = 1.0
    beta_init: float = 1.0
    gamma_init: float = 1.0
    theta1: float = 0.66
    theta2: float = 0.33
    normalize_strength: bool = True


@dataclass
class PredictorConfig:
    """Counterfactual damage predictor config."""

    state_dim: int = 7
    context_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    num_actions: int = 4
    dropout: float = 0.0
    predict_log_variance: bool = True
    pin_full_zero: bool = True
    mu_init: float = 0.02
    log_var_init: float = -4.0


@dataclass
class CounterfactualConfig:
    """Counterfactual verification config."""

    theta_sT: float = 0.50
    eta: float = 0.05
    max_checks_per_step: int = 4
    repair_net_dim: int = 128


@dataclass
class LCOCFConfig:
    """L-COCF module config."""

    strength: StrengthConfig = field(default_factory=StrengthConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    tau: int = 2
    vlm_name: str = "frozen-vlm"


@dataclass
class CertificateConfig:
    """Error-certificate coefficients and training weights."""

    kappa: float = 1.96  # uncertainty multiplier (95% one-sided)
    lambda_res: float = 0.10  # residual-to-anchor term
    lambda_bnd: float = 0.05  # boundary-gradient term
    lambda_age: float = 0.01  # anchor-age term
    lambda_cmsc: float = 0.20  # local CMSC term
    alpha_cert: float = 0.10  # penalty weight for exceeding tau_safe
    tau_safe: float = 0.20


@dataclass
class TriggerConfig:
    """Risk trigger and local repair config."""

    tau_low: float = 0.40
    tau_high: float = 0.80
    force_full_steps: int = 2  # steps to force FULL after a rollback
    sigma_bnd: float = 4.0  # boundary soft-mask bandwidth
    tau_anchor: float = 0.60  # snapshot the tube as a safe anchor when E_cert <= this
    seed_anchor_on_first_compute: bool = True  # always snapshot on first acceptable compute


@dataclass
class CMSCConfig:
    """Cross-modal semantic conservation loss weights."""

    lambda_align: float = 0.30  # text-tube alignment
    lambda_id: float = 0.20  # identity (DINO)
    lambda_motion: float = 0.20  # motion (RAFT)
    lambda_spatial: float = 0.15  # spatial relations
    lambda_ocr: float = 0.10  # OCR / text
    lambda_bnd: float = 0.05  # boundary
    temperature: float = 0.07  # softmax temperature
    align_dim: int = 256  # text-tube projection dim


@dataclass
class BudgetConfig:
    """Dynamic per-step compute budget config."""

    b_min: float = 0.30  # min fraction of full compute
    b_max: float = 1.00  # max fraction of full compute
    eta_scene: float = 0.10  # weight on scene complexity
    eta_uncertainty: float = 0.15  # weight on mean damage uncertainty
    eta_interaction: float = 0.10  # weight on tube interaction density
    q_early_boost: float = 0.30  # U-shaped time weight: early boost
    q_late_boost: float = 0.40  # U-shaped time weight: late boost
    q_mid_floor: float = 0.10  # U-shaped time weight: mid floor


@dataclass
class AllocatorConfig:
    """Budget-constrained action allocation config."""

    action_cost: Tuple[float, float, float, float] = (1.0, 0.25, 0.02, 0.0)  # per-action relative cost (FULL, LOWFREQ, INTERP, ANCHOR)
    risk_threshold: float = 0.80  # hard risk constraint
    greedy_fallback: bool = True  # use greedy knapsack if no LP solver


@dataclass
class EngineConfig:
    """Accelerated denoising loop config."""

    num_inference_steps: int = 30
    tube_build_step: int = 1  # step at which tubes are built (counted from start)
    tube_refresh_every: int = 0  # re-segment every N steps (0 = never)
    lowfreq_stride: int = 2  # LOWFREQ spatial stride
    dense_step_skip_below: float = 0.0  # promote steps with mask occupancy <= this to a whole-step skip
    max_unmeasured_steps: int = 3  # max steps a tube may go without a measured skip residual (0 = unbounded)
    force_all_full: bool = False  # debug: pin every tube to FULL through the engine path
    diagnostic_lowfreq_full: bool = False
    diagnostic_no_cache: bool = False
    lowfreq_fill: bool = False  # True: fill LOWFREQ holes with the lattice reconstruction (ablation)
    lowfreq_refresh_every: int = 2  # promote LOWFREQ tubes to FULL every N steps (0 = never)
    background_refresh_every: int = 4  # recompute un-tubed background every N steps (0 = never)
    measure_residual: bool = True  # measure skip residuals for the certificate
    cf_check_enabled: bool = True  # run single-hop counterfactual checks
    use_dynamic_budget: bool = True  # else spend b_max every step
    risk_control_enabled: bool = True  # enable RAEC trigger/repair at inference
    grad_window_steps: int = 4  # truncated BPTT window in computed steps (0 = full BPTT)
    log_every_steps: int = 1
    decode_grad_frames: int = 8  # temporal slots decoded on the autograd graph in Stage C (0 = all)


@dataclass
class BackboneConfig:
    """Backbone selection, weights location and VRAM residency policy."""

    name: str = "mock"  # registry key: wan22 | wan21 | hunyuanvideo | mock
    model_path: Optional[str] = None
    dtype: str = "bfloat16"  # compute dtype for the frozen backbone
    device: str = "cuda"
    offload_text_encoder: bool = True  # park the text encoder on CPU between prompts
    text_encoder_exclusive: bool = True  # offload the denoiser while the text encoder is resident
    offload_idle_expert: bool = True  # keep only the active MoE expert resident
    offload_device: str = "cpu"  # where offloaded components sleep ("cpu" or e.g. "cuda:1")
    vae_tiling: bool = True  # tiled/sliced VAE encode+decode
    vae_tile_size: int = 128  # tile edge in output pixels
    extra: Dict[str, Any] = field(default_factory=dict)  # backbone-specific knobs


@dataclass
class MemoryConfig:
    """Training memory-saving switches."""

    amp_dtype: str = "bfloat16"  # autocast dtype; "none" disables AMP
    gradient_checkpointing: bool = True  # checkpoint the LoRA-bearing DiT blocks
    offload_backbone_to_cpu: bool = False  # keep idle anchor snapshots off the compute device
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """Video+caption dataset reading and resolution bucketing config."""

    data_root: str = ""
    meta_file: str = ""  # jsonl/csv manifest with {video, caption[, scene]}
    video_subdir: str = "video"  # subdirectory holding the video clips
    processed_root: str = "LCOCF_OpenVid1M_Processed"  # root of the processed store
    num_frames: int = 49  # 4k+1 for the 4x causal-temporal VAE
    frame_interval: int = 1  # temporal stride when sampling source frames
    fps: int = 16
    resolution_buckets: Tuple[Tuple[int, int, int], ...] = (
        (49, 480, 832),
        (49, 720, 1280),
    )
    height: int = 480  # default bucket when aspect-ratio routing is off
    width: int = 832
    normalize_to_unit: bool = True  # videos returned in [-1, 1]
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 1234


@dataclass
class FilterConfig:
    """Four-level quality-filter thresholds."""

    min_resolution: int = 512  # L1: drop below 512x512; HD force-kept
    min_duration_s: float = 2.0
    max_duration_s: float = 15.0
    preferred_min_duration_s: float = 3.0
    preferred_max_duration_s: float = 10.0
    black_frame_max_frac: float = 0.30  # drop when exceeded
    watermark_max_frac: float = 0.20  # drop when exceeded
    blur_laplacian_min: float = 0.0  # Laplacian-variance floor (0 = disabled)
    min_caption_words: int = 5  # L2: semantic filter
    min_clip_align: float = 0.25
    aesthetic_drop_frac: float = 0.20  # drop the bottom fraction by aesthetic score
    drop_static: bool = True  # L3: drop static / no-semantic-change clips
    static_motion_max: float = 0.02  # motion score below this counts as static
    complex_min_frac: float = 0.30  # complex scenes kept at at least this fraction
    target_samples: int = 180_000  # L4: final sample count
    hd_min_frac: float = 0.60  # minimum OpenVidHD fraction
    val_frac: float = 0.10  # validation split fraction (by video_id)
    test_hard_frac: float = 0.05  # hard-sample test list fraction


@dataclass
class TeacherConfig:
    """Stage-A offline counterfactual teacher-data generation config."""

    out_dir: str = "cache/teacher"
    num_inference_steps: int = 20
    seeds_per_prompt: int = 2
    representative_step_fracs: Tuple[float, ...] = (0.8, 0.5, 0.2)  # step fractions receiving a full CF rollout
    probe_actions: Tuple[int, ...] = (1, 2, 3)  # skip actions probed per (tube, step)
    max_tubes_per_prompt: int = 4
    samples_per_video: int = 12
    free_memory_every: int = 0  # empty_cache cadence inside the rollout loop (0 = never)
    scene_balanced: bool = True  # balance scene categories when sampling
    use_preview_decode_for_tubes: bool = True  # segment a preview decode of z_t
    shard_size: int = 256  # records per on-disk shard
    lmdb_map_size_gib: int = 256  # virtual map size reserved for the LMDB store


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 200


@dataclass
class TrainingConfig:
    """Stage-B/C loss weights and optimisation config."""

    lambda_sta: float = 1.0  # tube-state loss weight
    lambda_cert: float = 1.0  # certificate loss weight
    lambda_cmsc: float = 1.0  # CMSC loss weight
    lambda_cost: float = 0.10  # budget loss weight
    optim: OptimConfig = field(default_factory=OptimConfig)
    lora_rank: int = 16  # Stage-C LoRA rank
    lora_alpha: float = 16.0
    lora_target_last_n_blocks: int = 4  # fine-tune the last N DiT blocks
    log_every: int = 20
    val_every_epochs: int = 1
    early_stop_patience: int = 3  # epochs without val improvement before stopping


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
    """Recursively construct a nested dataclass from a plain dict, ignoring unknown keys."""
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
            def convert_tuple(v, annotation):
                if typing.get_origin(annotation) is tuple and isinstance(v, (list, tuple)):
                    args = typing.get_args(annotation)
                    return tuple(convert_tuple(x, args[0] if len(args) == 2 and args[1] is Ellipsis else args[i])
                                 for i, x in enumerate(v))
                return v
            kwargs[f.name] = convert_tuple(value, ftype)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[call-arg]
