from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


@dataclass
class Config:
    # Training / optimization
    optimizer: str = "adam"
    epochs: int = 1000
    lr: float = 1e-4
    weight_decay: float = 1e-2
    momentum: float = 0.9
    n_steps: int = 10
    dt_bound: float = 1e-8
    rel_loss_bound: float = 1e-5
    energy_threshold: float = 2e-4
    steps_per_epoch: int = 1
    max_retrain_steps: int = 1000
    replay_steps: int = 1000
    replay_batch_size: int = 512
    min_replay_size: int = 2

    # Loss bounds
    E_lower: float = 1e-6
    E_upper: float = 1e-4
    L_lower: float = 1e-4
    L_upper: float = 1e-2

    # Feature / history
    history_len: int = 0
    feature_type: str = "delta_mag"
    normalize_inputs: bool = False
    norm_mode: str = "auto"  # "auto" or "manual"
    norm_L0: Optional[float] = None
    norm_V0: Optional[float] = None
    norm_A0: Optional[float] = None
    norm_T0: Optional[float] = None

    # Simulation / integrator
    integrator_mode: str = "analytic"
    dt: float = -1.0
    steps: int = -1
    duration: float | None = None
    Nperiod: int = 10
    eccentricity: float = 0.9
    semi_major: float = 1.0
    eps: float = 0.1
    num_particles: int = 2
    dim: int = 2
    mass: float = 1.0
    pos_scale: float = 0.1
    vel_scale: float = 1.0

    # External field (optional)
    external_field_mass: Optional[float] = None
    external_field_position: Optional[Tuple[float, float, float]] = None

    # Multi-orbit sampling (history multi)
    num_orbits: int = 8
    stop_at_period: bool = False
    e_min: float = 0.6
    e_max: float = 0.95
    a_min: float = 0.8
    a_max: float = 1.2

    # Device / dtype
    device: str = "auto"
    dtype: str = "float64"
    tf32: bool = False
    compile: bool = False
    detect_anomaly: bool = False

    # Model Architecture
    hidden_dims: Iterable[int] = field(default_factory=lambda: [200, 1000, 1000, 200])
    activation: str = "tanh"
    dropout: float = 0.2
    fourier_scale: float = -1.0
    fourier_dim: int = 256

    # Optimizer phases
    adam_epochs: int = 0
    adam_lr: Optional[float] = None
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 500
    lbfgs_history_size: int = 50
    lbfgs_line_search: str = "strong_wolfe"

    # Logging / misc
    save_name: Optional[str] = None
    seed: Optional[int] = None
    debug: bool = False
    debug_every: int = 1
    debug_replay_every: int = 10

    # Movie generation (simulation)
    movie: bool = False
    movie_dir: Optional[str] = None
    movie_max_frames: int = 1000
    movie_fps: int = 30
    movie_dpi: int = 120

    # Checkpoint / model loading
    model_path: Optional[str] = None

    # Extra / unknown fields
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, include: Optional[Iterable[str]] = None) -> None:
        include_set = set(include) if include is not None else None

        def want(key: str) -> bool:
            return include_set is None or key in include_set

        def add_arg(*args: Any, **kwargs: Any) -> None:
            for option in args:
                if option in parser._option_string_actions:
                    return
            parser.add_argument(*args, **kwargs)

        if want("train"):
            add_arg("--epochs", "-n", type=int, default=cls.epochs, help="number of training epochs")
            add_arg("--optimizer", "-o", type=str, default=cls.optimizer, help="optimizer type")
            add_arg("--lr", type=float, default=cls.lr, help="learning rate")
            add_arg("--momentum", type=float, default=cls.momentum, help="SGD momentum")
            add_arg("--weight-decay", type=float, default=cls.weight_decay, help="optimizer weight decay")
            add_arg("--n-steps", type=int, default=cls.n_steps, help="integration steps per loss eval")
            add_arg("--dt-bound", type=float, default=cls.dt_bound, help="dt bound (for loss heuristics)")
            add_arg("--rel-loss-bound", type=float, default=cls.rel_loss_bound, help="relative loss bound")
            add_arg("--energy-threshold", type=float, default=cls.energy_threshold, help="accept/reject energy threshold")
            add_arg("--steps-per-epoch", type=int, default=cls.steps_per_epoch, help="number of training steps per epoch")
            add_arg(
                "--max-retrain-steps",
                type=int,
                default=cls.max_retrain_steps,
                help="max retrain iterations per step (0 = unlimited)",
            )
            add_arg("--replay-steps", type=int, default=cls.replay_steps, help="max replay optimization steps per epoch")
            add_arg("--replay-batch-size", type=int, default=cls.replay_batch_size, help="replay batch size")
            add_arg("--replay-batch", type=int, default=cls.replay_batch_size, help="replay buffer batch size")
            add_arg("--min-replay-size", type=int, default=cls.min_replay_size, help="min replay buffer size before training")
            add_arg("--hidden-dims", type=int, nargs="+", default=[200, 1000, 1000, 200], help="hidden layer dimensions")
            add_arg("--activation", type=str, default="tanh", help="activation function (tanh, relu, etc.)")
            add_arg("--dropout", type=float, default=0.2, help="dropout probability")
            add_arg("--fourier-scale", type=float, default=-1.0, help="scale of gaussian fourier features (<0 to disable)")
            add_arg("--fourier-dim", type=int, default=256, help="dimension of fourier feature embedding")

        if want("bounds"):
            add_arg("--E_lower", type=float, default=cls.E_lower, help="lower energy bound for loss calculation")
            add_arg("--E_upper", type=float, default=cls.E_upper, help="upper energy bound for loss calculation")
            add_arg("--L_lower", type=float, default=cls.L_lower, help="lower angular momentum bound for loss calculation")
            add_arg("--L_upper", type=float, default=cls.L_upper, help="upper angular momentum bound for loss calculation")

        if want("history"):
            add_arg("--history-len", type=int, default=cls.history_len, help="number of past states to include")
            add_arg(
                "--feature-type",
                type=str,
                choices=["basic", "rich", "delta_mag"],
                default=cls.feature_type,
                help="feature type per state",
            )
            add_arg("--normalize-inputs", action="store_true", help="normalize model inputs")
            add_arg(
                "--norm-mode",
                type=str,
                choices=["auto", "manual"],
                default=cls.norm_mode,
                help="normalization mode (auto computes from ICs)",
            )
            add_arg("--norm-L0", type=float, default=cls.norm_L0, help="manual length scale")
            add_arg("--norm-V0", type=float, default=cls.norm_V0, help="manual velocity scale")
            add_arg("--norm-A0", type=float, default=cls.norm_A0, help="manual acceleration scale")
            add_arg("--norm-T0", type=float, default=cls.norm_T0, help="manual time scale")

        if want("orbit"):
            add_arg("--eccentricity", "-e", type=float, default=cls.eccentricity, help="eccentricity for generate_IC")
            add_arg("--semi-major", "-a", type=float, default=cls.semi_major, help="semi-major axis for generate_IC")

        if want("multi"):
            add_arg("--num-orbits", type=int, default=cls.num_orbits, help="number of independent orbits to batch together")
            add_arg("--stop-at-period", action="store_true", help="stop trajectory collection when orbit period is reached")
            add_arg("--e-min", type=float, default=cls.e_min, help="minimum eccentricity for sampling initial conditions")
            add_arg("--e-max", type=float, default=cls.e_max, help="maximum eccentricity for sampling initial conditions")
            add_arg("--a-min", type=float, default=cls.a_min, help="minimum semi-major axis for sampling initial conditions")
            add_arg("--a-max", type=float, default=cls.a_max, help="maximum semi-major axis for sampling initial conditions")

        if want("device"):
            add_arg("--device", type=str, choices=["auto", "cpu", "cuda"], default=cls.device, help="compute device")
            add_arg("--dtype", type=str, choices=["float32", "float64"], default=cls.dtype, help="tensor dtype")
            add_arg("--tf32", action="store_true", help="enable TF32 matmul on CUDA")
            add_arg("--compile", action="store_true", help="use torch.compile for model")
            add_arg("--detect-anomaly", action="store_true", help="enable autograd anomaly detection (slow)")

        if want("logging"):
            add_arg("--save-name", "-s", type=str, default=cls.save_name, help="base filename/dir under data/ to save outputs")
            add_arg("--seed", type=int, default=cls.seed, help="random seed for reproducibility")
            add_arg("--debug", action="store_true", help="enable debug printouts")
            add_arg("--debug-every", type=int, default=cls.debug_every, help="print debug info every N epochs")
            add_arg("--debug-replay-every", type=int, default=cls.debug_replay_every, help="print debug info every N replay steps")

        if want("sim"):
            add_arg("--dt", type=float, default=cls.dt, help="Time step for integrator")
            add_arg("--steps", type=int, default=cls.steps, help="Number of steps to evolve per period")
            add_arg("--duration", type=float, default=cls.duration, help="simulation or training duration in seconds")
            add_arg("--Nperiod", type=int, default=cls.Nperiod, help="Number of period")
            add_arg("--eps", type=float, default=cls.eps, help="time constant for dt update")
            add_arg("--integrator-mode", type=str, choices=["analytic", "ml", "history"], default=cls.integrator_mode)
            add_arg("--model-path", type=str, default=cls.model_path, help="Path to the model checkpoint to load")
            add_arg("--num-particles", type=int, default=cls.num_particles, help="number of particles in the system")
            add_arg("--dim", type=int, default=cls.dim, help="spatial dimension")
            add_arg("--mass", type=float, default=cls.mass, help="per-particle mass for random ICs")
            add_arg("--pos-scale", type=float, default=cls.pos_scale, help="position scale for random ICs")
            add_arg("--vel-scale", type=float, default=cls.vel_scale, help="velocity scale for random ICs")
            add_arg("--movie", action="store_true", help="save MP4 movie of the simulation")
            add_arg("--movie-dir", type=str, default=cls.movie_dir, help="output directory for movies")
            add_arg("--movie-max-frames", type=int, default=cls.movie_max_frames, help="max frames for movie downsampling (<=0 to include all frames)")
            add_arg("--movie-fps", type=int, default=cls.movie_fps, help="movie frames per second")
            add_arg("--movie-dpi", type=int, default=cls.movie_dpi, help="movie rendering DPI")

        if want("external"):
            add_arg("--external-field-mass", type=float, default=cls.external_field_mass, help="external field mass")
            add_arg(
                "--external-field-position",
                type=float,
                nargs=3,
                default=cls.external_field_position,
                metavar=("X", "Y", "Z"),
                help="external field position (x y z)",
            )

    @classmethod
    def from_cli(
        cls,
        parser_or_args: argparse.ArgumentParser | argparse.Namespace,
        *,
        include: Optional[Iterable[str]] = None,
        argv: Optional[Iterable[str]] = None,
    ) -> "Config":
        if isinstance(parser_or_args, argparse.ArgumentParser):
            parser = parser_or_args
            cls.add_cli_args(parser, include=include)
            args = parser.parse_args(argv)
        else:
            args = parser_or_args
        return cls.from_dict(vars(args))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        payload = dict(data) if data is not None else {}
        field_names = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in field_names and key != "extra":
                kwargs[key] = value
            else:
                extra[key] = value
        if "external_field_position" in kwargs and isinstance(kwargs["external_field_position"], list):
            kwargs["external_field_position"] = tuple(kwargs["external_field_position"])
        kwargs["extra"] = extra
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return data

    def as_wandb_dict(self) -> Dict[str, Any]:
        data = self.to_dict()
        extra = data.pop("extra", {}) or {}
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key not in data:
                    data[key] = value
        if isinstance(data.get("external_field_position"), tuple):
            data["external_field_position"] = list(data["external_field_position"])
        return data

    def summary(self) -> str:
        return (
            f"epochs={self.epochs} n_steps={self.n_steps} lr={self.lr} wd={self.weight_decay} "
            f"E_bounds=({self.E_lower},{self.E_upper}) L_bounds=({self.L_lower},{self.L_upper}) "
            f"history_len={self.history_len} feature_type={self.feature_type} device={self.device} dtype={self.dtype}"
        )

    def validate(self) -> None:
        if self.history_len and self.history_len > 0 and not self.feature_type:
            raise ValueError("history_len > 0 requires feature_type")
        if self.normalize_inputs and self.norm_mode not in {"auto", "manual"}:
            raise ValueError("norm_mode must be 'auto' or 'manual'")
        if self.normalize_inputs and self.norm_mode == "manual":
            if self.norm_L0 is None or self.norm_V0 is None:
                raise ValueError("manual normalization requires norm_L0 and norm_V0")
        if self.num_particles is not None and self.num_particles < 2:
            raise ValueError("num_particles must be >= 2")
        if self.dim is not None and self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.duration is not None and self.duration < 0:
            raise ValueError("duration must be >= 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1")
        if self.energy_threshold <= 0:
            raise ValueError("energy_threshold must be > 0")

    def resolve_device(self) -> torch.device:
        if isinstance(self.device, torch.device):
            return self.device
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resolve_dtype(self) -> torch.dtype:
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        if str(self.dtype) == "float32":
            return torch.float32
        return torch.float64

    def apply_torch_settings(self, device: Optional[torch.device] = None) -> None:
        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        resolved = device or self.resolve_device()
        if self.tf32 and resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def history_feature(self) -> str:
        return self.feature_type
