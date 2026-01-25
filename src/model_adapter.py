from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .config import Config
from .history_buffer import HistoryBuffer
from .normalization import apply_norm_scales_to_config, derive_norm_scales_from_config
from .particle import ParticleTorch


@dataclass
class ModelAdapter:
    config: Config
    device: torch.device
    dtype: torch.dtype
    history_buffer: Optional[HistoryBuffer] = None
    norm_scales: Optional[dict] = None

    def __init__(self, config: Config, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.config = config
        self.device = device or config.resolve_device()
        self.dtype = dtype or config.resolve_dtype()
        self.config.apply_torch_settings(self.device)
        self.history_buffer = None
        self.norm_scales = None
        if config.history_len and config.history_len > 0:
            self.history_buffer = HistoryBuffer(
                history_len=config.history_len,
                feature_type=config.feature_type,
                norm_scales=None,
            )
        if config.normalize_inputs and config.norm_mode == "manual":
            self.set_norm_scales(derive_norm_scales_from_config(config))

    @property
    def history_enabled(self) -> bool:
        return self.config.history_len is not None and self.config.history_len > 0

    def feature_mode(self) -> str:
        if self.config.feature_type in ("basic", "rich"):
            return self.config.feature_type
        return "basic"

    def build_feature_tensor(
        self,
        state: ParticleTorch,
        *,
        histories: Optional[Sequence[HistoryBuffer]] = None,
        history_buffer: Optional[HistoryBuffer] = None,
    ) -> torch.Tensor:
        if self.history_enabled:
            hb = history_buffer or self.history_buffer
            if histories is not None:
                feats = HistoryBuffer.features_for_histories(histories, state)
            elif hb is not None:
                feats = hb.features_for(state)
            else:
                raise RuntimeError("History-enabled adapter requires a HistoryBuffer")
        else:
            feats = state.system_features(mode=self.feature_mode(), norm_scales=self.norm_scales)
        if self.config.debug and not torch.isfinite(feats).all():
            self._debug_tensor("features", feats)
            self._debug_tensor("position", state.position)
            self._debug_tensor("velocity", state.velocity)
            self._debug_tensor("mass", state.mass)
        return feats

    def input_dim_from_state(
        self,
        state: ParticleTorch,
        *,
        histories: Optional[Sequence[HistoryBuffer]] = None,
        history_buffer: Optional[HistoryBuffer] = None,
    ) -> int:
        with torch.no_grad():
            feats = self.build_feature_tensor(state, histories=histories, history_buffer=history_buffer)
            if feats.dim() == 1:
                return int(feats.numel())
            return int(feats.shape[-1])

    def predict_dt(
        self,
        model: torch.nn.Module,
        state: ParticleTorch,
        *,
        histories: Optional[Sequence[HistoryBuffer]] = None,
        history_buffer: Optional[HistoryBuffer] = None,
        eps: float = 1e-6,
    ):
        feats = self.build_feature_tensor(state, histories=histories, history_buffer=history_buffer)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        feats = feats.to(device=self.device, dtype=self.dtype)

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            params = model(feats)
            if params.dim() == 1:
                params = params.unsqueeze(0)
            dt_raw = params[:, 0]

        dt_tensor = dt_raw + eps
        if dt_tensor.numel() == 1:
            return float(dt_tensor.squeeze(0).item())
        return dt_tensor

    def set_norm_scales(self, norm_scales: Optional[dict]) -> None:
        self.norm_scales = norm_scales
        apply_norm_scales_to_config(self.config, norm_scales)
        if self.history_buffer is not None:
            self.history_buffer.norm_scales = norm_scales

    def _debug_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            print(f"debug {name}: non-tensor {type(tensor)}")
            return
        t = tensor.detach()
        finite_mask = torch.isfinite(t)
        finite = bool(finite_mask.all().item())
        numel = t.numel()
        finite_count = int(finite_mask.sum().item())
        if finite_count > 0:
            finite_vals = t[finite_mask]
            tmin = finite_vals.min().item()
            tmax = finite_vals.max().item()
            tmean = finite_vals.mean().item()
        else:
            tmin = float("nan")
            tmax = float("nan")
            tmean = float("nan")
        print(
            f"debug {name}: finite={finite} count={finite_count}/{numel} "
            f"min={tmin:.4e} max={tmax:.4e} mean={tmean:.4e}"
        )

    def update_history(
        self,
        state: ParticleTorch,
        *,
        history_buffer: Optional[HistoryBuffer] = None,
        token: Optional[tuple] = None,
    ) -> None:
        if not self.history_enabled:
            return
        hb = history_buffer or self.history_buffer
        if hb is None:
            return
        if token is not None:
            last_token = getattr(hb, "_last_push_token", None)
            if last_token == token:
                return
            hb._last_push_token = token
        hb.push(state)
