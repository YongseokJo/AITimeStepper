from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .config import Config
from .history_buffer import HistoryBuffer
from .particle import ParticleTorch


@dataclass
class ModelAdapter:
    config: Config
    device: torch.device
    dtype: torch.dtype
    history_buffer: Optional[HistoryBuffer] = None

    def __init__(self, config: Config, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.config = config
        self.device = device or config.resolve_device()
        self.dtype = dtype or config.resolve_dtype()
        self.config.apply_torch_settings(self.device)
        self.history_buffer = None
        if config.history_len and config.history_len > 0:
            self.history_buffer = HistoryBuffer(history_len=config.history_len, feature_type=config.feature_type)

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
            feats = state.system_features(mode=self.feature_mode())
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
