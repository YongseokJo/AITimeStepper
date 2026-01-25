from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import inspect

import torch

from .config import Config


def _tensor_to_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _map_dict_values(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    out: Dict[str, Any] = {}
    for key, value in data.items():
        out[key] = _tensor_to_value(value)
    return out


def _config_payload(config: Optional[Config]) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    return config.as_wandb_dict()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    epoch: Optional[int] = None,
    loss: Optional[Any] = None,
    info: Optional[Dict[str, Any]] = None,
    logs: Optional[Dict[str, Any]] = None,
    config: Optional[Config] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "model_state": model_state,
        "optimizer_state_dict": optimizer_state,
        "optimizer_state": optimizer_state,
        "loss": _tensor_to_value(loss),
        "info": _map_dict_values(info),
        "logs": _map_dict_values(logs),
        "extra": extra,
    }

    if config is not None:
        payload["config"] = _config_payload(config)
        payload["config_summary"] = config.summary()
        payload["history_len"] = config.history_len
        payload["feature_type"] = config.feature_type
        payload["dtype"] = config.dtype

    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: Optional[str | torch.device] = None) -> Dict[str, Any]:
    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)


def load_model_state(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: Optional[str | torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = load_checkpoint(path, map_location=map_location)
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise KeyError("Checkpoint missing model_state_dict/model_state")
    model.load_state_dict(state, strict=strict)
    return ckpt


def load_config_from_checkpoint(path: str | Path) -> Optional[Config]:
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        return Config.from_dict(cfg)
    return None
