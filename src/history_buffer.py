from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, List, Literal, Sequence

import torch

from .nbody_features import system_features as nbody_system_features
from .nbody_features import _compute_acceleration as nbody_compute_acceleration

if TYPE_CHECKING:
    from .particle import ParticleTorch


FeatureType = Literal["basic", "rich", "delta_mag"]


@dataclass(frozen=True)
class _HistoryState:
    position: torch.Tensor
    velocity: torch.Tensor
    mass: torch.Tensor
    dt: torch.Tensor
    softening: float


class HistoryBuffer:
    """
    Stores the last K particle states (detached tensors) and provides a
    time-concatenated feature vector for the current step.

        - feature_type="basic": fixed-size N-body system stats
        - feature_type="rich": richer fixed-size N-body system stats
        - feature_type="delta_mag": stats of |Δx|, |Δv|, |Δa| plus dt

    For single-system usage, returns a 1D feature vector of length
    (K+1) * F, where K is `history_len` and F is per-step feature size.

    If there are fewer than K past states available, repeats the oldest
    available state to fill the sequence (left-padding).
    """

    def __init__(self, history_len: int = 3, feature_type: FeatureType = "delta_mag"):
        assert history_len >= 0
        self.history_len: int = int(history_len)
        self.feature_type: FeatureType = feature_type
        self._buf: Deque[_HistoryState] = deque(maxlen=self.history_len)

    def reset(self):
        self._buf.clear()

    @staticmethod
    def _normalize_mass(mass: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        target_shape = position.shape[:-1]  # (..., N)
        if mass.dim() == 0:
            return mass.expand(target_shape)
        if mass.shape == target_shape:
            return mass
        if mass.dim() == 1 and mass.shape[0] == target_shape[-1]:
            view_shape = (1,) * (len(target_shape) - 1) + (target_shape[-1],)
            return mass.reshape(view_shape).expand(target_shape)
        if mass.dim() < len(target_shape):
            view_shape = (1,) * (len(target_shape) - mass.dim()) + tuple(mass.shape)
            return mass.reshape(view_shape).expand(target_shape)
        if mass.shape[-1] != target_shape[-1]:
            raise ValueError(f"mass shape {mass.shape} incompatible with position shape {position.shape}")
        return mass

    @staticmethod
    def _normalize_dt(dt: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        target_shape = position.shape[:-2]  # leading dims only
        if len(target_shape) == 0:
            return dt.squeeze()
        if dt.shape == target_shape:
            return dt
        if dt.dim() == 0:
            return dt.expand(target_shape)
        numel_target = 1
        for s in target_shape:
            numel_target *= s
        if dt.numel() == numel_target:
            return dt.reshape(target_shape)
        if dt.dim() < len(target_shape):
            view_shape = (1,) * (len(target_shape) - dt.dim()) + tuple(dt.shape)
            return dt.reshape(view_shape).expand(target_shape)
        return dt

    @staticmethod
    def _state_from_tensors(
        position: torch.Tensor,
        velocity: torch.Tensor,
        mass,
        dt,
        softening: float,
        detach: bool,
    ) -> _HistoryState:
        if detach:
            pos = position.detach().clone()
            vel = velocity.detach().clone()
        else:
            pos = position
            vel = velocity

        mass_t = mass if torch.is_tensor(mass) else torch.as_tensor(mass, device=pos.device, dtype=pos.dtype)
        dt_t = dt if torch.is_tensor(dt) else torch.as_tensor(dt, device=pos.device, dtype=pos.dtype)
        mass_t = mass_t.to(device=pos.device, dtype=pos.dtype)
        dt_t = dt_t.to(device=pos.device, dtype=pos.dtype)

        mass_t = HistoryBuffer._normalize_mass(mass_t, pos)
        dt_t = HistoryBuffer._normalize_dt(dt_t, pos)

        if detach:
            mass_t = mass_t.detach().clone()
            dt_t = dt_t.detach().clone()

        return _HistoryState(
            position=pos,
            velocity=vel,
            mass=mass_t,
            dt=dt_t,
            softening=float(softening),
        )

    def _state_from_particle(self, p: "ParticleTorch", detach: bool) -> _HistoryState:
        return self._state_from_tensors(
            position=p.position,
            velocity=p.velocity,
            mass=p.mass,
            dt=p.dt,
            softening=getattr(p, "softening", 0.0),
            detach=detach,
        )

    def push(self, p: "ParticleTorch"):
        # store detached tensors so we don't keep old graphs alive
        self._buf.append(self._state_from_particle(p, detach=True))

    def clone(self) -> "HistoryBuffer":
        """Create a shallow copy with detached stored states."""
        hb = HistoryBuffer(history_len=self.history_len, feature_type=self.feature_type)
        for state in list(self._buf):
            hb._buf.append(
                _HistoryState(
                    position=state.position.detach().clone(),
                    velocity=state.velocity.detach().clone(),
                    mass=state.mass.detach().clone(),
                    dt=state.dt.detach().clone(),
                    softening=state.softening,
                )
            )
        return hb

    @staticmethod
    def _compute_acceleration(position: torch.Tensor, mass: torch.Tensor, softening: float, G: float = 1.0) -> torch.Tensor:
        m = HistoryBuffer._normalize_mass(mass, position)
        return nbody_compute_acceleration(position, m, softening=softening, G=G)

    @staticmethod
    def _basic_features(position: torch.Tensor, velocity: torch.Tensor, mass: torch.Tensor,
                        softening: float, which_accel: int = 0, G: float = 1.0) -> torch.Tensor:
        _ = which_accel
        return nbody_system_features(
            position=position,
            velocity=velocity,
            mass=mass,
            softening=softening,
            G=G,
            mode="basic",
        )

    @staticmethod
    def _rich_features(position: torch.Tensor, velocity: torch.Tensor, mass: torch.Tensor,
                       softening: float, which_accel: int = 0, G: float = 1.0) -> torch.Tensor:
        _ = which_accel
        return nbody_system_features(
            position=position,
            velocity=velocity,
            mass=mass,
            softening=softening,
            G=G,
            mode="rich",
        )

    @staticmethod
    def _features_from_sequence(
        position_seq: torch.Tensor,
        velocity_seq: torch.Tensor,
        mass_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        feature_type: FeatureType,
        softening: float,
    ) -> torch.Tensor:
        if feature_type in ("basic", "rich"):
            mode = "basic" if feature_type == "basic" else "rich"
            feats = nbody_system_features(
                position=position_seq,
                velocity=velocity_seq,
                mass=mass_seq,
                softening=softening,
                G=1.0,
                mode=mode,
            )
            return torch.cat(torch.unbind(feats, dim=0), dim=-1)
        if feature_type == "delta_mag":
            if position_seq.shape[0] < 2:
                raise ValueError("delta_mag requires history_len >= 1")
            acc_seq = HistoryBuffer._compute_acceleration(position_seq, mass_seq, softening)

            dpos = position_seq[1:] - position_seq[:-1]
            dvel = velocity_seq[1:] - velocity_seq[:-1]
            dacc = acc_seq[1:] - acc_seq[:-1]

            dpos_mag = torch.linalg.norm(dpos, dim=-1)
            dvel_mag = torch.linalg.norm(dvel, dim=-1)
            dacc_mag = torch.linalg.norm(dacc, dim=-1)

            def _stats(x):
                mean = x.mean(dim=-1, keepdim=True)
                maxv = x.max(dim=-1).values.unsqueeze(-1)
                rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-12)
                return mean, maxv, rms

            dpos_stats = _stats(dpos_mag)
            dvel_stats = _stats(dvel_mag)
            dacc_stats = _stats(dacc_mag)

            dt_feat = dt_seq[1:].unsqueeze(-1)
            delta_feats = torch.cat([*dpos_stats, *dvel_stats, *dacc_stats, dt_feat], dim=-1)
            return torch.cat(torch.unbind(delta_feats, dim=0), dim=-1)
        raise ValueError(f"Unsupported feature_type={feature_type!r}")

    @staticmethod
    def _expand_state_to_batch(state: _HistoryState, batch_size: int) -> _HistoryState:
        pos = state.position
        vel = state.velocity
        mass = state.mass
        dt = state.dt

        if pos.dim() == 3 and pos.shape[0] == batch_size:
            return state
        if pos.dim() == 3 and pos.shape[0] == 1:
            pos = pos.expand(batch_size, *pos.shape[1:])
            vel = vel.expand(batch_size, *vel.shape[1:])
            mass = mass.expand(batch_size, *mass.shape[1:])
            dt = dt.expand(batch_size)
            return _HistoryState(pos, vel, mass, dt, state.softening)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0).expand(batch_size, *pos.shape)
            vel = vel.unsqueeze(0).expand(batch_size, *vel.shape)
            mass = mass.unsqueeze(0).expand(batch_size, *mass.shape)
            dt = dt.expand(batch_size)
            return _HistoryState(pos, vel, mass, dt, state.softening)
        raise ValueError(f"Cannot expand history state of shape {pos.shape} to batch {batch_size}")

    def features_for(self, current: "ParticleTorch") -> torch.Tensor:
        """
        Build concatenated features over time: [past...past, current].
        Returns shape (F_total,) for single system, or (B, F_total) if
        `current` is batched and the history elements match shapes.
        """
        # gather up to history_len past states; if fewer, pad by repeating oldest
        past_list: List[_HistoryState] = list(self._buf)
        if len(past_list) < self.history_len:
            pad_count = self.history_len - len(past_list)
            if past_list:
                past_list = [past_list[0]] * pad_count + past_list
            else:
                past_list = [self._state_from_particle(current, detach=False)] * pad_count

        seq = past_list + [self._state_from_particle(current, detach=False)]
        softening = seq[0].softening
        for state in seq[1:]:
            if state.softening != softening:
                raise ValueError("All history states must share the same softening value.")

        pos_seq = torch.stack([s.position for s in seq], dim=0)
        vel_seq = torch.stack([s.velocity for s in seq], dim=0)
        mass_seq = torch.stack([s.mass for s in seq], dim=0)
        dt_seq = torch.stack([s.dt for s in seq], dim=0)

        feats_cat = self._features_from_sequence(
            pos_seq,
            vel_seq,
            mass_seq,
            dt_seq,
            self.feature_type,
            softening,
        )

        # squeeze batch if B==1
        if feats_cat.dim() > 1 and feats_cat.size(0) == 1:
            feats_cat = feats_cat.squeeze(0)
        return feats_cat

    def features_for_batch(self, batch_state: "ParticleTorch") -> torch.Tensor:
        """
        Build concatenated features for each sample in a batched `ParticleTorch`
        using the same stored history for all samples.
        """
        pos = batch_state.position
        vel = batch_state.velocity
        mass = batch_state.mass
        dt = batch_state.dt

        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
            vel = vel.unsqueeze(0)

        B = pos.shape[0]
        past_list: List[_HistoryState] = list(self._buf)
        if len(past_list) < self.history_len:
            pad_count = self.history_len - len(past_list)
            if past_list:
                past_list = [past_list[0]] * pad_count + past_list
            else:
                past_list = []

        current_state = self._state_from_tensors(
            position=pos,
            velocity=vel,
            mass=mass,
            dt=dt,
            softening=getattr(batch_state, "softening", 0.0),
            detach=False,
        )

        seq = [self._expand_state_to_batch(s, B) for s in past_list] + [current_state]
        if not seq:
            return torch.empty((B, 0), device=pos.device, dtype=pos.dtype)

        softening = seq[0].softening
        for state in seq[1:]:
            if state.softening != softening:
                raise ValueError("All history states must share the same softening value.")

        pos_seq = torch.stack([s.position for s in seq], dim=0)
        vel_seq = torch.stack([s.velocity for s in seq], dim=0)
        mass_seq = torch.stack([s.mass for s in seq], dim=0)
        dt_seq = torch.stack([s.dt for s in seq], dim=0)

        feats = self._features_from_sequence(
            pos_seq,
            vel_seq,
            mass_seq,
            dt_seq,
            self.feature_type,
            softening,
        )
        return feats

    @staticmethod
    def features_for_histories(
        histories: Sequence["HistoryBuffer"],
        batch_state: "ParticleTorch",
    ) -> torch.Tensor:
        if not histories:
            raise ValueError("histories must be a non-empty sequence")

        history_len = histories[0].history_len
        feature_type = histories[0].feature_type
        for h in histories[1:]:
            if h.history_len != history_len or h.feature_type != feature_type:
                raise ValueError("All histories must share history_len and feature_type")

        pos = batch_state.position
        vel = batch_state.velocity
        mass = batch_state.mass
        dt = batch_state.dt

        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
            vel = vel.unsqueeze(0)

        B = pos.shape[0]
        if len(histories) != B:
            raise ValueError("histories length must match batch size")

        steps = history_len + 1
        pos_steps: List[List[torch.Tensor]] = [[] for _ in range(steps)]
        vel_steps: List[List[torch.Tensor]] = [[] for _ in range(steps)]
        mass_steps: List[List[torch.Tensor]] = [[] for _ in range(steps)]
        dt_steps: List[List[torch.Tensor]] = [[] for _ in range(steps)]

        softening = None
        for i, history in enumerate(histories):
            past_list = list(history._buf)

            mass_i = mass[i] if torch.is_tensor(mass) and mass.dim() == 2 else mass
            if torch.is_tensor(dt) and pos.dim() == 3 and dt.dim() >= 1 and dt.shape[0] == B:
                dt_i = dt[i]
            else:
                dt_i = dt
            current_state = HistoryBuffer._state_from_tensors(
                position=pos[i],
                velocity=vel[i],
                mass=mass_i,
                dt=dt_i,
                softening=getattr(batch_state, "softening", 0.0),
                detach=False,
            )

            if len(past_list) < history_len:
                pad_count = history_len - len(past_list)
                if past_list:
                    past_list = [past_list[0]] * pad_count + past_list
                else:
                    past_list = [current_state] * pad_count

            seq = past_list + [current_state]
            if softening is None:
                softening = seq[0].softening
            for state in seq:
                if state.softening != softening:
                    raise ValueError("All history states must share the same softening value.")

            for t, state in enumerate(seq):
                pos_steps[t].append(state.position)
                vel_steps[t].append(state.velocity)
                mass_steps[t].append(state.mass)
                dt_steps[t].append(state.dt)

        pos_seq = torch.stack([torch.stack(step, dim=0) for step in pos_steps], dim=0)
        vel_seq = torch.stack([torch.stack(step, dim=0) for step in vel_steps], dim=0)
        mass_seq = torch.stack([torch.stack(step, dim=0) for step in mass_steps], dim=0)
        dt_seq = torch.stack([torch.stack(step, dim=0) for step in dt_steps], dim=0)

        feats = HistoryBuffer._features_from_sequence(
            pos_seq,
            vel_seq,
            mass_seq,
            dt_seq,
            feature_type,
            softening if softening is not None else 0.0,
        )
        return feats
