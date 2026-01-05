from collections import deque
from typing import Deque, List, Literal, Optional

import torch


FeatureType = Literal["basic", "rich"]


class HistoryBuffer:
    """
    Stores the last K particle states (detached clones) and provides a
    time-concatenated feature vector for the current step.

    - feature_type="basic" uses `ParticleTorch._get_batch_()` → [..., 2]
    - feature_type="rich" uses `ParticleTorch.get_batch()` → [..., F]

    For single-system usage, returns a 1D feature vector of length
    (K+1) * F, where K is `history_len` and F is per-step feature size.

    If there are fewer than K past states available, repeats the oldest
    available state to fill the sequence (left-padding).
    """

    def __init__(self, history_len: int = 3, feature_type: FeatureType = "basic"):
        assert history_len >= 0
        self.history_len: int = int(history_len)
        self.feature_type: FeatureType = feature_type
        self._buf: Deque["ParticleTorch"] = deque(maxlen=self.history_len)

    def reset(self):
        self._buf.clear()

    def push(self, p: "ParticleTorch"):
        # store a detached clone so we don't keep old graphs alive
        self._buf.append(p.clone_detached())

    def clone(self) -> "HistoryBuffer":
        """Create a shallow copy with detached stored states."""
        hb = HistoryBuffer(history_len=self.history_len, feature_type=self.feature_type)
        for p in list(self._buf):
            hb._buf.append(p.clone_detached())
        return hb

    def _features_one(self, p: "ParticleTorch") -> torch.Tensor:
        if self.feature_type == "basic":
            f = p._get_batch_()  # (..., 2)
        else:
            f = p.get_batch()    # (..., F)
        if f.dim() == 0:
            f = f.unsqueeze(0)
        return f

    def features_for(self, current: "ParticleTorch") -> torch.Tensor:
        """
        Build concatenated features over time: [past...past, current].
        Returns shape (F_total,) for single system, or (B, F_total) if
        `current` is batched and the history elements match shapes.
        """
        # gather up to history_len past states; if fewer, pad by repeating oldest
        past_list: List["ParticleTorch"] = list(self._buf)
        if len(past_list) < self.history_len:
            pad_count = self.history_len - len(past_list)
            if past_list:
                past_list = [past_list[0]] * pad_count + past_list
            else:
                past_list = [current] * pad_count

        seq = past_list + [current]

        feats = [self._features_one(p) for p in seq]
        # ensure each is at least 2D: (B, F) or (1, F)
        feats = [f if f.dim() > 1 else f.unsqueeze(0) for f in feats]
        # concatenate along feature dimension
        feats_cat = torch.cat(feats, dim=-1)  # (B, (K+1)*F)
        # squeeze batch if B==1
        if feats_cat.size(0) == 1:
            feats_cat = feats_cat.squeeze(0)
        return feats_cat

    def features_for_batch(self, batch_state: "ParticleTorch") -> torch.Tensor:
        """
        Build concatenated features for each sample in a batched `ParticleTorch`.
        Returns a tensor of shape (B, F_total).
        """
        pos = batch_state.position
        vel = batch_state.velocity
        mass = batch_state.mass
        dt = batch_state.dt

        # normalize to batched shapes
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
            vel = vel.unsqueeze(0)
            mass = mass.unsqueeze(0) if torch.is_tensor(mass) and mass.dim() == 1 else mass
        B = pos.shape[0]

        feats_list = []
        for i in range(B):
            p_i = batch_state.__class__.from_tensors(
                mass=mass[i], position=pos[i], velocity=vel[i], dt=(dt[i] if torch.is_tensor(dt) and dt.dim()>0 else dt)
            )
            feats_list.append(self.features_for(p_i))

        # stack per-sample features to (B, F_total)
        feats_list = [f if f.dim()>1 else f.unsqueeze(0) for f in feats_list]
        return torch.cat(feats_list, dim=0)
