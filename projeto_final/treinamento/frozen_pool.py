"""Frozen copies of a network (deterministic: mean action), used as opponents / teammates."""

import torch

from ppo import actor_state


class FrozenPool:
    def __init__(self, make_model, max_size=16):
        self.make_model, self.max_size = make_model, max_size
        self.models, self.names = [], []

    def __len__(self):
        return len(self.models)

    def _frozen(self, state_dict):
        m = self.make_model().cuda().eval()
        m.load_state_dict(actor_state(state_dict), strict=False)
        for q in m.parameters():
            q.requires_grad_(False)
        return m

    def add(self, state_dict, name):
        self.models.append(self._frozen(state_dict))
        self.names.append(name)

    def snapshot(self, state_dict, name):
        """Insert before the last ("latest") slot; when full drop the oldest but keep slot 0."""
        if len(self.models) >= self.max_size:
            del self.models[1], self.names[1]
        self.models.insert(len(self.models) - 1, self._frozen(state_dict))
        self.names.insert(len(self.names) - 1, name)

    def set(self, k, state_dict, name=None):
        self.models[k].load_state_dict(actor_state(state_dict), strict=False)
        if name:
            self.names[k] = name

    @torch.no_grad()
    def act(self, k, obs):
        m = self.models[k]
        return m.actor(m.norm(obs)).clamp(-1, 1)
