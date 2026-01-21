"""
Tests for generalization training functions.

Covers:
- sample_minibatch
- evaluate_minibatch
- generalize_on_trajectory
"""

import pytest
import torch
import torch.nn as nn

from src.config import Config
from src.particle import ParticleTorch
from src.model_adapter import ModelAdapter
from src.generalization_training import (
    generalize_on_trajectory,
    sample_minibatch,
    evaluate_minibatch,
)


class MockModel(nn.Module):
    """Mock model that returns configurable dt values.

    Uses a linear layer with zero weights so output is always dt_value,
    but gradients still flow through the computation graph.
    Dynamically adjusts input dimension based on first forward pass.
    """

    def __init__(self, dt_value: float = 0.001, input_dim: int = 11):
        super().__init__()
        self.dt_value = dt_value
        self.input_dim = input_dim
        self._initialized = False
        # Use float64 for compatibility with particle system
        self.linear = nn.Linear(input_dim, 2, dtype=torch.float64)
        # Zero out weights so output is just the bias
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, dt_value)

    def forward(self, x):
        # Ensure input dtype matches model
        x = x.to(dtype=self.linear.weight.dtype)
        # Dynamically resize linear layer if input dimension changed
        if x.shape[-1] != self.linear.in_features:
            new_linear = nn.Linear(x.shape[-1], 2, dtype=torch.float64, device=x.device)
            nn.init.constant_(new_linear.weight, 0.0)
            nn.init.constant_(new_linear.bias, self.dt_value)
            self.linear = new_linear
        # Output goes through linear layer to maintain gradients
        out = self.linear(x)
        # Ensure positive values
        return torch.abs(out) + 1e-9


class TrainableMockModel(nn.Module):
    """Mock model with trainable parameters for testing optimizer.

    Dynamically adjusts input dimension based on first forward pass.
    This model's weights can be trained to produce smaller dt values,
    which helps tests verify that training actually improves model behavior.
    """

    def __init__(self, input_dim: int = 11, initial_dt: float = 0.1):
        super().__init__()
        self.initial_dt = initial_dt
        # Use float64 for compatibility with particle system
        self.fc = nn.Linear(input_dim, 2, dtype=torch.float64)
        # Initialize to output approximately initial_dt
        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, initial_dt)

    def forward(self, x):
        # Ensure input dtype matches model
        x = x.to(dtype=self.fc.weight.dtype)
        # Dynamically resize linear layer if input dimension changed
        if x.shape[-1] != self.fc.in_features:
            new_fc = nn.Linear(x.shape[-1], 2, dtype=torch.float64, device=x.device)
            nn.init.constant_(new_fc.weight, 0.0)
            nn.init.constant_(new_fc.bias, self.initial_dt)
            self.fc = new_fc
        return torch.abs(self.fc(x)) + 1e-6  # Ensure positive


@pytest.fixture
def simple_particle():
    """Create a simple 2-body particle system."""
    pos = torch.tensor([[0.5, 0.0], [-0.5, 0.0]], dtype=torch.float64)
    vel = torch.tensor([[0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    mass = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel, dt=0.001)


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        energy_threshold=0.1,  # 10% for fast testing
        replay_batch_size=4,
        replay_steps=100,
        min_replay_size=2,
        E_lower=1e-6,
        E_upper=1e-2,
        num_particles=2,
        dim=2,
    )


@pytest.fixture
def adapter(config):
    """Create a model adapter."""
    return ModelAdapter(config, device=torch.device("cpu"), dtype=torch.float64)


@pytest.fixture
def sample_trajectory(simple_particle):
    """Create a sample trajectory for testing."""
    trajectory = []
    for i in range(5):
        # Create slightly perturbed particles
        p = simple_particle.clone_detached()
        trajectory.append((p, 0.001 * (i + 1)))
    return trajectory
