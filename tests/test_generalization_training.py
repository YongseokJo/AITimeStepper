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


class TestSampleMinibatch:
    """Tests for sample_minibatch function."""

    def test_returns_correct_size(self, sample_trajectory):
        """Should return requested batch_size samples."""
        result = sample_minibatch(sample_trajectory, batch_size=3)
        assert len(result) == 3

    def test_caps_at_trajectory_length(self, sample_trajectory):
        """Should cap batch_size at trajectory length."""
        result = sample_minibatch(sample_trajectory, batch_size=100)
        assert len(result) == len(sample_trajectory)

    def test_returns_valid_tuples(self, sample_trajectory):
        """Each sample should be (ParticleTorch, float) tuple."""
        result = sample_minibatch(sample_trajectory, batch_size=2)
        for particle, dt in result:
            assert isinstance(particle, ParticleTorch)
            assert isinstance(dt, float)

    def test_random_selection(self, sample_trajectory):
        """Multiple calls should return different samples (probabilistic)."""
        results = [sample_minibatch(sample_trajectory, batch_size=2) for _ in range(10)]
        # Check that we got at least 2 different selections
        unique = set()
        for r in results:
            unique.add(tuple(dt for _, dt in r))
        assert len(unique) > 1, "Should return different random samples"


class TestEvaluateMinibatch:
    """Tests for evaluate_minibatch function."""

    def test_returns_correct_structure(self, sample_trajectory, config, adapter):
        """Should return (all_pass, losses, rel_dE_list, metrics)."""
        model = MockModel(dt_value=1e-7)  # Very small dt for quick pass
        minibatch = sample_trajectory[:2]

        result = evaluate_minibatch(model, minibatch, config, adapter)

        assert len(result) == 4
        all_pass, losses, rel_dE_list, metrics = result
        assert isinstance(all_pass, bool)
        assert isinstance(losses, list)
        assert isinstance(rel_dE_list, list)
        assert isinstance(metrics, dict)

    def test_all_pass_with_small_dt(self, sample_trajectory, config, adapter):
        """Should return all_pass=True with very small dt."""
        model = MockModel(dt_value=1e-8)  # Very small dt
        minibatch = sample_trajectory[:2]

        all_pass, losses, rel_dE_list, metrics = evaluate_minibatch(
            model, minibatch, config, adapter
        )

        assert all_pass is True
        assert len(losses) == 0  # No failures

    def test_metrics_keys(self, sample_trajectory, config, adapter):
        """Metrics should contain expected keys."""
        model = MockModel(dt_value=1e-7)
        minibatch = sample_trajectory[:2]

        _, _, _, metrics = evaluate_minibatch(model, minibatch, config, adapter)

        assert 'pass_count' in metrics
        assert 'fail_count' in metrics
        assert 'mean_rel_dE' in metrics
        assert 'max_rel_dE' in metrics

    def test_particle_immutability(self, sample_trajectory, config, adapter):
        """Trajectory particles should NOT be modified during evaluation.

        This test verifies that evaluate_minibatch preserves immutability
        of the trajectory samples by comparing particle states before
        and after evaluation.
        """
        model = MockModel(dt_value=0.01)  # Larger dt to ensure integration happens
        minibatch = sample_trajectory[:3]

        # Record original positions before evaluation
        original_positions = [p.position.clone() for p, _ in minibatch]
        original_velocities = [p.velocity.clone() for p, _ in minibatch]

        # Evaluate minibatch (this should NOT modify original particles)
        _ = evaluate_minibatch(model, minibatch, config, adapter)

        # Verify particles are unchanged
        for i, (particle, _) in enumerate(minibatch):
            assert torch.allclose(particle.position, original_positions[i]), \
                f"Particle {i} position was modified during evaluation"
            assert torch.allclose(particle.velocity, original_velocities[i]), \
                f"Particle {i} velocity was modified during evaluation"
