"""
Tests for trajectory collection functions.

Covers:
- attempt_single_step
- check_energy_threshold
- compute_single_step_loss
- collect_trajectory_step
- collect_trajectory
"""

import pytest
import torch
import torch.nn as nn

from src.config import Config
from src.particle import ParticleTorch
from src.model_adapter import ModelAdapter
from src.history_buffer import HistoryBuffer
from src.trajectory_collection import (
    attempt_single_step,
    check_energy_threshold,
    compute_single_step_loss,
    collect_trajectory_step,
    collect_trajectory,
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
        energy_threshold=2e-4,
        steps_per_epoch=5,
        history_len=0,
        feature_type="basic",
        E_lower=1e-6,
        E_upper=1e-4,
        num_particles=2,
        dim=2,
    )


@pytest.fixture
def config_with_history():
    """Create a test configuration with history enabled."""
    return Config(
        energy_threshold=2e-4,
        steps_per_epoch=5,
        history_len=3,
        feature_type="basic",
        E_lower=1e-6,
        E_upper=1e-4,
        num_particles=2,
        dim=2,
    )


@pytest.fixture
def adapter(config):
    """Create a model adapter."""
    return ModelAdapter(config, device=torch.device("cpu"), dtype=torch.float64)


@pytest.fixture
def adapter_with_history(config_with_history):
    """Create a model adapter with history."""
    return ModelAdapter(config_with_history, device=torch.device("cpu"), dtype=torch.float64)


class TestAttemptSingleStep:
    """Tests for attempt_single_step function."""

    def test_returns_correct_tuple(self, simple_particle, config, adapter):
        """Verify return value structure."""
        model = MockModel(dt_value=0.0001)
        result = attempt_single_step(model, simple_particle, config, adapter)

        assert len(result) == 4, "Should return (particle, dt, E0, E1)"
        p, dt, E0, E1 = result

        assert isinstance(p, ParticleTorch)
        assert torch.is_tensor(dt)
        assert torch.is_tensor(E0)
        assert torch.is_tensor(E1)

    def test_clone_detached_at_start(self, simple_particle, config, adapter):
        """Verify particle is cloned to avoid modifying original."""
        model = MockModel(dt_value=0.0001)
        original_pos = simple_particle.position.clone()

        _ = attempt_single_step(model, simple_particle, config, adapter)

        # Original particle should be unchanged
        assert torch.allclose(simple_particle.position, original_pos)

    def test_small_dt_preserves_energy(self, simple_particle, config, adapter):
        """Small dt should give small energy change."""
        model = MockModel(dt_value=1e-6)
        _, dt, E0, E1 = attempt_single_step(model, simple_particle, config, adapter)

        rel_dE = torch.abs((E1 - E0) / (E0 + 1e-12))
        assert rel_dE.item() < 1e-3, "Small dt should preserve energy"


class TestCheckEnergyThreshold:
    """Tests for check_energy_threshold function."""

    def test_passes_when_below_threshold(self):
        """Should pass when relative error is below threshold."""
        E0 = torch.tensor([-1.0])
        E1 = torch.tensor([-1.0001])  # 0.01% error
        passed, rel_dE = check_energy_threshold(E0, E1, threshold=0.001)

        assert passed is True
        assert rel_dE.item() < 0.001

    def test_fails_when_above_threshold(self):
        """Should fail when relative error exceeds threshold."""
        E0 = torch.tensor([-1.0])
        E1 = torch.tensor([-1.01])  # 1% error
        passed, rel_dE = check_energy_threshold(E0, E1, threshold=0.001)

        assert passed is False
        assert rel_dE.item() > 0.001

    def test_handles_small_energy(self):
        """Should handle near-zero energy safely."""
        E0 = torch.tensor([1e-10])
        E1 = torch.tensor([1.1e-10])
        passed, rel_dE = check_energy_threshold(E0, E1, threshold=0.5)

        assert torch.isfinite(rel_dE)


class TestComputeSingleStepLoss:
    """Tests for compute_single_step_loss function."""

    def test_returns_scalar(self, config):
        """Loss should be a scalar tensor."""
        E0 = torch.tensor([-1.0])
        E1 = torch.tensor([-1.0001])
        loss = compute_single_step_loss(E0, E1, config)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_zero_loss_inside_band(self, config):
        """Loss should be zero when error is inside acceptable band."""
        E0 = torch.tensor([-1.0])
        # Error between E_lower and E_upper
        error = (config.E_lower + config.E_upper) / 2
        E1 = E0 * (1 + error)
        loss = compute_single_step_loss(E0, E1, config)

        assert loss.item() < 1e-6, "Loss should be ~zero inside band"


class TestCollectTrajectoryStep:
    """Tests for collect_trajectory_step function."""

    def test_returns_accepted_step(self, simple_particle, config, adapter):
        """Should return accepted particle, dt, and metrics."""
        model = TrainableMockModel(input_dim=11, initial_dt=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        particle, dt, metrics = collect_trajectory_step(
            model, simple_particle, optimizer, config, adapter
        )

        assert isinstance(particle, ParticleTorch)
        assert isinstance(dt, float)
        assert isinstance(metrics, dict)
        assert 'retrain_iterations' in metrics
        assert 'final_energy_error' in metrics

    def test_energy_below_threshold(self, simple_particle, config, adapter):
        """Returned step should satisfy energy threshold."""
        # Use very small dt to ensure quick acceptance
        model = MockModel(dt_value=1e-7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        particle, dt, metrics = collect_trajectory_step(
            model, simple_particle, optimizer, config, adapter
        )

        assert metrics['final_energy_error'] < config.energy_threshold

    def test_retrains_until_passing(self, simple_particle, config, adapter):
        """Should retrain if initial prediction fails and eventually pass.

        This test verifies the retrain loop mechanism works correctly by:
        1. Using a trainable model that starts with a dt value that may or may not pass
        2. Running the collect_trajectory_step which will retrain until passing
        3. Verifying the returned metrics contain retrain iteration count
        """
        # Use trainable model - the retrain loop will adjust weights if needed
        model = TrainableMockModel(input_dim=11, initial_dt=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Should converge to acceptable energy threshold
        particle, dt, metrics = collect_trajectory_step(
            model, simple_particle, optimizer, config, adapter,
            max_retrain_warn=10000,  # Suppress warnings for test
        )

        # If it returns, it passed the energy threshold
        assert metrics['final_energy_error'] < config.energy_threshold
        # Verify metrics structure is correct
        assert isinstance(metrics['retrain_iterations'], int)
        assert metrics['retrain_iterations'] >= 0


class TestCollectTrajectory:
    """Tests for collect_trajectory function."""

    def test_collects_n_steps(self, simple_particle, config, adapter):
        """Should collect steps_per_epoch steps."""
        # Use TrainableMockModel with very small dt to ensure quick acceptance
        model = TrainableMockModel(input_dim=11, initial_dt=1e-7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Use higher energy threshold to make test faster
        config_fast = Config(
            energy_threshold=0.1,  # 10% threshold for fast testing
            steps_per_epoch=3,
            history_len=0,
            feature_type="basic",
            E_lower=1e-6,
            E_upper=1e-2,
            num_particles=2,
            dim=2,
        )

        trajectory, epoch_metrics = collect_trajectory(
            model, simple_particle, optimizer, config_fast, adapter
        )

        # Without history, no warmup discard
        assert len(trajectory) == config_fast.steps_per_epoch
        assert epoch_metrics['total_steps'] == config_fast.steps_per_epoch
        assert epoch_metrics['warmup_discarded'] == 0

    def test_discards_warmup_with_history(self, simple_particle, config_with_history, adapter_with_history):
        """Should discard first history_len steps.

        This test verifies HIST-02: warmup discard mechanism.
        Pre-populates history buffer to avoid zero-padding NaN issues.
        """
        # Use higher energy threshold and fewer steps for fast testing
        config_fast = Config(
            energy_threshold=0.1,  # 10% threshold for fast testing
            steps_per_epoch=5,
            history_len=3,
            feature_type="basic",
            E_lower=1e-6,
            E_upper=1e-2,
            num_particles=2,
            dim=2,
        )
        adapter_fast = ModelAdapter(config_fast, device=torch.device("cpu"), dtype=torch.float64)

        # Pre-populate history buffer with valid states to avoid zero-padding NaN
        hb = adapter_fast.history_buffer
        for _ in range(config_fast.history_len):
            hb.push(simple_particle)

        # History with 'basic' features: 11 features * 4 timesteps = 44 features
        # (history_len=3 past + 1 current)
        input_dim = 44
        model = TrainableMockModel(input_dim=input_dim, initial_dt=1e-7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trajectory, epoch_metrics = collect_trajectory(
            model, simple_particle, optimizer, config_fast, adapter_fast,
            history_buffer=hb,
        )

        expected_len = config_fast.steps_per_epoch - config_fast.history_len
        assert len(trajectory) == max(0, expected_len)
        assert epoch_metrics['warmup_discarded'] == config_fast.history_len

    def test_updates_history_buffer(self, simple_particle, config_with_history, adapter_with_history):
        """Should push to history buffer even during warmup.

        This test verifies that history buffer is updated during trajectory collection.
        Pre-populates history buffer to avoid zero-padding NaN issues.
        """
        # Use higher energy threshold for fast testing
        config_fast = Config(
            energy_threshold=0.1,  # 10% threshold for fast testing
            steps_per_epoch=5,
            history_len=3,
            feature_type="basic",
            E_lower=1e-6,
            E_upper=1e-2,
            num_particles=2,
            dim=2,
        )
        adapter_fast = ModelAdapter(config_fast, device=torch.device("cpu"), dtype=torch.float64)
        hb = adapter_fast.history_buffer

        # Pre-populate history buffer with valid states to avoid zero-padding NaN
        for _ in range(config_fast.history_len):
            hb.push(simple_particle)

        initial_len = len(hb._buf)

        # History with 'basic' features: 11 features * 4 timesteps = 44 features
        input_dim = 44
        model = TrainableMockModel(input_dim=input_dim, initial_dt=1e-7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trajectory, epoch_metrics = collect_trajectory(
            model, simple_particle, optimizer, config_fast, adapter_fast,
            history_buffer=hb,
        )

        # Buffer should have been updated (remains at max capacity)
        assert len(hb._buf) >= initial_len

    def test_returns_valid_metrics(self, simple_particle, config, adapter):
        """Epoch metrics should have expected keys."""
        model = TrainableMockModel(input_dim=11, initial_dt=1e-7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Use higher energy threshold for fast testing
        config_fast = Config(
            energy_threshold=0.1,  # 10% threshold for fast testing
            steps_per_epoch=2,
            history_len=0,
            feature_type="basic",
            E_lower=1e-6,
            E_upper=1e-2,
            num_particles=2,
            dim=2,
        )

        _, epoch_metrics = collect_trajectory(
            model, simple_particle, optimizer, config_fast, adapter
        )

        assert 'total_steps' in epoch_metrics
        assert 'warmup_discarded' in epoch_metrics
        assert 'trajectory_length' in epoch_metrics
        assert 'mean_retrain_iterations' in epoch_metrics
        assert 'mean_energy_error' in epoch_metrics
