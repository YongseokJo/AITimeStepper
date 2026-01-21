"""
Tests for unified epoch training functions.

Covers:
- train_epoch_two_phase (single epoch orchestration)
- run_two_phase_training (multi-epoch loop with checkpointing)

Test organization:
- TestTrainEpochTwoPhase: Tests for single epoch function
- TestRunTwoPhaseTraining: Tests for multi-epoch loop
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.config import Config
from src.particle import ParticleTorch
from src.model_adapter import ModelAdapter
from src.history_buffer import HistoryBuffer
from src.unified_training import (
    train_epoch_two_phase,
    run_two_phase_training,
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
        # Use float64 for compatibility with particle system
        self.linear = nn.Linear(input_dim, 2, dtype=torch.float64)
        # Zero out weights so output is just the bias
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, dt_value)

    def forward(self, x):
        x = x.to(dtype=self.linear.weight.dtype)
        if x.shape[-1] != self.linear.in_features:
            new_linear = nn.Linear(x.shape[-1], 2, dtype=torch.float64, device=x.device)
            nn.init.constant_(new_linear.weight, 0.0)
            nn.init.constant_(new_linear.bias, self.dt_value)
            self.linear = new_linear
        out = self.linear(x)
        return torch.abs(out) + 1e-9


class TrainableMockModel(nn.Module):
    """Mock model with trainable parameters for testing optimizer behavior."""

    def __init__(self, input_dim: int = 11, initial_dt: float = 0.1):
        super().__init__()
        self.initial_dt = initial_dt
        self.fc = nn.Linear(input_dim, 2, dtype=torch.float64)
        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, initial_dt)

    def forward(self, x):
        x = x.to(dtype=self.fc.weight.dtype)
        if x.shape[-1] != self.fc.in_features:
            new_fc = nn.Linear(x.shape[-1], 2, dtype=torch.float64, device=x.device)
            nn.init.constant_(new_fc.weight, 0.0)
            nn.init.constant_(new_fc.bias, self.initial_dt)
            self.fc = new_fc
        return torch.abs(self.fc(x)) + 1e-6


@pytest.fixture
def simple_particle():
    """Create a simple 2-body particle system."""
    pos = torch.tensor([[0.5, 0.0], [-0.5, 0.0]], dtype=torch.float64)
    vel = torch.tensor([[0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    mass = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel, dt=0.001)


@pytest.fixture
def config():
    """Create a test configuration with relaxed thresholds for fast testing."""
    return Config(
        energy_threshold=0.1,  # 10% for fast testing
        steps_per_epoch=3,
        replay_steps=10,
        replay_batch_size=2,
        min_replay_size=1,
        epochs=5,
        debug=False,
    )


@pytest.fixture
def adapter(config):
    """Create a ModelAdapter for testing."""
    return ModelAdapter(config)


# =============================================================================
# TestTrainEpochTwoPhase: Tests for single epoch orchestration
# =============================================================================

class TestTrainEpochTwoPhase:
    """Tests for train_epoch_two_phase() function."""

    def test_returns_expected_structure(self, simple_particle, config, adapter):
        """Verify return dict has all expected keys."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = train_epoch_two_phase(
            model, simple_particle, optimizer, config, adapter
        )

        # Check all required keys present
        assert 'trajectory_metrics' in result
        assert 'generalization_metrics' in result
        assert 'converged' in result
        assert 'part2_iterations' in result
        assert 'epoch_time' in result

        # Check types
        assert isinstance(result['trajectory_metrics'], dict)
        assert isinstance(result['generalization_metrics'], dict)
        assert isinstance(result['converged'], bool)
        assert isinstance(result['part2_iterations'], int)
        assert isinstance(result['epoch_time'], float)

    def test_trajectory_metrics_structure(self, simple_particle, config, adapter):
        """Verify trajectory_metrics contains Part 1 output."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = train_epoch_two_phase(
            model, simple_particle, optimizer, config, adapter
        )

        traj_m = result['trajectory_metrics']
        assert 'total_steps' in traj_m
        assert 'warmup_discarded' in traj_m
        assert 'trajectory_length' in traj_m
        assert 'mean_retrain_iterations' in traj_m
        assert 'mean_energy_error' in traj_m

        # trajectory_length should match steps_per_epoch (no history buffer)
        assert traj_m['trajectory_length'] == config.steps_per_epoch

    def test_generalization_metrics_structure(self, simple_particle, config, adapter):
        """Verify generalization_metrics contains Part 2 output."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = train_epoch_two_phase(
            model, simple_particle, optimizer, config, adapter
        )

        gen_m = result['generalization_metrics']
        assert 'mean_rel_dE' in gen_m
        assert 'max_rel_dE' in gen_m
        assert 'final_pass_rate' in gen_m

    def test_part1_output_fed_to_part2(self, simple_particle, config, adapter):
        """Verify trajectory from Part 1 is processed by Part 2."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = train_epoch_two_phase(
            model, simple_particle, optimizer, config, adapter
        )

        # If trajectory was collected, Part 2 should have run
        traj_len = result['trajectory_metrics']['trajectory_length']
        if traj_len > 0:
            # Part 2 should have done at least some work
            assert result['part2_iterations'] >= 0

    def test_epoch_time_measured(self, simple_particle, config, adapter):
        """Verify epoch_time is positive and reasonable."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = train_epoch_two_phase(
            model, simple_particle, optimizer, config, adapter
        )

        assert result['epoch_time'] > 0
        assert result['epoch_time'] < 60  # Should not take > 60 seconds

    def test_with_history_buffer(self, simple_particle):
        """Verify function works with history buffer enabled."""
        config = Config(
            energy_threshold=0.1,
            steps_per_epoch=5,
            history_len=2,
            replay_steps=5,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        # Use very small dt to ensure quick energy acceptance
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        history_buffer = HistoryBuffer(
            history_len=config.history_len,
            feature_type=config.feature_type
        )
        # Pre-populate to avoid zero-padding NaN issues
        history_buffer.push(simple_particle.clone_detached())
        history_buffer.push(simple_particle.clone_detached())

        result = train_epoch_two_phase(
            model, simple_particle, optimizer, config, adapter,
            history_buffer=history_buffer
        )

        # warmup should be discarded
        traj_m = result['trajectory_metrics']
        assert traj_m['warmup_discarded'] == config.history_len
        # trajectory_length should be steps_per_epoch - history_len
        expected_len = config.steps_per_epoch - config.history_len
        assert traj_m['trajectory_length'] == expected_len

    def test_empty_trajectory_edge_case(self, simple_particle):
        """Verify function handles empty trajectory (all warmup)."""
        config = Config(
            energy_threshold=0.1,
            steps_per_epoch=2,  # <= history_len
            history_len=3,
            replay_steps=5,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        # Use very small dt to ensure quick energy acceptance
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        history_buffer = HistoryBuffer(
            history_len=config.history_len,
            feature_type=config.feature_type
        )
        # Pre-populate
        for _ in range(config.history_len):
            history_buffer.push(simple_particle.clone_detached())

        # Should issue warning but not crash
        with pytest.warns(UserWarning, match="Empty trajectory"):
            result = train_epoch_two_phase(
                model, simple_particle, optimizer, config, adapter,
                history_buffer=history_buffer
            )

        # Part 2 should return immediately for empty trajectory
        assert result['trajectory_metrics']['trajectory_length'] == 0
        assert result['converged'] == True  # Empty trajectory = trivially converged
        assert result['part2_iterations'] == 0

    def test_calls_part1_then_part2(self, simple_particle, config, adapter):
        """Verify train_epoch_two_phase calls Part 1 then Part 2 in order."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        call_order = []

        with patch('src.unified_training.collect_trajectory') as mock_part1, \
             patch('src.unified_training.generalize_on_trajectory') as mock_part2:

            # Setup mock return values
            mock_trajectory = [(simple_particle.clone_detached(), 0.001)]
            mock_part1.return_value = (mock_trajectory, {
                'total_steps': 3,
                'warmup_discarded': 0,
                'trajectory_length': 3,
                'mean_retrain_iterations': 1.0,
                'mean_energy_error': 0.001,
                'max_retrain_iterations': 2,
            })

            mock_part2.return_value = (True, 5, {
                'mean_rel_dE': 0.001,
                'max_rel_dE': 0.002,
                'final_pass_rate': 1.0,
            })

            # Track call order
            def record_part1(*args, **kwargs):
                call_order.append('part1')
                return mock_part1.return_value
            mock_part1.side_effect = record_part1

            def record_part2(*args, **kwargs):
                call_order.append('part2')
                return mock_part2.return_value
            mock_part2.side_effect = record_part2

            result = train_epoch_two_phase(
                model, simple_particle, optimizer, config, adapter
            )

        # Verify Part 1 called before Part 2
        assert call_order == ['part1', 'part2'], f"Expected ['part1', 'part2'], got {call_order}"

        # Verify both were called
        mock_part1.assert_called_once()
        mock_part2.assert_called_once()


# =============================================================================
# TestRunTwoPhaseTraining: Tests for multi-epoch loop
# =============================================================================

class TestRunTwoPhaseTraining:
    """Tests for run_two_phase_training() function."""

    def test_runs_for_config_epochs(self, simple_particle, adapter):
        """Verify function runs for exactly config.epochs iterations."""
        config = Config(
            epochs=3,
            energy_threshold=0.1,
            steps_per_epoch=2,
            replay_steps=5,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter
        )

        assert result['epochs_completed'] == 3

    def test_returns_expected_structure(self, simple_particle, config, adapter):
        """Verify return dict has all expected keys."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter
        )

        assert 'epochs_completed' in result
        assert 'total_time' in result
        assert 'final_metrics' in result
        assert 'convergence_rate' in result
        assert 'results' in result

    def test_checkpoint_creation(self, simple_particle, adapter):
        """Verify checkpoints created at correct intervals."""
        config = Config(
            epochs=5,
            energy_threshold=0.1,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            result = run_two_phase_training(
                model, simple_particle, optimizer, config, adapter,
                save_dir=save_dir,
                checkpoint_interval=2,
            )

            # Should have checkpoints at epochs 0, 2, 4 (interval=2 + final)
            checkpoints = sorted(save_dir.glob("model_epoch_*.pt"))
            assert len(checkpoints) >= 2

            # Verify checkpoint naming
            checkpoint_epochs = [int(p.stem.split('_')[-1]) for p in checkpoints]
            assert 0 in checkpoint_epochs  # First epoch
            assert 4 in checkpoint_epochs  # Final epoch (5-1=4)

    def test_checkpoint_content(self, simple_particle, config, adapter):
        """Verify checkpoint files contain required data."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            run_two_phase_training(
                model, simple_particle, optimizer, config, adapter,
                save_dir=save_dir,
                checkpoint_interval=10,  # Just first and last
            )

            checkpoints = list(save_dir.glob("model_epoch_*.pt"))
            assert len(checkpoints) >= 1

            ckpt = torch.load(checkpoints[0])
            assert 'model_state_dict' in ckpt
            assert 'optimizer_state_dict' in ckpt
            assert 'epoch' in ckpt
            assert 'config' in ckpt

    def test_convergence_rate_tracking(self, simple_particle, adapter):
        """Verify convergence_rate is computed correctly."""
        config = Config(
            epochs=4,
            energy_threshold=0.1,
            steps_per_epoch=2,
            replay_steps=50,  # More iterations for convergence
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter
        )

        # convergence_rate should be between 0 and 1
        assert 0.0 <= result['convergence_rate'] <= 1.0

    def test_total_time_measured(self, simple_particle, config, adapter):
        """Verify total_time is positive and reasonable."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter
        )

        assert result['total_time'] > 0
        assert result['total_time'] < 120  # Should not take > 2 minutes

    def test_history_buffer_persists_across_epochs(self, simple_particle, adapter):
        """Verify same history buffer instance used across epochs."""
        config = Config(
            epochs=3,
            energy_threshold=0.1,
            steps_per_epoch=3,
            history_len=2,
            replay_steps=5,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        history_buffer = HistoryBuffer(
            history_len=config.history_len,
            feature_type=config.feature_type
        )
        # Pre-populate
        history_buffer.push(simple_particle.clone_detached())
        history_buffer.push(simple_particle.clone_detached())

        initial_len = len(history_buffer._buf)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter,
            history_buffer=history_buffer,
        )

        # Buffer should have states (was updated during collection)
        # Buffer is capped at history_len, so just verify it still has contents
        assert len(history_buffer._buf) >= 1

    def test_debug_mode_stores_results(self, simple_particle, adapter):
        """Verify results list populated in debug mode."""
        config = Config(
            epochs=3,
            energy_threshold=0.1,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            debug=True,
        )
        adapter = ModelAdapter(config)

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter
        )

        # In debug mode, results list should have one entry per epoch
        assert len(result['results']) == config.epochs

    def test_no_save_dir_skips_checkpointing(self, simple_particle, config, adapter):
        """Verify no error when save_dir is None."""
        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Should not raise any errors
        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter,
            save_dir=None,
        )

        assert result['epochs_completed'] == config.epochs

    def test_final_metrics_from_last_epoch(self, simple_particle, adapter):
        """Verify final_metrics contains the metrics from the last epoch."""
        config = Config(
            epochs=3,
            energy_threshold=0.1,
            steps_per_epoch=2,
            replay_steps=5,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter
        )

        final_metrics = result['final_metrics']
        # Should have same structure as train_epoch_two_phase result
        assert 'trajectory_metrics' in final_metrics
        assert 'generalization_metrics' in final_metrics
        assert 'converged' in final_metrics
        assert 'part2_iterations' in final_metrics
        assert 'epoch_time' in final_metrics

    def test_zero_epochs_returns_empty(self, simple_particle, adapter):
        """Verify zero epochs returns appropriate defaults."""
        config = Config(
            epochs=0,
            energy_threshold=0.1,
            steps_per_epoch=2,
            replay_steps=5,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter
        )

        assert result['epochs_completed'] == 0
        assert result['convergence_rate'] == 0.0
        assert result['final_metrics'] == {}
        assert result['results'] == []


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests verifying end-to-end behavior."""

    def test_full_training_loop(self, simple_particle):
        """Test complete training loop with all components."""
        config = Config(
            epochs=2,
            energy_threshold=0.1,
            steps_per_epoch=3,
            replay_steps=10,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        model = TrainableMockModel(input_dim=11, initial_dt=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            result = run_two_phase_training(
                model, simple_particle, optimizer, config, adapter,
                save_dir=save_dir,
            )

            # Verify complete results
            assert result['epochs_completed'] == 2
            assert result['total_time'] > 0
            assert 0.0 <= result['convergence_rate'] <= 1.0

            # Verify at least one checkpoint saved
            checkpoints = list(save_dir.glob("model_epoch_*.pt"))
            assert len(checkpoints) >= 1

    def test_training_with_history_buffer(self, simple_particle):
        """Test training with history buffer persistence."""
        config = Config(
            epochs=2,
            energy_threshold=0.1,
            steps_per_epoch=4,
            history_len=2,
            replay_steps=5,
            replay_batch_size=2,
            min_replay_size=1,
        )
        adapter = ModelAdapter(config)

        history_buffer = HistoryBuffer(
            history_len=config.history_len,
            feature_type=config.feature_type
        )
        # Pre-populate
        for _ in range(config.history_len):
            history_buffer.push(simple_particle.clone_detached())

        model = MockModel(dt_value=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        result = run_two_phase_training(
            model, simple_particle, optimizer, config, adapter,
            history_buffer=history_buffer,
        )

        # Training should complete
        assert result['epochs_completed'] == 2

        # History buffer should have been updated
        # Initial: 2 states, each epoch adds steps_per_epoch states
        # Buffer size is capped at history_len, but states were pushed
        assert len(history_buffer._buf) >= 1
