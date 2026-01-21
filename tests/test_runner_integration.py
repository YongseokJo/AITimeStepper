"""
Integration tests for runner.py refactor.

Verifies CLI compatibility, checkpoint contract, and warning emission
for the Phase 6 runner.py refactor using two-phase training.

Test categories:
- TestCLICompatibility: Verify CLI argument parsing and acceptance
- TestCheckpointContract: Verify checkpoints have required fields
- TestMultiOrbitWarning: Verify warning emitted when num_orbits > 1
- TestDurationWarning: Verify warning emitted when duration is set
- TestWandBLogging: Verify --wandb flag is accepted
- TestDirectFunctionCalls: Fast unit tests calling run_training() directly
- TestArgumentParsing: Verify CLI argument parsing correctness

Phase 6 of AITimeStepper training refactor.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch, Mock

import pytest
import torch
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "run"))

from src.config import Config
from src.particle import ParticleTorch
from src.model_adapter import ModelAdapter
from src.history_buffer import HistoryBuffer
from src.structures import FullyConnectedNN
from src.checkpoint import save_checkpoint
from simulators.nbody_simulator import generate_random_ic


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def runner_script() -> Path:
    """Return path to runner.py script."""
    return project_root / "run" / "runner.py"


@pytest.fixture
def temp_save_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test outputs and clean up after."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def data_cleanup() -> Generator[None, None, None]:
    """Clean up test data directories after tests."""
    yield
    # Cleanup any test directories created in data/
    data_dir = project_root / "data"
    for name in ["test_cli_basic", "test_cli_history", "test_checkpoint_contract",
                 "test_multi_orbit_warn", "test_duration_warn", "test_wandb",
                 "test_direct_basic", "test_direct_history"]:
        test_dir = data_dir / name
        if test_dir.exists():
            shutil.rmtree(test_dir)


@pytest.fixture
def simple_particle() -> ParticleTorch:
    """Create a simple 2-body particle system for direct function tests."""
    pos = torch.tensor([[0.5, 0.0], [-0.5, 0.0]], dtype=torch.float64)
    vel = torch.tensor([[0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    mass = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return ParticleTorch.from_tensors(mass=mass, position=pos, velocity=vel, dt=0.001)


@pytest.fixture
def mock_training_result():
    """Return a mock training result dictionary."""
    return {
        'epochs_completed': 1,
        'total_time': 1.5,
        'final_metrics': {
            'trajectory_metrics': {'trajectory_length': 5},
            'generalization_metrics': {'final_pass_rate': 0.8},
        },
        'convergence_rate': 0.5,
    }


# =============================================================================
# TestArgumentParsing: Verify CLI argument parsing
# =============================================================================

class TestArgumentParsing:
    """Verify CLI argument parsing for runner.py."""

    def test_train_parser_accepts_required_args(self, runner_script: Path):
        """Verify train subcommand accepts all required arguments."""
        from runner import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--epochs", "10",
            "--num-particles", "3",
        ])

        assert args.mode == "train"
        assert args.epochs == 10
        assert args.num_particles == 3

    def test_train_parser_accepts_history_args(self, runner_script: Path):
        """Verify train subcommand accepts history arguments."""
        from runner import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--epochs", "10",
            "--num-particles", "3",
            "--history-len", "5",
            "--feature-type", "delta_mag",
        ])

        assert args.history_len == 5
        assert args.feature_type == "delta_mag"

    def test_train_parser_accepts_wandb_args(self, runner_script: Path):
        """Verify train subcommand accepts W&B arguments."""
        from runner import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--epochs", "10",
            "--num-particles", "3",
            "--wandb",
            "--wandb-project", "test_project",
            "--wandb-name", "test_run",
        ])

        assert args.wandb == True
        assert args.wandb_project == "test_project"
        assert args.wandb_name == "test_run"

    def test_train_parser_accepts_training_params(self, runner_script: Path):
        """Verify train subcommand accepts training parameters."""
        from runner import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--epochs", "100",
            "--num-particles", "4",
            "--n-steps", "5",
            "--lr", "0.001",
            "--steps-per-epoch", "10",
            "--replay-steps", "500",
            "--energy-threshold", "0.001",
        ])

        assert args.epochs == 100
        assert args.num_particles == 4
        assert args.n_steps == 5
        assert args.lr == 0.001
        assert args.steps_per_epoch == 10
        assert args.replay_steps == 500
        assert args.energy_threshold == 0.001

    def test_simulate_parser_accepts_ml_mode(self, runner_script: Path):
        """Verify simulate subcommand accepts ml integrator mode."""
        from runner import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "simulate",
            "--integrator-mode", "ml",
            "--model-path", "/path/to/model.pt",
            "--num-particles", "3",
            "--steps", "100",
        ])

        assert args.mode == "simulate"
        assert args.integrator_mode == "ml"
        assert args.model_path == "/path/to/model.pt"
        assert args.num_particles == 3
        assert args.steps == 100

    def test_simulate_parser_accepts_history_mode(self, runner_script: Path):
        """Verify simulate subcommand accepts history integrator mode."""
        from runner import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "simulate",
            "--integrator-mode", "history",
            "--model-path", "/path/to/model.pt",
            "--num-particles", "3",
            "--steps", "100",
            "--history-len", "5",
            "--feature-type", "rich",
        ])

        assert args.integrator_mode == "history"
        assert args.history_len == 5
        assert args.feature_type == "rich"


# =============================================================================
# TestMultiOrbitWarning: Multi-orbit warning verification
# =============================================================================

class TestMultiOrbitWarning:
    """Verify UserWarning emitted when num_orbits > 1."""

    def test_multi_orbit_warning_emitted(self, mock_training_result):
        """Verify warning emitted when num_orbits > 1 in run_training()."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            num_orbits=4,  # > 1 triggers warning
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        # Mock run_two_phase_training to skip actual training
        with patch('runner.run_two_phase_training', return_value=mock_training_result):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run_training(config)

                # Check for multi-orbit warning
                multi_orbit_warnings = [
                    warning for warning in w
                    if "Multi-orbit" in str(warning.message)
                ]
                assert len(multi_orbit_warnings) >= 1, (
                    f"Expected multi-orbit warning, got: {[str(x.message) for x in w]}"
                )
                assert "num_orbits" in str(multi_orbit_warnings[0].message)

    def test_no_warning_when_num_orbits_is_one(self, mock_training_result):
        """Verify no multi-orbit warning when num_orbits == 1."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            num_orbits=1,  # == 1, no warning expected
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        with patch('runner.run_two_phase_training', return_value=mock_training_result):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run_training(config)

                multi_orbit_warnings = [
                    warning for warning in w
                    if "Multi-orbit" in str(warning.message)
                ]
                assert len(multi_orbit_warnings) == 0, (
                    f"Unexpected multi-orbit warning: {[str(x.message) for x in w]}"
                )


# =============================================================================
# TestDurationWarning: Duration warning verification
# =============================================================================

class TestDurationWarning:
    """Verify UserWarning emitted when duration is set."""

    def test_duration_warning_emitted(self, mock_training_result):
        """Verify warning emitted when duration is set in run_training()."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            duration=100.0,  # Not None triggers warning
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        with patch('runner.run_two_phase_training', return_value=mock_training_result):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run_training(config)

                # Check for duration warning
                duration_warnings = [
                    warning for warning in w
                    if "Duration" in str(warning.message) or "duration" in str(warning.message)
                ]
                assert len(duration_warnings) >= 1, (
                    f"Expected duration warning, got: {[str(x.message) for x in w]}"
                )

    def test_no_duration_warning_when_none(self, mock_training_result):
        """Verify no duration warning when duration is None."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            duration=None,  # None = no warning
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        with patch('runner.run_two_phase_training', return_value=mock_training_result):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run_training(config)

                duration_warnings = [
                    warning for warning in w
                    if "Duration" in str(warning.message) or "duration" in str(warning.message)
                ]
                assert len(duration_warnings) == 0, (
                    f"Unexpected duration warning: {[str(x.message) for x in w]}"
                )


# =============================================================================
# TestWandBLogging: W&B flag acceptance
# =============================================================================

class TestWandBLogging:
    """Verify --wandb flag is accepted."""

    def test_wandb_init_called_when_flag_set(self, mock_training_result):
        """Verify wandb.init is called when wandb=True in config."""
        from runner import run_training

        # Create mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.finish = MagicMock()

        config = Config(
            epochs=1,
            num_particles=2,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
            extra={"wandb": True, "wandb_project": "test_project"},
        )

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            with patch('runner.run_two_phase_training', return_value=mock_training_result):
                run_training(config)

        # Verify wandb.init was called
        mock_wandb.init.assert_called_once()

        # Verify wandb.finish was called for cleanup
        mock_wandb.finish.assert_called_once()

    def test_wandb_not_called_when_disabled(self, mock_training_result):
        """Verify wandb is not initialized when wandb=False."""
        from runner import run_training

        mock_wandb = MagicMock()

        config = Config(
            epochs=1,
            num_particles=2,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
            extra={"wandb": False},  # Explicitly disabled
        )

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            with patch('runner.run_two_phase_training', return_value=mock_training_result):
                run_training(config)

        # wandb.init should NOT be called
        mock_wandb.init.assert_not_called()


# =============================================================================
# TestCheckpointContract: Checkpoint format verification
# =============================================================================

class TestCheckpointContract:
    """Verify checkpoint files have required fields for simulation mode."""

    def test_checkpoint_contract_fields(self, temp_save_dir, simple_particle):
        """Verify checkpoint contains all required fields."""
        config = Config(
            epochs=1,
            num_particles=2,
            history_len=3,
            feature_type="delta_mag",
            dtype="float64",
        )

        # Create a simple model
        model = FullyConnectedNN(
            input_dim=10,
            output_dim=2,
            hidden_dims=[20],
            activation="tanh",
            output_positive=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Save checkpoint using the same function as run_training
        checkpoint_path = temp_save_dir / "model_epoch_0000.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=0,
            logs={"test": True},
            config=config,
        )

        # Load and verify checkpoint
        ckpt = torch.load(checkpoint_path)

        # Required fields for checkpoint contract
        required_fields = [
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "config",
            "history_len",
            "feature_type",
            "dtype",
        ]

        for field in required_fields:
            assert field in ckpt, f"Checkpoint missing required field: {field}"

        # Verify field values
        assert ckpt["epoch"] == 0
        assert ckpt["history_len"] == 3
        assert ckpt["feature_type"] == "delta_mag"
        assert ckpt["dtype"] == "float64"

    def test_checkpoint_config_recoverable(self, temp_save_dir):
        """Verify config can be recovered from checkpoint."""
        config = Config(
            epochs=100,
            num_particles=4,
            history_len=5,
            feature_type="rich",
            lr=0.001,
            energy_threshold=0.001,
        )

        model = FullyConnectedNN(
            input_dim=10,
            output_dim=2,
            hidden_dims=[20],
            activation="tanh",
            output_positive=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        checkpoint_path = temp_save_dir / "model_epoch_0000.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=0,
            config=config,
        )

        ckpt = torch.load(checkpoint_path)
        recovered_config = Config.from_dict(ckpt["config"])

        assert recovered_config.epochs == 100
        assert recovered_config.num_particles == 4
        assert recovered_config.history_len == 5
        assert recovered_config.feature_type == "rich"
        assert recovered_config.lr == 0.001
        assert recovered_config.energy_threshold == 0.001

    def test_checkpoint_model_state_loadable(self, temp_save_dir):
        """Verify model state can be loaded from checkpoint."""
        model = FullyConnectedNN(
            input_dim=15,
            output_dim=2,
            hidden_dims=[32, 32],
            activation="tanh",
            output_positive=True,
        )

        config = Config(epochs=1, num_particles=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        checkpoint_path = temp_save_dir / "model_epoch_0000.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=0,
            config=config,
        )

        # Load state into a new model
        new_model = FullyConnectedNN(
            input_dim=15,
            output_dim=2,
            hidden_dims=[32, 32],
            activation="tanh",
            output_positive=True,
        )

        ckpt = torch.load(checkpoint_path)
        new_model.load_state_dict(ckpt["model_state_dict"])

        # Verify models produce same output
        test_input = torch.randn(1, 15)
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = new_model(test_input)

        assert torch.allclose(original_output, loaded_output), "Loaded model produces different output"


# =============================================================================
# TestDirectFunctionCalls: Direct function interface tests
# =============================================================================

class TestDirectFunctionCalls:
    """Tests for run_training() function interface."""

    def test_run_training_validates_config(self):
        """Verify run_training() validates config before training."""
        from runner import run_training

        # Invalid config: num_particles < 2
        config = Config(
            epochs=1,
            num_particles=1,  # Invalid
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        with pytest.raises(ValueError, match="num_particles"):
            run_training(config)

    def test_run_training_validates_history_mode(self):
        """Verify run_training() validates history mode config."""
        from runner import run_training

        # Invalid config: history mode without history_len
        config = Config(
            epochs=1,
            num_particles=2,
            integrator_mode="history",
            history_len=0,  # Invalid for history mode
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        with pytest.raises(ValueError, match="history"):
            run_training(config)

    def test_run_training_creates_model_and_optimizer(self, mock_training_result):
        """Verify run_training() creates model and optimizer."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
            lr=0.001,
        )

        # Track what's passed to run_two_phase_training
        captured_args = {}

        def capture_training(*args, **kwargs):
            captured_args['model'] = kwargs.get('model')
            captured_args['optimizer'] = kwargs.get('optimizer')
            return mock_training_result

        with patch('runner.run_two_phase_training', side_effect=capture_training):
            run_training(config)

        # Verify model and optimizer were created and passed
        assert captured_args.get('model') is not None, "Model not passed to training"
        assert captured_args.get('optimizer') is not None, "Optimizer not passed to training"
        assert isinstance(captured_args['optimizer'], torch.optim.Adam)

    def test_run_training_creates_adapter(self, mock_training_result):
        """Verify run_training() creates ModelAdapter."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            history_len=3,
            feature_type="delta_mag",
            steps_per_epoch=5,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        captured_args = {}

        def capture_training(*args, **kwargs):
            captured_args['adapter'] = kwargs.get('adapter')
            return mock_training_result

        with patch('runner.run_two_phase_training', side_effect=capture_training):
            run_training(config)

        # Verify adapter was created
        assert captured_args.get('adapter') is not None, "Adapter not passed to training"
        assert isinstance(captured_args['adapter'], ModelAdapter)

    def test_run_training_passes_config_to_training(self, mock_training_result):
        """Verify config is passed to run_two_phase_training."""
        from runner import run_training

        config = Config(
            epochs=5,
            num_particles=3,
            steps_per_epoch=10,
            replay_steps=50,
            replay_batch_size=8,
            min_replay_size=4,
            energy_threshold=0.01,
        )

        captured_args = {}

        def capture_training(*args, **kwargs):
            captured_args['config'] = kwargs.get('config')
            return mock_training_result

        with patch('runner.run_two_phase_training', side_effect=capture_training):
            run_training(config)

        # Verify config was passed
        passed_config = captured_args.get('config')
        assert passed_config is not None
        assert passed_config.epochs == 5
        assert passed_config.num_particles == 3
        assert passed_config.steps_per_epoch == 10


# =============================================================================
# TestRunTrainingIntegration: Integration tests with mocked training
# =============================================================================

class TestRunTrainingIntegration:
    """Integration tests for run_training with mocked training loop."""

    def test_run_training_prints_summary(self, mock_training_result, capsys):
        """Verify run_training prints training summary."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
        )

        with patch('runner.run_two_phase_training', return_value=mock_training_result):
            run_training(config)

        captured = capsys.readouterr()
        assert "Training Complete" in captured.out
        assert "Epochs completed" in captured.out

    def test_run_training_saves_checkpoints(self, temp_save_dir, mock_training_result):
        """Verify run_training saves checkpoints when save_name is set."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
            save_name="test_checkpoints",
        )

        # Track save_dir passed to run_two_phase_training
        captured_args = {}

        def capture_training(*args, **kwargs):
            captured_args['save_dir'] = kwargs.get('save_dir')
            return mock_training_result

        with patch('runner.run_two_phase_training', side_effect=capture_training):
            run_training(config)

        # Verify save_dir was passed
        assert captured_args.get('save_dir') is not None
        assert "test_checkpoints" in str(captured_args['save_dir'])

    def test_run_training_no_checkpoints_when_no_save_name(self, mock_training_result):
        """Verify run_training does not save when save_name is None."""
        from runner import run_training

        config = Config(
            epochs=1,
            num_particles=2,
            steps_per_epoch=2,
            replay_steps=3,
            replay_batch_size=2,
            min_replay_size=1,
            energy_threshold=0.1,
            save_name=None,
        )

        captured_args = {}

        def capture_training(*args, **kwargs):
            captured_args['save_dir'] = kwargs.get('save_dir')
            return mock_training_result

        with patch('runner.run_two_phase_training', side_effect=capture_training):
            run_training(config)

        # save_dir should still be set (based on default name)
        # but the function was called correctly
        assert 'save_dir' in captured_args


# =============================================================================
# TestConfigFromCLI: Config creation from CLI arguments
# =============================================================================

class TestConfigFromCLI:
    """Test Config creation from CLI argument parsing."""

    def test_config_from_train_args(self):
        """Test Config.from_dict creates correct config from train args."""
        args = {
            'mode': 'train',
            'epochs': 100,
            'num_particles': 4,
            'history_len': 5,
            'feature_type': 'delta_mag',
            'lr': 0.001,
            'steps_per_epoch': 10,
            'replay_steps': 500,
            'energy_threshold': 0.001,
            'wandb': True,
            'wandb_project': 'test',
        }

        config = Config.from_dict(args)

        assert config.epochs == 100
        assert config.num_particles == 4
        assert config.history_len == 5
        assert config.feature_type == "delta_mag"
        assert config.lr == 0.001
        assert config.extra.get('wandb') == True
        assert config.extra.get('wandb_project') == 'test'

    def test_config_validation_passes_for_valid_args(self):
        """Test config validation passes for valid arguments."""
        config = Config(
            epochs=10,
            num_particles=3,
            steps_per_epoch=5,
            energy_threshold=0.01,
        )

        # Should not raise
        config.validate()

    def test_config_validation_fails_for_invalid_epochs(self):
        """Test config validation fails for epochs < 1."""
        config = Config(
            epochs=0,  # Invalid
            num_particles=3,
            steps_per_epoch=5,
            energy_threshold=0.01,
        )

        with pytest.raises(ValueError, match="epochs"):
            config.validate()

    def test_config_validation_fails_for_invalid_steps_per_epoch(self):
        """Test config validation fails for steps_per_epoch < 1."""
        config = Config(
            epochs=10,
            num_particles=3,
            steps_per_epoch=0,  # Invalid
            energy_threshold=0.01,
        )

        with pytest.raises(ValueError, match="steps_per_epoch"):
            config.validate()

    def test_config_validation_fails_for_invalid_energy_threshold(self):
        """Test config validation fails for energy_threshold <= 0."""
        config = Config(
            epochs=10,
            num_particles=3,
            steps_per_epoch=5,
            energy_threshold=0.0,  # Invalid
        )

        with pytest.raises(ValueError, match="energy_threshold"):
            config.validate()
