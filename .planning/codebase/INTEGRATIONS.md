# External Integrations - AITimeStepper

## Overview
AITimeStepper integrates with cloud-based ML experiment tracking and HPC cluster infrastructure. Most external integrations are optional or conditionally loaded.

---

## External APIs & Services

### Weights & Biases (wandb)
**Status**: Optional but active
**Purpose**: ML experiment tracking and visualization
**Version**: 0.23.1

#### Integration Points
1. **Training Logging** (`run/runner.py`, lines 240-395)
   - Conditionally imported only if `--wandb` flag enabled
   - Runtime check raises RuntimeError if wandb not installed and flag is set

2. **Configuration**
   ```python
   wandb_project = config.extra.get("wandb_project") or "AITimeStepper"
   wandb_name = config.extra.get("wandb_name") or config.save_name
   wandb.init(project=wandb_project, name=wandb_name, config=config.as_wandb_dict())
   ```

3. **Logged Metrics** (per epoch)
   - `epoch`: Training epoch number
   - `loss`: Total loss value
   - `dt`: Predicted time step
   - `E0`: Initial energy
   - `rel_dE`: Relative energy drift
   - `rel_dE_mean`: Mean relative energy drift
   - `rel_dL_mean`: Mean relative angular momentum drift

4. **Metadata Captured**
   - Full config snapshot
   - Python executable path
   - CUDA version (from system)
   - GPU model (e.g., NVIDIA A100-SXM4-40GB)
   - Git commit hash
   - Git remote URL
   - CPU count and logical cores
   - Email address (from job submission)

5. **Lifecycle**
   - `wandb.init()` - Start run at training start
   - `wandb.log()` - Log metrics each epoch
   - `wandb.finish()` - Cleanup at training end

#### Environment Variables (Implicit)
- W&B uses implicit authentication via `~/.netrc` or WANDB_API_KEY
- Project endpoint: `https://api.wandb.ai` (default)

#### Artifacts
- W&B stores metadata in `/run/wandb/` subdirectories
- Local copies of config, requirements, metadata saved per run
- Format: `/run/wandb/run-TIMESTAMP-ID/files/`

---

## Compute Infrastructure

### HPC Cluster Integration (SLURM)
**Status**: Active
**Scheduler**: SLURM Workload Manager
**Cluster**: Delta (XSEDE/ACCESS)

#### Job Submission Scripts
1. **GPU Training** (`run/run_ml.slurm`)
   - GPU: 1x GPU per node (A40 or A100)
   - Memory: 16GB
   - CPUs: 4 per task
   - Partition: Configurable (gpuA40x4, gpuA100x4, etc.)
   - Account: bgak-delta-gpu
   - Walltime: 12:00:00
   - Notifications: Email begin/end

2. **CPU Simulation** (`run/run_sim.slurm`)
   - No GPU
   - CPU-only workload
   - Standard node

3. **GPU Simulation** (`run/run_sim_gpu.slurm`)
   - GPU-accelerated simulation
   - Single GPU node

#### SLURM Configuration
```bash
#SBATCH --gpus-per-node=1
#SBATCH --account=bgak-delta-gpu
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --mail-type="BEGIN,END"
#SBATCH --constraint="scratch"
```

#### Modules Loaded
- python (system Python)
- cuda/11.8.0 (GPU compute toolkit)
- ffmpeg (for video encoding, optional)

#### Virtual Environment
- Activation: `source $HOME/pyenv/torch/bin/activate`
- PyTorch environment with CUDA 11.8 support pre-configured

---

## Data Sources & I/O

### File-Based Data
**Type**: Filesystem-based
**Format**: NumPy arrays (.npy) or text files (.txt)

#### Training Data Source
- Symlink: `data/` → `/work/hdd/bfpt/gkerex/AITimeStepper`
- Initial conditions (ICs) can be loaded via `--ic-path` flag
- Supports both `.npy` and `.txt` formats (auto-detected)

#### Output Destinations
```
data/<save_name>/
  ├── model/
  │   ├── model_epoch_0000.pt
  │   ├── model_epoch_0010.pt
  │   └── model_epoch_XXXX.pt
  └── logs/
```

#### Model Checkpoints
- Format: PyTorch state dict (.pt)
- Includes:
  - Model weights
  - Optimizer state
  - Epoch number
  - Loss value
  - Config snapshot
  - Training logs

---

## Remote Code Repository

### GitHub Integration
**Status**: Active
**Repository**: git@github.com:YongseokJo/AITimeStepper.git

#### Integration Points
1. **Git Metadata Capture**
   - Captured by W&B metadata collection
   - Commit hash logged in run metadata
   - Remote URL stored in W&B config

2. **Repository Reference**
   - Used for reproducibility
   - Accessible from W&B run dashboard

#### Authentication
- SSH key-based (from `$HOME/.ssh/`)
- Implicit in SLURM environment

---

## Hyperparameter Optimization Service

### Optuna Integration
**Status**: Active
**Version**: 4.6.0
**Location**: `optuna/main.py`

#### Configuration
1. **Artifact Storage**
   ```python
   artifact_base_path = "../data/optuna/artifacts"
   artifact_store = FileSystemArtifactStore(base_path=artifact_base_path)
   ```

2. **Trial Tracking**
   - File-system based (no remote database)
   - Local filesystem at `data/optuna/artifacts/`

3. **Parallel Execution**
   - joblib backend for parallel trials
   - Configurable via `parallel_backend()`

#### No External Service Dependency
- Optuna runs purely locally
- No cloud API required
- Study data stored locally

---

## Email Notifications

**Type**: SLURM job notifications
**Provider**: System mail service
**Recipients**: g.kerex@gmail.com

#### Trigger Points
```bash
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --mail-type="BEGIN,END"
```

#### Events
- Job BEGIN: When GPU allocated and job starts
- Job END: When job completes or fails
- Optional FAIL, REQUEUE events

#### Configuration
- Email configured in SLURM job headers
- Requires cluster mail relay setup

---

## System & Environment APIs

### CUDA/GPU Runtime
**Type**: System library integration
**Purpose**: GPU acceleration

#### Integration Method
- PyTorch transparently interfaces with CUDA runtime
- No explicit API calls in application code
- Managed through `torch.device("cuda")`

#### Libraries Used
- libcuda.so (NVIDIA driver)
- libcudart.so (CUDA runtime)
- cudnn64_*.so (cuDNN deep learning)
- cublas64_*.so (cuBLAS linear algebra)

### System Information APIs
**Accessed by W&B**
- `/proc/cpuinfo` - CPU count
- GPU queries via `nvidia-smi` equivalent
- Filesystem queries for disk usage

---

## Authentication & Access Control

### API Keys
| Service | Key Location | Type | Required |
|---------|-------------|------|----------|
| W&B | ~/.netrc or env var | OAuth token | Optional (for --wandb) |
| GitHub | ~/.ssh/id_rsa | SSH key | Required for git clone |
| SLURM | (implicit) | HPC account | Required for job submission |

### Account Configuration
```bash
# SLURM account (in job script)
#SBATCH --account=bgak-delta-gpu
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Training/Simulation                      │
└──────────────────────────────────────────────────────────────┘
           │
           ├─→ Load ICs from filesystem (--ic-path)
           │
           ├─→ W&B logging (if --wandb)
           │    └─→ wandb.log() per epoch
           │    └─→ Artifacts to W&B cloud
           │
           ├─→ Save checkpoints to data/<save_name>/model/
           │
           └─→ Output JSON metrics to stdout

┌─────────────────────────────────────────────────────────────┐
│              SLURM Cluster Submission                        │
└──────────────────────────────────────────────────────────────┘
           │
           ├─→ Load modules (python, cuda/11.8.0)
           │
           ├─→ Allocate GPU (1x A40/A100)
           │
           ├─→ Run runner.py
           │
           ├─→ Send email notification (BEGIN/END)
           │
           └─→ Log output to slurm-*.out/err

┌─────────────────────────────────────────────────────────────┐
│           Hyperparameter Optimization (Optuna)              │
└──────────────────────────────────────────────────────────────┘
           │
           ├─→ Run trials in parallel (joblib)
           │
           ├─→ Store trial data locally in data/optuna/artifacts/
           │
           └─→ No external service calls
```

---

## Integration Summary Table

| Service | Type | Required | Version | Purpose |
|---------|------|----------|---------|---------|
| **Weights & Biases** | ML Platform | Optional | 0.23.1 | Experiment tracking & logging |
| **SLURM** | HPC Scheduler | For clusters | (system) | Job submission, GPU allocation |
| **GitHub** | VCS | For dev | (system) | Code versioning, reproducibility |
| **CUDA/cuDNN** | GPU Runtime | For GPU | 11.8 | Deep learning acceleration |
| **Optuna** | HP Optimization | Optional | 4.6.0 | Hyperparameter tuning |
| **System Mail** | Notifications | Optional | (system) | Job completion alerts |
| **Filesystem API** | Data I/O | Required | (system) | Model/checkpoint persistence |

---

## Security Considerations

### No Credential Exposure
- W&B keys stored in `~/.netrc` or environment variables (not in code)
- SSH keys in standard `~/.ssh/` location
- No hardcoded credentials in repository

### Data Privacy
- W&B data stored on Weights & Biases cloud servers
- SLURM logs on cluster filesystem (restricted access)
- Local checkpoints in user's `data/` directory

### Reproducibility
- Config snapshots saved with checkpoints
- Git commit hash captured for version tracking
- Full requirements.txt preserved per W&B run

---

## Notes for Integration Updates

1. **W&B Dashboard**: Accessible at https://wandb.ai/ (project: AITimeStepper)
2. **SLURM Cluster**: Jobs can be monitored via `squeue` on login node
3. **Data Persistence**: Symlink `data/` points to HDD storage on cluster
4. **Experiment Reproducibility**: Always save model + config checkpoint together
