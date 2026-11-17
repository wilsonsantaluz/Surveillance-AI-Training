# Reproducibility and provenance

The dataset split is deterministic for a fixed seed and source layout. Training records a SHA-256 fingerprint of sorted relative image paths and file contents, a split-membership fingerprint, a canonical configuration fingerprint, the Git commit, Python/PyTorch/TorchVision/Optuna/MLflow/DVC versions, model initialization mode, and compute device. Optuna stores the same study-level dataset fingerprint and runtime provenance; the DVC lockfile fingerprint is included when available.

Each trial derives from the same base configuration, seed, and split. Trial checkpoints and reports use `trial_NNN` output directories so sequential or resumed trials cannot overwrite one another. The persistent Optuna SQLite store makes studies resumable; its sampler is seeded, but resumed adaptive TPE suggestions naturally depend on the trials already recorded.

CPU execution is supported and the demo defaults to no pretrained-weight download. CUDA is selected when available; mixed precision is CUDA-only. Deterministic settings reduce variation, but GPU kernels, library versions, hardware, and nondeterministic operations can still prevent bitwise-identical results. Dataset fingerprints identify content but do not validate labels, data provenance, licensing, or representativeness.
