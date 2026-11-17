# Experiment comparison reports

Every study writes reports from actual MLflow trial runs:

- `artifacts/reports/experiment_comparison.json`
- `artifacts/reports/experiment_comparison.csv`
- `artifacts/reports/experiment_comparison.md`
- `artifacts/reports/charts/*.png`

Generate or refresh the report without launching new trials:

```bash
uv run python scripts/compare_experiments.py \
  --config configs/optimization_demo.yaml \
  --study surveillance-resnet18-demo
```

Ranking includes only completed trials with the configured validation metric and compatible dataset fingerprint, split fingerprint, dataset version, label schema, image size, seed, split fractions, architecture, initialization mode, device, and Torch/TorchVision versions. Incompatible, pruned, failed, or missing-metric runs remain visible in the report with a status or warning but are not ranked. Reports include selected parameters, trial counts, provenance, and explicit limitations.

Charts are generated only when compatible completed trial data exists. The five views use the configured objective (validation macro-F1 or validation loss): trial scores, best-so-far history, learning rate, training time, and mean score by fine-tuning strategy. With very few trials, charts are exploratory only and do not imply statistical significance. Reports and charts are also logged to an MLflow comparison run.
