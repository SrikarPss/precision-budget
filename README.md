# Precision Budget

Precision Budget tunes an ONNX classifier for a **device-specific latency and quality budget**. It calibrates static INT8 QDQ variants, measures validation accuracy and latency, and keeps sensitive MatMul/Conv nodes in FP32. The deployment decision uses measured results rather than a fixed quantization recipe.

The iterative deployment selector rejects quantization when it is slower or too inaccurate. The current CLI supports single-input classifiers whose output is a batch of class logits.

## Setup and input

Use Python 3.13. From the repository root, create an environment and install:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start with a generated example:

```bash
python -m precision_budget.demo example
python -m precision_budget example/model.onnx example/calibration.npz \
  example/validation.npz --output-dir example/results
```

For your own model, provide an ONNX file and two NPZ files. Each NPZ needs `inputs` (batch dimension first) and `labels` (one integer class per row). Calibration and validation sets should be disjoint. The input array must have the model's expected dtype and shape; the first output must be class logits.

```bash
python -m precision_budget model.onnx calibration.npz validation.npz \
  --max-accuracy-drop 0.01 --output-dir results
```

The command writes candidate ONNX files and `report.json`. The report records baseline and candidate accuracy, median latency, file size, node exclusions, and which candidate was selected. Selection requires at least a 5% measured latency improvement to reduce timing-noise decisions. All measurements are local to the machine running the command. Use `--max-candidates` to cap sensitivity probes on large graphs.

The selection planner and its tests use only the Python standard library:

```bash
python3 -m unittest discover -s tests -v
```

The generated example was run end to end with ONNX Runtime on macOS; the tiny model showed no reliable speed gain, so the float baseline was selected. Quantization uses the [ONNX Runtime static QDQ API](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), including `nodes_to_exclude` to retain sensitive nodes in float precision. Results for a real model will depend on its data and deployment hardware.
