import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .planner import Result, choose, sensitivity_order


def load_data(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        inputs = data["inputs"]
        labels = data["labels"]
    if len(inputs) != len(labels) or not len(inputs):
        raise ValueError(f"{path}: inputs and labels must have matching nonzero batch sizes")
    return inputs, labels


def make_reader(inputs, input_name):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.position = 0

        def get_next(self):
            if self.position >= len(inputs):
                return None
            row = inputs[self.position:self.position + 1]
            self.position += 1
            return {input_name: row}

        def rewind(self):
            self.position = 0

    return Reader()


def measure(path: Path, inputs, labels, input_name: str, repeats: int) -> Result:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise ValueError("only single-input models are supported")
    if session.get_inputs()[0].name != input_name:
        raise ValueError("model input name changed across variants")
    output_name = session.get_outputs()[0].name
    # One warmup, then matched batch calls across all variants.
    session.run([output_name], {input_name: inputs[:1]})
    times = []
    correct = 0
    for i in range(len(inputs)):
        feed = {input_name: inputs[i:i + 1]}
        outputs = None
        for _ in range(repeats):
            start = time.perf_counter()
            outputs = session.run([output_name], feed)[0]
            times.append((time.perf_counter() - start) * 1000)
        if outputs.ndim != 2 or outputs.shape[0] != 1:
            raise ValueError("first model output must be [batch, classes] logits")
        correct += int(int(np.argmax(outputs[0])) == int(labels[i]))
    return Result(path.stem, (), correct / len(inputs), statistics.median(times), path.stat().st_size)


def quantize(source: Path, target: Path, calibration, input_name: str, excluded: tuple[str, ...]):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        str(source), str(target), make_reader(calibration, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Conv"],
        nodes_to_exclude=list(excluded),
    )


def run(args):
    import onnx
    import onnxruntime as ort

    if args.repeats < 1 or args.max_candidates < 0:
        raise ValueError("repeats must be positive and max-candidates nonnegative")
    calibration, _ = load_data(args.calibration)
    validation, labels = load_data(args.validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph = onnx.load(str(args.model)).graph
    nodes = [n.name for n in graph.node if n.op_type in ("MatMul", "Conv") and n.name]
    if not nodes:
        raise ValueError("model needs named MatMul or Conv nodes")
    input_name = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    baseline = measure(args.model, validation, labels, input_name, args.repeats)
    candidates = []

    def candidate(name, excluded):
        path = args.output_dir / f"{name}.onnx"
        quantize(args.model, path, calibration, input_name, excluded)
        measured = measure(path, validation, labels, input_name, args.repeats)
        result = Result(name, excluded, measured.accuracy, measured.latency_ms, measured.size_bytes)
        candidates.append(result)
        return result

    full = candidate("all_int8", ())
    probes = [candidate(f"float_{index}", (node,)) for index, node in enumerate(nodes[:args.max_candidates])]
    ordered = sensitivity_order(full, probes)
    excluded = []
    for index, node in enumerate(ordered, start=1):
        excluded.append(node)
        candidate(f"mixed_{index}", tuple(excluded))
    selected = choose(baseline, candidates, args.max_accuracy_drop)
    report = {
        "model": str(args.model), "onnxruntime_version": ort.__version__,
        "max_accuracy_drop": args.max_accuracy_drop,
        "baseline": asdict(baseline),
        "candidates": [asdict(c) for c in candidates],
        "selected": asdict(selected),
        "selected_path": str(args.model if selected is baseline else args.output_dir / f"{selected.name}.onnx"),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_path": report["selected_path"], "report": str(report_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Tune static INT8 quantization for a classifier")
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
