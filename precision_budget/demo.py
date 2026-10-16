"""Generate a small deterministic classifier and disjoint NPZ datasets."""
import argparse
from pathlib import Path


def create(directory: Path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    first = rng.normal(0, .5, (8, 16)).astype(np.float32)
    second = rng.normal(0, .5, (16, 3)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "first_weight"], ["hidden"], name="feature_projection"),
         helper.make_node("Relu", ["hidden"], ["activated"], name="activation"),
         helper.make_node("MatMul", ["activated", "second_weight"], ["logits"], name="class_projection")],
        "tiny_classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 3])],
        [numpy_helper.from_array(first, "first_weight"), numpy_helper.from_array(second, "second_weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    model_path = directory / "model.onnx"
    onnx.save(model, str(model_path))
    for name, count in (("calibration", 40), ("validation", 80)):
        inputs = rng.normal(size=(count, 8)).astype(np.float32)
        labels = np.argmax(np.maximum(inputs @ first, 0) @ second, axis=1).astype(np.int64)
        np.savez(directory / f"{name}.npz", inputs=inputs, labels=labels)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(create(args.output_dir))
