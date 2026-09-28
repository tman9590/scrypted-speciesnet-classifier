#!/usr/bin/env python3
"""Export Google's SpeciesNet crop classifier for Scrypted classifier backends."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).parents[1]
DEFAULT_HANDLE = "google/speciesnet/pyTorch/v4.0.3a/1"
IMAGE_SIZE = 480
MODEL_BASENAME = "speciesnet-v4.0.3a"


def parse_speciesnet_labels(lines: Iterable[str]) -> list[str]:
    """Convert SpeciesNet taxonomy rows into unique, readable Scrypted labels."""
    rows: list[tuple[str, str, str]] = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split(";")
        if len(fields) != 7:
            raise ValueError(f"SpeciesNet label row {number} has {len(fields)} fields, expected 7")
        taxonomy_id, animal_class, order, family, genus, species, common_name = fields
        scientific = " ".join(part for part in (genus, species) if part)
        fallback = scientific or family or order or animal_class or f"taxonomy {fields[0]}"
        rows.append((common_name or fallback, scientific, taxonomy_id))

    counts = Counter(label.casefold() for label, _, _ in rows)
    labels = [
        f"{label} ({scientific})" if counts[label.casefold()] > 1 and scientific else label
        for label, scientific, _ in rows
    ]
    candidate_counts = Counter(label.casefold() for label in labels)
    labels = [
        f"{label} [{taxonomy_id[:8]}]" if candidate_counts[label.casefold()] > 1 else label
        for label, (_, _, taxonomy_id) in zip(labels, rows)
    ]
    return labels


def scrypted_config(labels: list[str], files: list[str]) -> dict:
    return {
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "model": "resnet",
        # Scrypted converts uint8 RGB crops to [0, 1] before this normalization.
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "files": files,
        "labels": {str(index): label for index, label in enumerate(labels)},
    }


def write_config(directory: Path, labels: list[str], files: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(scrypted_config(labels, files), indent=2, ensure_ascii=False) + "\n"
    )


def load_checkpoint(handle: str, model_dir: Path | None):
    import kagglehub
    import torch

    root = model_dir or Path(kagglehub.model_download(handle))
    info = json.loads((root / "info.json").read_text())
    labels = parse_speciesnet_labels((root / info["classifier_labels"]).read_text().splitlines())
    model = torch.load(root / info["classifier"], map_location="cpu", weights_only=False).eval()
    return root, info, labels, model


def build_adapter(model):
    import torch

    class NchwSpeciesNet(torch.nn.Module):
        """Adapt Scrypted's NCHW float crop to SpeciesNet's NHWC input."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image):
            return self.inner(image.permute(0, 2, 3, 1))

    return NchwSpeciesNet(model).eval()


def export_onnx(model, example, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        destination,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )


def export_openvino(onnx_path: Path, directory: Path, labels: list[str], compress: bool) -> None:
    import openvino as ov

    directory.mkdir(parents=True, exist_ok=True)
    xml = directory / f"{MODEL_BASENAME}.xml"
    model = ov.convert_model(str(onnx_path))
    if compress:
        import nncf

        model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(model, xml, compress_to_fp16=not compress)
    write_config(directory, labels, [xml.name, xml.with_suffix(".bin").name])


def export_coreml(model, example, directory: Path, labels: list[str], compress: bool) -> None:
    import coremltools as ct
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    package = directory / f"{MODEL_BASENAME}.mlpackage"
    if package.exists():
        shutil.rmtree(package)
    traced = torch.jit.trace(model, example, strict=True)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=tuple(example.shape))],
        outputs=[ct.TensorType(name="logits")],
    )
    if compress:
        from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights

        converted = palettize_weights(
            converted,
            OptimizationConfig(
                global_config=OpPalettizerConfig(mode="kmeans", nbits=8, num_kmeans_workers=1)
            ),
        )
    converted.save(str(package))
    files = [str(path.relative_to(directory)) for path in sorted(package.rglob("*")) if path.is_file()]
    write_config(directory, labels, files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle", default=DEFAULT_HANDLE)
    parser.add_argument("--model-dir", type=Path, help="Use an already downloaded SpeciesNet model directory")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--work", type=Path, default=ROOT / "work" / "speciesnet-export")
    parser.add_argument("--backends", nargs="+", choices=["onnx", "openvino", "coreml"], default=["openvino", "coreml"])
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Keep FP16 artifacts even when a file will exceed GitHub's 100 MB limit",
    )
    args = parser.parse_args()

    import torch

    source, info, labels, checkpoint = load_checkpoint(args.handle, args.model_dir)
    if len(labels) != 2498:
        raise ValueError(f"Unexpected SpeciesNet class count: {len(labels)}")
    model = build_adapter(checkpoint)
    torch.manual_seed(7)
    example = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        output = model(example)
    if tuple(output.shape) != (1, len(labels)) or not torch.isfinite(output).all():
        raise ValueError(f"Invalid checkpoint output shape or values: {tuple(output.shape)}")

    args.work.mkdir(parents=True, exist_ok=True)
    onnx_path = args.work / f"{MODEL_BASENAME}.onnx"
    if any(backend in args.backends for backend in ("onnx", "openvino")):
        export_onnx(model, example, onnx_path)
    if "onnx" in args.backends:
        destination = args.models / "onnx" / onnx_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, destination)
        write_config(destination.parent, labels, [destination.name])
    if "openvino" in args.backends:
        export_openvino(onnx_path, args.models / "openvino", labels, not args.no_compression)
    if "coreml" in args.backends:
        export_coreml(model, example, args.models / "coreml", labels, not args.no_compression)

    manifest = {
        "source": str(source),
        "version": info["version"],
        "handle": args.handle,
        "class_count": len(labels),
        "input_shape": list(example.shape),
        "backends": args.backends,
        "weight_compression": "int8" if not args.no_compression else "fp16",
    }
    (args.work / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
