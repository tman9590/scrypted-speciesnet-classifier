# SpeciesNet Animal Classifier for Scrypted

This repository packages Google's **SpeciesNet v4.0.3a** camera-trap classifier
for Scrypted's existing custom-classifier contract (`model: resnet`). It is a
second-stage classifier: Scrypted first detects an `animal`, crops that box, and
then this model assigns wildlife metadata used by NVR search.

The deployable model is packaged for all four Scrypted plugins that implement
the custom-classifier loader: CoreML, OpenVINO, ONNX, and NCNN. TensorFlow Lite
does not currently expose that loader.

SpeciesNet uses an EfficientNet V2 M classifier at 480 × 480 and exposes 2,498
taxonomy labels, including 2,066 species plus higher taxonomic groups, human,
vehicle, and blank. It was trained for camera-trap imagery, unlike the previous
experimental model in this repository.

## Why the model changed

The original experiment attempted to train a 530-class YOLO student from 8,175
public photos and provisional Scrypted pseudo-labels. That is too little data
for reliable fine-grained classification, and the resulting unvalidated weights
produced implausible labels. The old teacher/student scripts remain available
for research, but they are no longer the checked-in Scrypted model.

## Build

Use Python 3.11. The official checkpoint is downloaded from Kaggle and cached
outside the repository.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-build.txt

.venv/bin/python distill/speciesnet_scrypted.py \
  --backends coreml openvino onnx ncnn \
  --no-compression
```

`--no-compression` is intentional. Both FP16 exports passed parity validation.
Experimental 8-bit weight compression fit ordinary Git storage but caused
material confidence drift and is rejected for deployment. The large FP16
weight files are therefore stored with Git LFS.

## Validate

Validation checks every configured artifact, label order, finite outputs,
source/export top-1 agreement, top-5 overlap, and probability drift.

```sh
.venv/bin/python distill/validate_speciesnet_scrypted.py \
  --images work/source-images \
  --samples 20

python3 -m unittest discover -s tests -v
python3 -m compileall -q distill tests
```

Passing export parity proves that the packaged backends preserve the source
model's behavior. It does not substitute for a labeled validation set from the
installed cameras.

## Add to Scrypted

The repository root contains the one canonical [`config.json`](config.json).
Use the same raw URL in CoreML, OpenVINO, ONNX, or NCNN:

`https://raw.githubusercontent.com/tman9590/scrypted-speciesnet-classifier/main/config.json`

In the matching Scrypted detector plugin, choose **Create Device** under
**Models**, name it `SpeciesNet Animals`, and paste that URL.
Select the resulting classifier as the camera's animal classifier. Start with a
classification threshold of `0.50`; review real day/night events before using
species labels for alerts.

Paste the full `config.json` URL rather than the repository URL. Scrypted's
repository shortcut still assumes separate `models/<backend>/config.json`
files, while this package intentionally keeps one canonical root config.

## Optional site-specific fine-tuning

The RTX 5070 Ti is useful after camera crops have been reviewed and labeled.
Fine-tuning should adapt the classifier to the site's common animals and night
vision, while retaining a broad background/unknown set. Do not retrain the old
529-class detector on its current sparse pseudo-labels.

## Licenses and upstream

Repository code is Apache-2.0. SpeciesNet is maintained by Google at
<https://github.com/google/cameratrapai>; review its model card and license
before redistribution. The legacy BioCLIP/MegaDetector distillation pipeline
has its own upstream licenses documented in the source and model card.
