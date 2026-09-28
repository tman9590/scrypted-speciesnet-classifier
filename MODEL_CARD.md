# Model card: SpeciesNet v4.0.3a for Scrypted

## Intended use

This package classifies an animal crop supplied by Scrypted. It does not detect
or localize animals by itself. The returned common-name label is intended for
NVR metadata, search, and notification assistance.

## Source model

- Upstream: Google SpeciesNet v4.0.3a
- Architecture: EfficientNet V2 M
- Input: one 480 × 480 RGB animal crop
- Output: 2,498 logits
- Training domain: camera-trap imagery

The output vocabulary contains species, higher taxa, human, vehicle, and blank.
Four upstream rows without common names use their scientific names. Duplicate
common names are disambiguated with scientific names and, only when necessary,
the first eight characters of the upstream taxonomy identifier. Output indices
are never reordered or removed.

## Scrypted conversion

Scrypted supplies NCHW float RGB data in the [0, 1] range. A wrapper permutes it
to the NHWC tensor expected by SpeciesNet. The Scrypted configs use the ResNet
classifier parser with identity normalization. OpenVINO and CoreML artifacts
are FP16-weight exports; the large weight files are stored with Git LFS.

## Validation status

The export validator compares source PyTorch, OpenVINO, and CoreML inference on
real wildlife images. A build is rejected if a prediction at or above Scrypted's
default 0.50 threshold changes, fewer than four of the source top five remain,
probability drift exceeds 0.10, an artifact is missing, or a non-finite value
appears. Below 0.50, a nearly tied FP16 top class may swap because Scrypted would
emit neither result. Experimental 8-bit exports failed these criteria and are
not deployed.

This is conversion validation, not a claim of site-specific accuracy. Accuracy
must also be measured on reviewed Scrypted animal crops, split by camera,
species, daylight, infrared/night, weather, motion blur, and partial views.

## Limitations

- Similar species, domestic breeds, juveniles, partial animals, and tiny crops
  can be confused.
- A global classifier without SpeciesNet's location/date roll-up may suggest a
  visually similar species that is geographically implausible.
- Camera-trap training data does not perfectly match residential security
  cameras.
- Softmax confidence is not a calibrated biological probability.
- Human, vehicle, blank, or higher-level taxa may appear when the input crop is
  not a clear animal species.

Use the labels as search metadata, not as scientific evidence or for
safety-critical wildlife decisions.
