# Third-party notices

## SpeciesNet

The model artifacts in `models/coreml` and `models/openvino` are converted from
Google SpeciesNet v4.0.3a:

- Source: https://github.com/google/cameratrapai
- Checkpoint: `google/speciesnet/pyTorch/v4.0.3a/1`
- Copyright: Google and SpeciesNet contributors
- License: Apache License 2.0

The upstream license is available at
https://github.com/google/cameratrapai/blob/main/LICENSE.

The exported artifacts add an NCHW-to-NHWC input adapter and convert the
checkpoint to CoreML and OpenVINO formats. No taxonomy output is removed or
reordered.
