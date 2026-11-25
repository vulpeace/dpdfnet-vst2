# DPDFNet native inference

This directory contains the Ceva-IP DPDFNet C runtime and the generated DPDFNet8 48 kHz HR model from upstream commit `6a5dbd3ea5dbea88c30a2be8e7c689b5eb26b533`. The files were taken from the copy in `dpdf-voice/vendor/hushmic-denoiser/native`, including its MSVC portability changes in `runtime/dpdf_dprnn.c`, `runtime/fp16.c`, `runtime/int8.c`, and `runtime/internal.h`.

The C sources and model code use the [Apache License 2.0](LICENSE). Fetch the matching weight file with [fetch-native-weights.ps1](../../scripts/fetch-native-weights.ps1).
