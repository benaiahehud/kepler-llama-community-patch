# Kepler (GTX 650 Ti, sm_30) CUDA line: community-patch version

This is llama.cpp v0.8.0 (upstream commit 8642541) with Kepler support for CUDA 10.2 built with clang, plus custom
Kepler matrix-vector, attention, prompt-path and im2col kernels (rounds r4 to r29).

## Origin and license status: keep this repository private

The CUDA compatibility port in this line was forward-ported from the patch published by GitHub user 6uO in
llama.cpp discussion #29961 (https://github.com/ggml-org/llama.cpp/discussions/29961), written for ggml 0.25.3.
That patch states no license. Permission to release it under MIT was requested in that discussion on
2026-11-23. Until the author agrees, this version must not be made public or redistributed.

llama.cpp itself is MIT licensed (see LICENSE). The Kepler kernels written for this project are also
available, without the #29961-derived code, as a separate clean-room patch series (kepler-llama).
