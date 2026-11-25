# Kepler (GTX 650 Ti, sm_30) CUDA line: community-patch version

This is llama.cpp v0.8.0 (upstream commit 8642541) with Kepler support for CUDA 10.2 built with clang, plus custom
Kepler matrix-vector, attention, prompt-path and im2col kernels (rounds r4 to r29).

## Credit and license

The CUDA compatibility port in this line was forward-ported from the patch by GitHub user 6uO, published in
llama.cpp discussion #29961 (https://github.com/ggml-org/llama.cpp/discussions/29961) for ggml 0.25.3.
On 2026-11-23, 6uO agreed to release that patch under the MIT license, in a reply in that discussion:
https://github.com/ggml-org/llama.cpp/discussions/29961#discussioncomment-19833442

Everything in this repository is MIT licensed (see LICENSE):

- llama.cpp and ggml: Copyright (c) 2023-2026 The ggml authors.
- The Kepler / old-GPU CUDA port from discussion #29961: Copyright (c) 2026 6uO, MIT.
- The Kepler kernels, tooling and changes made for this project: Copyright (c) 2026 The oldcard-lab authors, MIT.

The same patches (6uO's port plus this project's kernels), moved to llama.cpp v1.0.0, are also published as a small
patch series with build and test scripts: [oldcard-lab/llama-cuda102](https://github.com/oldcard-lab/llama-cuda102).

Security notes for this build (old llama.cpp base, end-of-life CUDA and driver): see KEPLER-SECURITY.md.
