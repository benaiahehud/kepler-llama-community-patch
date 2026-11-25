# llama.cpp for NVIDIA Kepler (sm_30): community-patch line

> **This is not the official llama.cpp.** It is llama.cpp **v0.8.0** (upstream commit `8642541`) with CUDA support
> for Kepler GPUs of compute capability 3.0 (GeForce GTX 600 / 700 series such as the GTX 650 Ti, 660, 670, 680)
> and custom Kepler kernels. For current llama.cpp, go to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
> The upstream README follows below this section and still applies to everything that is not Kepler-specific.
>
> **Most people should use [oldcard-lab/llama-cuda102](https://github.com/oldcard-lab/llama-cuda102) instead:** the
> same patches (6uO's port plus the Kepler kernels), moved to llama.cpp v1.0.0, as a small patch series with install,
> build and test scripts and a strict kernel checker. This repository is the original development line on v0.8.0:
> the same patches applied to a full llama.cpp tree.

Current CUDA releases dropped Kepler. This line builds llama.cpp's CUDA backend with CUDA 10.2, the last toolkit
that supports sm_30, using clang as the CUDA compiler, and adds:

- **Kepler matrix-vector kernels** for token generation (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, IQ3_XXS, F16) and the
  attention matrix-vector products. Other quant types use the stock CUDA kernels.
- **A faster prompt path** for short batches on Kepler (row-chunked dequantization, cuBLAS shapes padded to its fast
  sm_30 kernel).
- **A Kepler im2col** for convolution models.
- **A FunctionGemma tool-call parser** for `llama-server --jinja`.

The Kepler kernels only run below compute capability 5.0, and each Kepler path can be turned off:
`GGML_CUDA_KEPLER_MMV=0`, `GGML_CUDA_KEPLER_PP=off`, `GGML_CUDA_KEPLER_IM2COL=off`. `GGML_CUDA_KEPLER_TRACE=1` prints
which kernel ran. The compatibility port targets old CUDA toolkits and old GPUs and was tested only on the GTX 650 Ti.

## Measured speeds

GeForce GTX 650 Ti (GK106, 1 GB) with an i7-4770, Ubuntu 26.04, NVIDIA 470.256.02. Release gate of 2026-11-23
(round 29): `llama-bench -ngl 99 -t 1 -fa 0`, generation = tg128, prompt = pp256 at the micro-batch shown.

| Model | Generation (t/s) | Prompt (t/s) |
|---|---:|---:|
| LFM2.5-350M Q4_K_M | 88.0 | 616 (`-ub 128`) |
| SmolLM2-360M-Instruct Q8_0 | 44.7 | |
| Qwen3-0.6B Q8_0 | 32.6 | 322 (`-ub 128`) |
| LFM2.5-1.2B-Instruct Q4_K_M | 32.5 | 95 (`-ub 64`) |
| Llama-3.2-1B-Instruct Q4_K_M | 29.0 | |
| gemma-3-1b-it Q4_K_M | 20.1 | |
| Qwen3-1.7B Q3_K_S (own imatrix quant) | 23.2 | 32.5 (`-ub 32`) |

For comparison, before the custom kernels the same card generated 42.7 t/s on LFM2.5-350M with the stock CUDA
kernels, and 71.8 t/s with a tuned build of llama.cpp's Vulkan backend. The release gate also checks perplexity against stored values on 29 models, llama.cpp's
full operator test suite on the GPU, and greedy output.

## Requirements

- A GPU with compute capability 3.0. Only the GTX 650 Ti has been tested. The 660 / 670 / 680 and other sm_30 cards
  should work but are untested; sm_35 / sm_37 (GTX 780, Titan, Tesla K40 / K80) are not targeted by this build.
- Linux x86-64 with the NVIDIA **470** legacy driver, the last branch that supports Kepler.
- The **CUDA 10.2 toolkit** only (do not install the 440 driver bundled with it).
- **clang 22** (`clang-22`, `clang++-22`, `ld.lld-22`), cmake, ninja.

## Build

Install the CUDA 10.2 toolkit into a private directory, without root and without its driver:

```bash
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
echo "3d816c583759efa8567a92c1a2af90db  cuda_10.2.89_440.33.01_linux.run" | md5sum -c -
sh cuda_10.2.89_440.33.01_linux.run --noexec --keep --target cuda-payload
cp -a cuda-payload/builds/cuda-toolkit ~/cuda-10.2
cp -a cuda-payload/builds/cublas/lib64/. ~/cuda-10.2/targets/x86_64-linux/lib/
```

Then configure and build from the root of this repository (this is the configuration our releases are built with):

```bash
C=~/cuda-10.2
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-22 -DCMAKE_CXX_COMPILER=clang++-22 \
  -DCMAKE_CUDA_COMPILER=clang++-22 -DCMAKE_CUDA_HOST_COMPILER=clang++-22 -DCUDAToolkit_ROOT=$C \
  -DCMAKE_CUDA_ARCHITECTURES=30 \
  -DCMAKE_CUDA_FLAGS="--cuda-path=$C --no-cuda-version-check -Wno-unknown-cuda-version" \
  -DCMAKE_EXE_LINKER_FLAGS="--ld-path=$(command -v ld.lld-22) -Wl,--disable-new-dtags" \
  -DCMAKE_SHARED_LINKER_FLAGS="--ld-path=$(command -v ld.lld-22) -Wl,--disable-new-dtags" \
  -DCMAKE_BUILD_RPATH="$PWD/build/bin;$C/lib64" \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FA=OFF -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_NO_VMM=ON \
  -DLLAMA_BUILD_HTML=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF
cmake --build build -j "$(nproc)"
build/bin/test-backend-ops -b CUDA0 -o MUL_MAT     # correctness check on the GPU against the CPU
```

## Using it

```bash
build/bin/llama-server -m LFM2.5-350M-Q4_K_M.gguf -ngl 99 -fa off -t 1
build/bin/llama-cli    -m Qwen3-1.7B-Q3_K_S.gguf  -ngl 99 -fa off -c 1024 -ub 32
```

## Known limits

- **Memory:** a 1 GB card fits models up to about 900 MB with a modest context. Models of about 700 MB and up need a
  smaller micro-batch (`-ub 128`; `-ub 64` for gemma-3-1b). 1.7B models fit only at about 3 bits per weight
  (Q3_K_S / Q3_K_M, best with an importance matrix) with `-c 1024 -ub 32`; `-ctk q8_0` (8-bit K cache) frees room
  for `-ub 64`. The default 4096-token context does not fit for the larger models; pass `-c 512` or `-c 1024`.
- **No flash attention** for sm_30. Pass `-fa off` (`-fa 0` for `llama-bench`): the default "auto" costs about 8%.
- **No CUDA graphs, no BF16 on the GPU** (CUDA 10.2 has none; BF16 operations run on the CPU).
- **Frozen on v0.8.0:** newer models that need later llama.cpp versions will not load, and upstream fixes made after
  v0.8.0 are not included. See [KEPLER-SECURITY.md](KEPLER-SECURITY.md) before running it anywhere that matters.

## How it was made: mostly by AI

Most of the engineering in this line, and arguably all of the heavy lifting, was done by AI coding agents:

- **Kernels** (matrix-vector, attention, prompt path, im2col): written and iterated by AI coding agents (OpenAI
  Codex and Google Gemini via AGY), round by round, each round driven by a written design brief and measured on the
  real card.
- **Forward port of 6uO's patch onto v0.8.0:** OpenAI Codex agents (the parts that no longer applied) and an
  Anthropic Claude agent (debugging and fixes).
- **Coordination, test design, release engineering and this documentation:** the Claude agent.
- **Reviews:** independent reviews by other models (Gemini, GPT, Grok, Muse Code) at design points and when
  progress stalled.
- **The human owner** set the goals, provided the hardware, approved each design direction and release policy, and
  makes every public post.

Because AI wrote the kernels, they were never trusted on average-error tests alone. During development every
release had to pass a strict per-row checker on top of llama.cpp's own tests: outputs poisoned with NaN before each
run (catching rows a kernel never writes), two runs required to be bitwise identical, and every output row compared
against an fp64 reference. Those checks follow Wafer's
[field guide to reward hacking in AI kernel generation](https://www.wafer.ai/blog/reward-hacks-field-guide).

## Credit and license

The CUDA compatibility port comes from the patch by **6uO** in llama.cpp discussion
[#29961](https://github.com/ggml-org/llama.cpp/discussions/29961), which 6uO released under the MIT license. The
whole repository is MIT licensed; details in [KEPLER-NOTICE.md](KEPLER-NOTICE.md). Security notes:
[KEPLER-SECURITY.md](KEPLER-SECURITY.md).

## Related work

We found no other public llama.cpp work with custom kernels for sm_30. For later Kepler parts (sm_35 / sm_37, such
as the Tesla K40 / K80), see [babal35/llamacpp-kepler](https://github.com/babal35/llamacpp-kepler) (CUDA 11.x build
fixes) and [theIvanR/llama-on-legacy-gpu](https://github.com/theIvanR/llama-on-legacy-gpu) (compute capability
3.5-6.2, multi-GPU K40c). Reports from other Kepler cards are welcome as issues.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
