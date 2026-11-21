#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_mmvk(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

bool ggml_cuda_mul_mat_vec_kepler(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_mul_mat_vec_kepler_attn(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
