#include "mmvk.cuh"

#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

static_assert(sizeof(block_q4_K) == 144, "unexpected block_q4_K size");
static_assert(offsetof(block_q4_K, scales) == 4, "unexpected block_q4_K scales offset");
static_assert(offsetof(block_q4_K, qs) == 16, "unexpected block_q4_K qs offset");

static_assert(sizeof(block_q6_K) == 210, "unexpected block_q6_K size");
static_assert(offsetof(block_q6_K, ql) == 0, "unexpected block_q6_K ql offset");
static_assert(offsetof(block_q6_K, qh) == 128, "unexpected block_q6_K qh offset");
static_assert(offsetof(block_q6_K, scales) == 192, "unexpected block_q6_K scales offset");
static_assert(offsetof(block_q6_K, d) == 208, "unexpected block_q6_K d offset");

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Selectable Kepler (sm_30) launch geometries:
// - GGML_CUDA_KEPLER_MMV_Q4K: "auto" (default), "vk2", "sy", "c8", "swiz", "vk", "32x1x4", "32x1x8", "32x2x4", "8x1x4"
// - GGML_CUDA_KEPLER_MMV_Q6K: "auto" (default), "d32", "nb4", "smem", "vk", "32x2x4", "32x1x4", "32x1x8", "16x1x4"
// - GGML_CUDA_KEPLER_MMV_Q8: "auto" (default), "pair", "safe"
enum mmvk_q4_k_variant {
    MMVK_Q4_K_AUTO = 0,
    MMVK_Q4_K_VK2,
    MMVK_Q4_K_SY,
    MMVK_Q4_K_C8,
    MMVK_Q4_K_SWIZ,
    MMVK_Q4_K_VK,
    MMVK_Q4_K_32x1x4,
    MMVK_Q4_K_32x1x8,
    MMVK_Q4_K_32x2x4,
    MMVK_Q4_K_8x1x4,
};

enum mmvk_q6_k_variant {
    MMVK_Q6_K_AUTO = 0,
    MMVK_Q6_K_D32,
    MMVK_Q6_K_NB4,
    MMVK_Q6_K_SMEM,
    MMVK_Q6_K_VK,
    MMVK_Q6_K_32x2x4,
    MMVK_Q6_K_32x1x4,
    MMVK_Q6_K_32x1x8,
    MMVK_Q6_K_16x1x4,
};

enum mmvk_q8_0_variant {
    MMVK_Q8_0_AUTO = 0,
    MMVK_Q8_0_PAIR,
    MMVK_Q8_0_SAFE,
    MMVK_Q8_0_WS,
};

static mmvk_q8_0_variant ggml_cuda_kepler_mmv_q8_variant() {
    static const mmvk_q8_0_variant variant = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_MMV_Q8");
        if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0) {
            return MMVK_Q8_0_AUTO;
        }
        if (std::strcmp(env, "pair") == 0) {
            return MMVK_Q8_0_PAIR;
        }
        if (std::strcmp(env, "safe") == 0) {
            return MMVK_Q8_0_SAFE;
        }
        if (std::strcmp(env, "ws") == 0) {
            return MMVK_Q8_0_WS;
        }
        GGML_LOG_WARN("%s: unknown GGML_CUDA_KEPLER_MMV_Q8=\"%s\", using auto (valid: auto ws pair safe)\n", __func__, env);
        return MMVK_Q8_0_AUTO;
    }();
    return variant;
}

static mmvk_q4_k_variant ggml_cuda_kepler_mmv_q4k_variant() {
    static const mmvk_q4_k_variant variant = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_MMV_Q4K");
        if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0) {
            return MMVK_Q4_K_AUTO;
        }
        if (std::strcmp(env, "vk2") == 0) {
            return MMVK_Q4_K_VK2;
        }
        if (std::strcmp(env, "sy") == 0) {
            return MMVK_Q4_K_SY;
        }
        if (std::strcmp(env, "c8") == 0) {
            return MMVK_Q4_K_C8;
        }
        if (std::strcmp(env, "swiz") == 0) {
            return MMVK_Q4_K_SWIZ;
        }
        if (std::strcmp(env, "vk") == 0) {
            return MMVK_Q4_K_VK;
        }
        if (std::strcmp(env, "32x1x4") == 0) {
            return MMVK_Q4_K_32x1x4;
        }
        if (std::strcmp(env, "32x1x8") == 0) {
            return MMVK_Q4_K_32x1x8;
        }
        if (std::strcmp(env, "32x2x4") == 0) {
            return MMVK_Q4_K_32x2x4;
        }
        if (std::strcmp(env, "8x1x4") == 0) {
            return MMVK_Q4_K_8x1x4;
        }
        GGML_LOG_WARN("%s: unknown GGML_CUDA_KEPLER_MMV_Q4K=\"%s\", using auto "
                      "(valid: auto vk2 sy c8 swiz vk 32x1x4 32x1x8 32x2x4 8x1x4)\n", __func__, env);
        return MMVK_Q4_K_AUTO;
    }();
    return variant;
}

static mmvk_q6_k_variant ggml_cuda_kepler_mmv_q6k_variant() {
    static const mmvk_q6_k_variant variant = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_MMV_Q6K");
        if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0) {
            return MMVK_Q6_K_AUTO;
        }
        if (std::strcmp(env, "d32") == 0) {
            return MMVK_Q6_K_D32;
        }
        if (std::strcmp(env, "nb4") == 0) {
            return MMVK_Q6_K_NB4;
        }
        if (std::strcmp(env, "smem") == 0) {
            return MMVK_Q6_K_SMEM;
        }
        if (std::strcmp(env, "vk") == 0) {
            return MMVK_Q6_K_VK;
        }
        if (std::strcmp(env, "32x2x4") == 0) {
            return MMVK_Q6_K_32x2x4;
        }
        if (std::strcmp(env, "32x1x4") == 0) {
            return MMVK_Q6_K_32x1x4;
        }
        if (std::strcmp(env, "32x1x8") == 0) {
            return MMVK_Q6_K_32x1x8;
        }
        if (std::strcmp(env, "16x1x4") == 0) {
            return MMVK_Q6_K_16x1x4;
        }
        GGML_LOG_WARN("%s: unknown GGML_CUDA_KEPLER_MMV_Q6K=\"%s\", using auto "
                      "(valid: auto d32 nb4 smem vk 32x2x4 32x1x4 32x1x8 16x1x4)\n", __func__, env);
        return MMVK_Q6_K_AUTO;
    }();
    return variant;
}

static __device__ __forceinline__ float q4_k_dot4_lo_bias(
        const uint32_t u, const float4 y, float sum) {
    // Exponents 0x4B/0x47 place low nibbles (bits 0..3 and 8..11) at integer scale without I2F.
    const uint32_t u_hi = u >> 16;
    sum = fmaf(__fadd_rn(__uint_as_float((u    & 0x000Fu) | 0x4B000000u), -8388608.0f), y.x, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((u    & 0x0F00u) | 0x47000000u), -32768.0f),   y.y, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((u_hi & 0x000Fu) | 0x4B000000u), -8388608.0f), y.z, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((u_hi & 0x0F00u) | 0x47000000u), -32768.0f),   y.w, sum);
    return sum;
}

static __device__ __forceinline__ float q4_k_dot4_hi_bias(
        const uint32_t u, const float4 y, float sum) {
    // Exponents 0x49/0x45 fold the 1/16 shift of high nibbles (bits 4..7 and 12..15) directly.
    const uint32_t u_hi = u >> 16;
    sum = fmaf(__fadd_rn(__uint_as_float((u    & 0x00F0u) | 0x49000000u), -524288.0f), y.x, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((u    & 0xF000u) | 0x45000000u), -2048.0f),   y.y, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((u_hi & 0x00F0u) | 0x49000000u), -524288.0f), y.z, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((u_hi & 0xF000u) | 0x45000000u), -2048.0f),   y.w, sum);
    return sum;
}

static __device__ __forceinline__ float q4_k_apply_hdr(
        const uint4 hdr, const uint32_t sc_shift, const uint32_t hi_mask,
        const float sum_q0, const float sum_q1,
        const float sum_y0, const float sum_y1,
        float acc) {
    half2 dm_h2;
    memcpy(&dm_h2, &hdr.x, sizeof(dm_h2));
    const float2 dm = __half22float2(dm_h2);

    const uint32_t s0_16 = hdr.y >> sc_shift;
    const uint32_t s1_16 = hdr.z >> sc_shift;
    const uint32_t s2_16 = hdr.w >> sc_shift;

    const uint32_t sc_lo   = s0_16 & 0x3F3Fu;
    const uint32_t m_lo    = s1_16 & 0x3F3Fu;
    const uint32_t sc_hi   = (s2_16 & 0x0F0Fu) | ((s0_16 & 0xC0C0u) >> 2);
    const uint32_t m_hi    = ((s2_16 >> 4) & 0x0F0Fu) | ((s1_16 & 0xC0C0u) >> 2);
    const uint32_t sc_pair = (sc_lo & ~hi_mask) | (sc_hi & hi_mask);
    const uint32_t m_pair  = (m_lo  & ~hi_mask) | (m_hi  & hi_mask);

    // Decode one sub-block's scale and min at a time to keep register pressure low.
    const float sc0 = __fadd_rn(__uint_as_float((sc_pair & 0x003Fu) | 0x4B000000u), -8388608.0f);
    const float m0  = __fadd_rn(__uint_as_float((m_pair  & 0x003Fu) | 0x4B000000u), -8388608.0f);
    acc = fmaf(dm.x * sc0, sum_q0, fmaf(-(dm.y * m0), sum_y0, acc));

    const float sc1 = __fadd_rn(__uint_as_float((sc_pair & 0x3F00u) | 0x47000000u), -32768.0f);
    const float m1  = __fadd_rn(__uint_as_float((m_pair  & 0x3F00u) | 0x47000000u), -32768.0f);
    acc = fmaf(dm.x * sc1, sum_q1, fmaf(-(dm.y * m1), sum_y1, acc));
    return acc;
}

template <int THREADS_PER_ROW, int ROWS_PER_GROUP, int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
static __global__ void mul_mat_vec_q4_K_kepler(
        const block_q4_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    static_assert(THREADS_PER_ROW >= 8 && THREADS_PER_ROW <= WARP_SIZE && (THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0,
                  "THREADS_PER_ROW must be power of 2 in [8, 32]");
    static_assert(ROWS_PER_GROUP >= 1, "ROWS_PER_GROUP must be >= 1");

    constexpr int GROUPS_PER_WARP  = WARP_SIZE / THREADS_PER_ROW;
    constexpr int GROUPS_PER_BLOCK = WARPS_PER_BLOCK * GROUPS_PER_WARP;
    constexpr int ROWS_PER_BLOCK   = GROUPS_PER_BLOCK * ROWS_PER_GROUP;
    constexpr int BLOCKS_PER_ITER  = THREADS_PER_ROW / 8;

    const int warp_row0 = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * GROUPS_PER_WARP * ROWS_PER_GROUP;
    if (warp_row0 >= nrows) {
        return;
    }

    const int lane          = threadIdx.x;
    const int group_in_warp = lane / THREADS_PER_ROW;
    const int lane_in_group = lane % THREADS_PER_ROW;
    const int row0          = warp_row0 + group_in_warp * ROWS_PER_GROUP;

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q4_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    // 8 threads cooperate per block_q4_K: each loads one uint4 (16 bytes = 32 weights) of qs.
    const int blk_in_group = lane_in_group >> 3;
    const int sub          = lane_in_group & 7;
    const int chunk        = sub >> 1; // 0..3 -> 64-element chunk
    const int half_chunk   = sub & 1;  // 0..1 -> first or second 16 bytes of chunk

    const uint32_t sc_shift = static_cast<uint32_t>((chunk & 1) << 4);
    const uint32_t hi_mask  = static_cast<uint32_t>(-static_cast<int32_t>(chunk >= 2));

    float acc[ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        acc[r] = 0.0f;
    }

    if (blk_in_group < nb) {
        const block_q4_K * row_ptrs[ROWS_PER_GROUP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r) {
            const int row = (row0 + r < nrows) ? (row0 + r) : 0;
            row_ptrs[r] = x_base + row * stride_row_x + blk_in_group;
        }

        const float4 * y4_lo = reinterpret_cast<const float4 *>(
            y_base + static_cast<int64_t>(blk_in_group) * QK_K + 64 * chunk + 16 * half_chunk);

        for (int ib = blk_in_group; ; ) {
            uint4 qs_v[ROWS_PER_GROUP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                qs_v[r] = reinterpret_cast<const uint4 *>(row_ptrs[r]->qs)[sub];
            }

            [[maybe_unused]] uint4 hdr0 = make_uint4(0u, 0u, 0u, 0u);
            if constexpr (ROWS_PER_GROUP == 1) {
                // Issue header load alongside qs_v to hide L2/DRAM latency behind dot products.
                hdr0 = *reinterpret_cast<const uint4 *>(row_ptrs[0]);
            }

            float sum_q0[ROWS_PER_GROUP];
            float sum_q1[ROWS_PER_GROUP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                sum_q0[r] = 0.0f;
                sum_q1[r] = 0.0f;
            }

            float sum_y0 = 0.0f;
            float sum_y1 = 0.0f;

            // Stream one float4 of y at a time (y_lo then y_hi) per 8 weights.
            {
                const float4 y_lo = y4_lo[0];
                sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q0[r] = q4_k_dot4_lo_bias(qs_v[r].x, y_lo, sum_q0[r]);
                }
            }
            {
                const float4 y_hi = y4_lo[8];
                sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q1[r] = q4_k_dot4_hi_bias(qs_v[r].x, y_hi, sum_q1[r]);
                }
            }
            {
                const float4 y_lo = y4_lo[1];
                sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q0[r] = q4_k_dot4_lo_bias(qs_v[r].y, y_lo, sum_q0[r]);
                }
            }
            {
                const float4 y_hi = y4_lo[9];
                sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q1[r] = q4_k_dot4_hi_bias(qs_v[r].y, y_hi, sum_q1[r]);
                }
            }
            {
                const float4 y_lo = y4_lo[2];
                sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q0[r] = q4_k_dot4_lo_bias(qs_v[r].z, y_lo, sum_q0[r]);
                }
            }
            {
                const float4 y_hi = y4_lo[10];
                sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q1[r] = q4_k_dot4_hi_bias(qs_v[r].z, y_hi, sum_q1[r]);
                }
            }
            if constexpr (ROWS_PER_GROUP == 2) {
                // 6 of 8 qs_v regs are dead; prefetch row 0 header behind word 3 dot products.
                hdr0 = *reinterpret_cast<const uint4 *>(row_ptrs[0]);
            }
            {
                const float4 y_lo = y4_lo[3];
                sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q0[r] = q4_k_dot4_lo_bias(qs_v[r].w, y_lo, sum_q0[r]);
                }
            }
            {
                const float4 y_hi = y4_lo[11];
                sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    sum_q1[r] = q4_k_dot4_hi_bias(qs_v[r].w, y_hi, sum_q1[r]);
                }
            }

            if constexpr (ROWS_PER_GROUP == 1) {
                acc[0] = q4_k_apply_hdr(hdr0, sc_shift, hi_mask, sum_q0[0], sum_q1[0], sum_y0, sum_y1, acc[0]);
            } else if constexpr (ROWS_PER_GROUP == 2) {
                const uint4 hdr1 = *reinterpret_cast<const uint4 *>(row_ptrs[1]);
                acc[0] = q4_k_apply_hdr(hdr0, sc_shift, hi_mask, sum_q0[0], sum_q1[0], sum_y0, sum_y1, acc[0]);
                acc[1] = q4_k_apply_hdr(hdr1, sc_shift, hi_mask, sum_q0[1], sum_q1[1], sum_y0, sum_y1, acc[1]);
            } else {
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    const uint4 hdr = *reinterpret_cast<const uint4 *>(row_ptrs[r]);
                    acc[r] = q4_k_apply_hdr(hdr, sc_shift, hi_mask, sum_q0[r], sum_q1[r], sum_y0, sum_y1, acc[r]);
                }
            }

            ib += BLOCKS_PER_ITER;
            if (ib >= nb) {
                break;
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                row_ptrs[r] += BLOCKS_PER_ITER;
            }
            y4_lo += BLOCKS_PER_ITER * (QK_K / 4);
        }
    }

    // Recompute output indices after the loop so they do not occupy registers during K-loop.
    const int lane_out = threadIdx.x;
    const int row0_out = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.y * GROUPS_PER_WARP + lane_out / THREADS_PER_ROW) * ROWS_PER_GROUP;
    const int lig_out  = lane_out % THREADS_PER_ROW;
    float * dst_base   = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        const float sum = warp_reduce_sum<THREADS_PER_ROW>(acc[r]);
        if (lig_out == 0 && row0_out + r < nrows) {
            dst_base[row0_out + r] = sum;
        }
    }
}

static __device__ __forceinline__ void q4_k_dot8_bias_0(
        const uint32_t u, const float4 y_lo, const float4 y_hi, float & sx, float & sy) {
    // Reuse a single u_ff register across low then high half to avoid extra register pressure.
    uint32_t u_ff;
    asm("or.b32 %0, %1, 0xffff0000;" : "=r"(u_ff) : "r"(u));
    sx = __fadd_rn(__uint_as_float(u_ff & 0x4B00000Fu), -8388608.0f) * y_lo.x;
    sy = __fadd_rn(__uint_as_float(u_ff & 0x490000F0u), -524288.0f)  * y_hi.x;
    sx = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x47000F00u), -32768.0f),   y_lo.y, sx);
    sy = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x4500F000u), -2048.0f),    y_hi.y, sy);
    asm("or.b32 %0, %1, 0xffff0000;" : "=r"(u_ff) : "r"(u >> 16));
    sx = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x4B00000Fu), -8388608.0f), y_lo.z, sx);
    sy = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x490000F0u), -524288.0f),  y_hi.z, sy);
    sx = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x47000F00u), -32768.0f),   y_lo.w, sx);
    sy = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x4500F000u), -2048.0f),    y_hi.w, sy);
}

static __device__ __forceinline__ void q4_k_dot8_bias_acc(
        const uint32_t u, const float4 y_lo, const float4 y_hi, float & sx, float & sy) {
    uint32_t u_ff;
    asm("or.b32 %0, %1, 0xffff0000;" : "=r"(u_ff) : "r"(u));
    sx = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x4B00000Fu), -8388608.0f), y_lo.x, sx);
    sy = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x490000F0u), -524288.0f),  y_hi.x, sy);
    sx = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x47000F00u), -32768.0f),   y_lo.y, sx);
    sy = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x4500F000u), -2048.0f),    y_hi.y, sy);
    asm("or.b32 %0, %1, 0xffff0000;" : "=r"(u_ff) : "r"(u >> 16));
    sx = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x4B00000Fu), -8388608.0f), y_lo.z, sx);
    sy = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x490000F0u), -524288.0f),  y_hi.z, sy);
    sx = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x47000F00u), -32768.0f),   y_lo.w, sx);
    sy = fmaf(__fadd_rn(__uint_as_float(u_ff & 0x4500F000u), -2048.0f),    y_hi.w, sy);
}

static __device__ __forceinline__ float q4_k_vk_apply_half(
        const float2 dm, const uint32_t sc_pair, const uint32_t m_pair,
        const float sx, const float sy, const float sum_y0, const float sum_y1, float acc) {
    const float sc0 = __fadd_rn(__uint_as_float((sc_pair & 0x003Fu) | 0x4B000000u), -8388608.0f);
    const float sc1 = __fadd_rn(__uint_as_float((sc_pair & 0x3F00u) | 0x47000000u), -32768.0f);
    const float m0  = __fadd_rn(__uint_as_float((m_pair  & 0x003Fu) | 0x4B000000u), -8388608.0f);
    const float m1  = __fadd_rn(__uint_as_float((m_pair  & 0x3F00u) | 0x47000000u), -32768.0f);
    const float smin = fmaf(sum_y0, m0, sum_y1 * m1);
    return fmaf(dm.x, fmaf(sx, sc0, sy * sc1), fmaf(-dm.y, smin, acc));
}

static __device__ __forceinline__ float q4_k_vk_apply_packed(
        const float2 dm, const uint32_t packed,
        const float sx, const float sy, const float sum_y0, const float sum_y1, float acc) {
    // packed holds sc_hi in bits 0..15 and m_hi in bits 16..31 (m0 in bits 16..21 needs no shift).
    const float sc0 = __fadd_rn(__uint_as_float((packed & 0x0000003Fu) | 0x4B000000u), -8388608.0f);
    const float sc1 = __fadd_rn(__uint_as_float((packed & 0x00003F00u) | 0x47000000u), -32768.0f);
    const float m0  = __fadd_rn(__uint_as_float((packed & 0x003F0000u) | 0x43000000u), -128.0f);
    const float m1  = __fadd_rn(__uint_as_float((packed >> 24)         | 0x4B000000u), -8388608.0f);
    const float smin = fmaf(sum_y0, m0, sum_y1 * m1);
    return fmaf(dm.x, fmaf(sx, sc0, sy * sc1), fmaf(-dm.y, smin, acc));
}

__launch_bounds__(128)
static __global__ void mul_mat_vec_q4_K_kepler_vk(
        const block_q4_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int WARPS_PER_BLOCK = 4; // 128 threads/block
    constexpr int ROWS_PER_WARP   = 4; // 16 rows/block, 4-row y reuse, zero __syncthreads()
    constexpr int ROWS_PER_BLOCK  = WARPS_PER_BLOCK * ROWS_PER_WARP;
    constexpr int BLOCKS_PER_ITER = 2; // 16 lanes/block x 2 blocks/warp

    const int first_row_in = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    if (first_row_in >= nrows) {
        return;
    }

    const int ix     = threadIdx.x >> 4; // 0 or 1 K-block within warp tile
    const int y_off0 = 64 * ((threadIdx.x >> 3) & 1) + 4 * (threadIdx.x & 7);

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q4_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    float acc[ROWS_PER_WARP] = { 0.0f, 0.0f, 0.0f, 0.0f };

    if (ix < nb) {
        const uint8_t * x_ptr   = reinterpret_cast<const uint8_t *>(x_base + static_cast<int64_t>(first_row_in) * stride_row_x + ix);
        const int64_t row_bytes = stride_row_x * static_cast<int64_t>(sizeof(block_q4_K));
        const int row_lim       = (nrows - 1 - first_row_in < 3) ? (nrows - 1 - first_row_in) : 3;
        const float4 * y4       = reinterpret_cast<const float4 *>(y_base + static_cast<int64_t>(ix) * QK_K + y_off0);

#pragma unroll 1
        for (int rem = nb - ix; ; ) {
            float2   dm_save[ROWS_PER_WARP];
            uint32_t packed_hi[ROWS_PER_WARP];
            const int itid = threadIdx.x & 15;
            const uint32_t sc_shift = static_cast<uint32_t>((threadIdx.x & 8) << 1);

            // Half 0 (chunks 0 & 1): share one pair of float4 y loads across all 4 rows,
            // load qs in pairs and 1 header at a time to keep dm_save[4] within 63 regs.
            {
                const float4 by0 = y4[0];
                const float4 by1 = y4[8];
                const float sum_y0 = (by0.x + by0.y) + (by0.z + by0.w);
                const float sum_y1 = (by1.x + by1.y) + (by1.z + by1.w);

#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int r0 = 2 * p;
                    const int r1 = 2 * p + 1;
                    const uint8_t * p0 = x_ptr + ((r0 <= row_lim) ? r0 * row_bytes : 0);
                    const uint8_t * p1 = x_ptr + ((r1 <= row_lim) ? r1 * row_bytes : 0);
                    const uint32_t qs0 = reinterpret_cast<const uint32_t *>(p0 + offsetof(block_q4_K, qs))[itid];
                    const uint32_t qs1 = reinterpret_cast<const uint32_t *>(p1 + offsetof(block_q4_K, qs))[itid];

                    {
                        const uint4 hdr = *reinterpret_cast<const uint4 *>(p0);
                        float sx, sy;
                        q4_k_dot8_bias_0(qs0, by0, by1, sx, sy);

                        half2 dm_h2;
                        memcpy(&dm_h2, &hdr.x, sizeof(dm_h2));
                        dm_save[r0] = __half22float2(dm_h2);

                        const uint32_t s0    = hdr.y >> sc_shift;
                        const uint32_t s1    = hdr.z >> sc_shift;
                        const uint32_t s2    = hdr.w >> sc_shift;
                        const uint32_t sc_hi = (s2 & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2);
                        const uint32_t m_hi  = ((s2 >> 4) & 0x0F0Fu) | ((s1 & 0xC0C0u) >> 2);
                        packed_hi[r0] = sc_hi | (m_hi << 16);

                        acc[r0] = q4_k_vk_apply_half(dm_save[r0], s0, s1, sx, sy, sum_y0, sum_y1, acc[r0]);
                    }
                    {
                        const uint4 hdr = *reinterpret_cast<const uint4 *>(p1);
                        float sx, sy;
                        q4_k_dot8_bias_0(qs1, by0, by1, sx, sy);

                        half2 dm_h2;
                        memcpy(&dm_h2, &hdr.x, sizeof(dm_h2));
                        dm_save[r1] = __half22float2(dm_h2);

                        const uint32_t s0    = hdr.y >> sc_shift;
                        const uint32_t s1    = hdr.z >> sc_shift;
                        const uint32_t s2    = hdr.w >> sc_shift;
                        const uint32_t sc_hi = (s2 & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2);
                        const uint32_t m_hi  = ((s2 >> 4) & 0x0F0Fu) | ((s1 & 0xC0C0u) >> 2);
                        packed_hi[r1] = sc_hi | (m_hi << 16);

                        acc[r1] = q4_k_vk_apply_half(dm_save[r1], s0, s1, sx, sy, sum_y0, sum_y1, acc[r1]);
                    }
                }
            }

            // Half 1 (chunks 2 & 3): reuse saved float2 dm and packed half-1 scales (zero header reloads, zero F2F).
            {
                const float4 by0 = y4[32];
                const float4 by1 = y4[40];

                uint32_t qs[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const uint8_t * pr = x_ptr + ((r <= row_lim) ? r * row_bytes : 0);
                    qs[r] = reinterpret_cast<const uint32_t *>(pr + offsetof(block_q4_K, qs))[itid + 16];
                }

                const float sum_y0 = (by0.x + by0.y) + (by0.z + by0.w);
                const float sum_y1 = (by1.x + by1.y) + (by1.z + by1.w);

#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    float sx, sy;
                    q4_k_dot8_bias_0(qs[r], by0, by1, sx, sy);
                    acc[r] = q4_k_vk_apply_packed(dm_save[r], packed_hi[r], sx, sy, sum_y0, sum_y1, acc[r]);
                }
            }

            rem -= BLOCKS_PER_ITER;
            if (rem <= 0) {
                break;
            }
            x_ptr += BLOCKS_PER_ITER * sizeof(block_q4_K);
            y4    += BLOCKS_PER_ITER * (QK_K / 4);
        }
    }

    const int lane_out  = threadIdx.x;
    const int first_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    float * dst_base    = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float sum = warp_reduce_sum<WARP_SIZE>(acc[r]);
        if (lane_out == 0 && first_row + r < nrows) {
            dst_base[first_row + r] = sum;
        }
    }
}

static __device__ __forceinline__ float warp_reduce_sum_4rows_packed(
        const float r0, const float r1, const float r2, const float r3, const int lane) {
    // Reduces 4 row sums in 6 SHFLs instead of 20; lanes 0..3 hold final sums for rows 0..3.
    const bool odd  = (lane & 1) != 0;
    const bool hi2  = (lane & 2) != 0;
    const float p01 = (odd ? r1 : r0) + __shfl_xor_sync(0xffffffffu, odd ? r0 : r1, 1);
    const float p23 = (odd ? r3 : r2) + __shfl_xor_sync(0xffffffffu, odd ? r2 : r3, 1);
    float p         = (hi2 ? p23 : p01) + __shfl_xor_sync(0xffffffffu, hi2 ? p01 : p23, 2);
    p += __shfl_xor_sync(0xffffffffu, p, 4);
    p += __shfl_xor_sync(0xffffffffu, p, 8);
    p += __shfl_xor_sync(0xffffffffu, p, 16);
    return p;
}

static __device__ __forceinline__ void q4_k_decode_chunk_hdr(
        const uint4 hdr, const int c_hdr,
        float & d_sc0, float & d_sc1, float & d_m0, float & d_m1) {
    half2 dm_h2;
    memcpy(&dm_h2, &hdr.x, sizeof(dm_h2));
    const float2 dm = __half22float2(dm_h2);

    const uint32_t sc_shift = static_cast<uint32_t>((c_hdr & 1) << 4);
    const uint32_t s0 = hdr.y >> sc_shift;
    const uint32_t s1 = hdr.z >> sc_shift;
    const uint32_t s2 = hdr.w >> sc_shift;

    const bool hi_half = (c_hdr >= 2);
    const uint32_t sc_pair = hi_half ? ((s2 & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2)) : (s0 & 0x3F3Fu);
    const uint32_t m_pair  = hi_half ? (((s2 >> 4) & 0x0F0Fu) | ((s1 & 0xC0C0u) >> 2)) : (s1 & 0x3F3Fu);

    uint32_t sc_ff, m_ff;
    asm("or.b32 %0, %1, 0xff000000;" : "=r"(sc_ff) : "r"(sc_pair));
    asm("or.b32 %0, %1, 0xff000000;" : "=r"(m_ff)  : "r"(m_pair));

    const float sc0 = __fadd_rn(__uint_as_float(sc_ff & 0x4B00003Fu), -8388608.0f);
    const float sc1 = __fadd_rn(__uint_as_float(sc_ff & 0x47003F00u), -32768.0f);
    const float m0  = __fadd_rn(__uint_as_float(m_ff  & 0x4B00003Fu), -8388608.0f);
    const float m1  = __fadd_rn(__uint_as_float(m_ff  & 0x47003F00u), -32768.0f);

    d_sc0 = dm.x * sc0;
    d_sc1 = dm.x * sc1;
    d_m0  = -dm.y * m0;
    d_m1  = -dm.y * m1;
}

template <int WARPS_PER_BLOCK, bool EVEN_NB, bool FULL4, bool SHARED_Y>
__launch_bounds__(WARP_SIZE * WARPS_PER_BLOCK)
static __global__ void mul_mat_vec_q4_K_kepler_vk2(
        const block_q4_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int ROWS_PER_WARP   = 4;
    constexpr int ROWS_PER_BLOCK  = WARPS_PER_BLOCK * ROWS_PER_WARP;
    constexpr int BLOCKS_PER_ITER = 2;

    const int first_row_in = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    if (!SHARED_Y && first_row_in >= nrows) {
        return;
    }

    const int ix_init = threadIdx.x >> 4;
    const int y_off0  = 64 * ((threadIdx.x >> 3) & 1) + 4 * (threadIdx.x & 7);

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q4_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    extern __shared__ float shared[];
    float * shared_y = shared;
    float * shared_y_sums = shared + nb * QK_K;
    if (SHARED_Y) {
        const int lane = threadIdx.x;
        const int warp_base = threadIdx.y * WARP_SIZE;
        const int count = nb * QK_K;
        for (int i = warp_base + lane; i < count; i += WARP_SIZE * WARPS_PER_BLOCK) {
            const float value = y_base[i];
            shared_y[i] = value;
            float sum = value;
            sum += __shfl_down_sync(0xffffffffu, sum, 16);
            sum += __shfl_down_sync(0xffffffffu, sum, 8);
            sum += __shfl_down_sync(0xffffffffu, sum, 4);
            sum += __shfl_down_sync(0xffffffffu, sum, 2);
            sum += __shfl_down_sync(0xffffffffu, sum, 1);
            if ((lane & 31) == 0) {
                shared_y_sums[i / 32] = sum;
            }
        }
        __syncthreads();
    }

    const uint8_t * x_ptr   = reinterpret_cast<const uint8_t *>(x_base + static_cast<int64_t>(first_row_in) * stride_row_x + ix_init);
    const int64_t row_bytes = stride_row_x * static_cast<int64_t>(sizeof(block_q4_K));
    const int row_lim       = FULL4 ? 3 : ((nrows - 1 - first_row_in < 3) ? (nrows - 1 - first_row_in) : 3);
    const float4 * y4       = reinterpret_cast<const float4 *>((SHARED_Y ? shared_y : y_base) + static_cast<int64_t>(ix_init) * QK_K + y_off0);

    float acc[ROWS_PER_WARP] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float mt_acc = 0.0f;

#pragma unroll 1
    for (int rem = nb; ; ) {
        const int ix    = threadIdx.x >> 4;
        const int block_index = nb - rem + ix;
        const int itid  = threadIdx.x & 15;
        const int c0    = (itid >> 3) & 1;
        const int r_hdr = itid >> 2;
        const int c_hdr = itid & 3;
        const bool valid = (first_row_in < nrows) && (EVEN_NB ? true : (ix < rem));

        // Stepped row pointers avoid 64-bit IMUL and row_lim checks on FULL4 shapes.
        const uint8_t * p_hdr = FULL4 ? (x_ptr + r_hdr * row_bytes) : (x_ptr + ((r_hdr <= row_lim) ? r_hdr * row_bytes : 0));
        const uint4 hdr = valid ? *reinterpret_cast<const uint4 *>(p_hdr) : make_uint4(0u, 0u, 0u, 0u);

        // Pre-issue Half 0 y and row 0/1 qs loads alongside hdr for DRAM MLP.
        const float4 by0_h0 = valid ? y4[0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 by1_h0 = valid ? y4[8] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const uint8_t * x_qs = x_ptr + offsetof(block_q4_K, qs) + 4 * itid;
        const uint8_t * q0   = x_qs;
        const uint8_t * q1   = FULL4 ? (q0 + row_bytes) : (x_qs + ((1 <= row_lim) ? row_bytes : 0));
        uint32_t qs_a = valid ? *reinterpret_cast<const uint32_t *>(q0) : 0u;
        uint32_t qs_b = valid ? *reinterpret_cast<const uint32_t *>(q1) : 0u;

        float d_sc0, d_sc1, d_m0, d_m1;
        q4_k_decode_chunk_hdr(hdr, c_hdr, d_sc0, d_sc1, d_m0, d_m1);

        // Half 0 (chunks 0 & 1)
        {
            float my_sy0, my_sy1;
            if (SHARED_Y) {
                my_sy0 = valid ? shared_y_sums[block_index * 8 + c_hdr * 2] : 0.0f;
                my_sy1 = valid ? shared_y_sums[block_index * 8 + c_hdr * 2 + 1] : 0.0f;
            } else {
                const float sy0 = (by0_h0.x + by0_h0.y) + (by0_h0.z + by0_h0.w);
                const float sy1 = (by1_h0.x + by1_h0.y) + (by1_h0.z + by1_h0.w);
                const bool odd_l = (itid & 1) != 0;
                float sp = (odd_l ? sy1 : sy0) + __shfl_xor_sync(0xffffffffu, odd_l ? sy0 : sy1, 1, 8);
                sp += __shfl_xor_sync(0xffffffffu, sp, 2, 8);
                sp += __shfl_xor_sync(0xffffffffu, sp, 4, 8);
                const int sy_src = (c_hdr & 1) << 3;
                my_sy0 = __shfl_sync(0xffffffffu, sp, sy_src, 16);
                my_sy1 = __shfl_sync(0xffffffffu, sp, sy_src | 1, 16);
            }
            if (valid && c_hdr < 2) {
                mt_acc = fmaf(my_sy0, d_m0, fmaf(my_sy1, d_m1, mt_acc));
            }

            const uint8_t * q2 = FULL4 ? (q1 + row_bytes) : (x_qs + ((2 <= row_lim) ? 2 * row_bytes : 0));
            const uint8_t * q3 = FULL4 ? (q2 + row_bytes) : (x_qs + ((3 <= row_lim) ? 3 * row_bytes : 0));
            const uint32_t qs_c = valid ? *reinterpret_cast<const uint32_t *>(q2) : 0u;
            const uint32_t qs_d = valid ? *reinterpret_cast<const uint32_t *>(q3) : 0u;

            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 0 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 0 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_a, by0_h0, by1_h0, sx, sy);
                if (valid) { acc[0] = fmaf(sx, sc0, fmaf(sy, sc1, acc[0])); }
            }
            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 4 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 4 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_b, by0_h0, by1_h0, sx, sy);
                if (valid) { acc[1] = fmaf(sx, sc0, fmaf(sy, sc1, acc[1])); }
            }
            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 8 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 8 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_c, by0_h0, by1_h0, sx, sy);
                if (valid) { acc[2] = fmaf(sx, sc0, fmaf(sy, sc1, acc[2])); }
            }
            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 12 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 12 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_d, by0_h0, by1_h0, sx, sy);
                if (valid) { acc[3] = fmaf(sx, sc0, fmaf(sy, sc1, acc[3])); }
            }
        }

        // Half 1 (chunks 2 & 3, constant +64 byte offset on qs)
        {
            const float4 by0 = valid ? y4[32] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 by1 = valid ? y4[40] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            qs_a = valid ? *reinterpret_cast<const uint32_t *>(q0 + 64) : 0u;
            qs_b = valid ? *reinterpret_cast<const uint32_t *>(q1 + 64) : 0u;

            float my_sy0, my_sy1;
            if (SHARED_Y) {
                my_sy0 = valid ? shared_y_sums[block_index * 8 + c_hdr * 2] : 0.0f;
                my_sy1 = valid ? shared_y_sums[block_index * 8 + c_hdr * 2 + 1] : 0.0f;
            } else {
                const float sy0 = (by0.x + by0.y) + (by0.z + by0.w);
                const float sy1 = (by1.x + by1.y) + (by1.z + by1.w);
                const bool odd_l = (itid & 1) != 0;
                float sp = (odd_l ? sy1 : sy0) + __shfl_xor_sync(0xffffffffu, odd_l ? sy0 : sy1, 1, 8);
                sp += __shfl_xor_sync(0xffffffffu, sp, 2, 8);
                sp += __shfl_xor_sync(0xffffffffu, sp, 4, 8);
                const int sy_src = (c_hdr & 1) << 3;
                my_sy0 = __shfl_sync(0xffffffffu, sp, sy_src, 16);
                my_sy1 = __shfl_sync(0xffffffffu, sp, sy_src | 1, 16);
            }
            if (valid && c_hdr >= 2) {
                mt_acc = fmaf(my_sy0, d_m0, fmaf(my_sy1, d_m1, mt_acc));
            }

            const uint8_t * q2 = FULL4 ? (q1 + row_bytes) : (x_qs + ((2 <= row_lim) ? 2 * row_bytes : 0));
            const uint8_t * q3 = FULL4 ? (q2 + row_bytes) : (x_qs + ((3 <= row_lim) ? 3 * row_bytes : 0));
            const uint32_t qs_c = valid ? *reinterpret_cast<const uint32_t *>(q2 + 64) : 0u;
            const uint32_t qs_d = valid ? *reinterpret_cast<const uint32_t *>(q3 + 64) : 0u;

            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 2 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 2 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_a, by0, by1, sx, sy);
                if (valid) { acc[0] = fmaf(sx, sc0, fmaf(sy, sc1, acc[0])); }
            }
            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 6 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 6 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_b, by0, by1, sx, sy);
                if (valid) { acc[1] = fmaf(sx, sc0, fmaf(sy, sc1, acc[1])); }
            }
            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 10 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 10 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_c, by0, by1, sx, sy);
                if (valid) { acc[2] = fmaf(sx, sc0, fmaf(sy, sc1, acc[2])); }
            }
            {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc0, 14 + c0, 16);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc1, 14 + c0, 16);
                float sx, sy;
                q4_k_dot8_bias_0(qs_d, by0, by1, sx, sy);
                if (valid) { acc[3] = fmaf(sx, sc0, fmaf(sy, sc1, acc[3])); }
            }
        }

        rem -= BLOCKS_PER_ITER;
        if (rem <= 0) {
            break;
        }
        x_ptr += BLOCKS_PER_ITER * sizeof(block_q4_K);
        y4    += BLOCKS_PER_ITER * (QK_K / 4);
    }

    const int lane_out  = threadIdx.x;
    const int r_hdr_out = (lane_out >> 2) & 3;
    if (r_hdr_out == 0) acc[0] += mt_acc;
    if (r_hdr_out == 1) acc[1] += mt_acc;
    if (r_hdr_out == 2) acc[2] += mt_acc;
    if (r_hdr_out == 3) acc[3] += mt_acc;

    const int first_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    const float sum = warp_reduce_sum_4rows_packed(acc[0], acc[1], acc[2], acc[3], lane_out);
    if (lane_out < ROWS_PER_WARP && first_row + lane_out < nrows) {
        float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
        dst_base[first_row + lane_out] = sum;
    }
}

static __device__ __forceinline__ void q4_k_decode_sub_hdr(
        const uint4 hdr, const int sub, float & d_sc, float & d_m) {
    half2 dm_h2;
    memcpy(&dm_h2, &hdr.x, sizeof(dm_h2));
    const float2 dm = __half22float2(dm_h2);

    const uint32_t sh = static_cast<uint32_t>((sub & 3) << 3);
    const uint32_t b0 = hdr.y >> sh;
    const uint32_t b1 = hdr.z >> sh;
    const uint32_t b2 = hdr.w >> sh;
    const bool is_hi  = (sub >= 4);
    const uint32_t sc_u = is_hi ? ((b2 & 0x0Fu) | ((b0 & 0xC0u) >> 2)) : (b0 & 0x3Fu);
    const uint32_t m_u  = is_hi ? (((b2 >> 4) & 0x0Fu) | ((b1 & 0xC0u) >> 2)) : (b1 & 0x3Fu);

    const float sc_f = __fadd_rn(__uint_as_float(sc_u | 0x4B000000u), -8388608.0f);
    const float m_f  = __fadd_rn(__uint_as_float(m_u  | 0x4B000000u), -8388608.0f);
    d_sc = dm.x * sc_f;
    d_m  = -dm.y * m_f;
}

__launch_bounds__(128)
static __global__ void mul_mat_vec_q4_K_kepler_c8(
        const block_q4_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int ROWS_PER_WARP   = 4;
    constexpr int ROWS_PER_BLOCK  = WARPS_PER_BLOCK * ROWS_PER_WARP;

    const int first_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    if (first_row >= nrows) {
        return;
    }

    const int lane = threadIdx.x;
    const int ib   = lane >> 3; // 0..3 super-blocks per warp
    const int sub  = lane & 7;  // 0..7 lanes per super-block (32 bytes coalesced per chunk)

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q4_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    const uint8_t * x_ptr   = reinterpret_cast<const uint8_t *>(x_base + static_cast<int64_t>(first_row) * stride_row_x + ib);
    const int64_t row_bytes = stride_row_x * static_cast<int64_t>(sizeof(block_q4_K));
    const int row_lim       = (nrows - 1 - first_row < 3) ? (nrows - 1 - first_row) : 3;

    const float4 * y4 = reinterpret_cast<const float4 *>(y_base + static_cast<int64_t>(ib) * QK_K + 4 * sub);

    float acc[ROWS_PER_WARP] = { 0.0f, 0.0f, 0.0f, 0.0f };

#pragma unroll 1
    for (int ib0 = 0; ib0 < nb; ib0 += 4) {
        const bool valid = (ib0 + ib < nb);

        float d_sc[ROWS_PER_WARP];
        float d_m[ROWS_PER_WARP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const uint8_t * pr = x_ptr + ((r <= row_lim) ? r * row_bytes : 0);
            const uint4 hdr = valid ? *reinterpret_cast<const uint4 *>(pr) : make_uint4(0u, 0u, 0u, 0u);
            q4_k_decode_sub_hdr(hdr, sub, d_sc[r], d_m[r]);
        }

        float my_sy = 0.0f;

#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
            uint32_t qs[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const uint8_t * pr = x_ptr + ((r <= row_lim) ? r * row_bytes : 0);
                qs[r] = valid ? reinterpret_cast<const uint32_t *>(pr + offsetof(block_q4_K, qs))[8 * c + sub] : 0u;
            }

            const float4 by0 = valid ? y4[16 * c]     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 by1 = valid ? y4[16 * c + 8] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            const float sy0_32 = warp_reduce_sum<8>((by0.x + by0.y) + (by0.z + by0.w));
            const float sy1_32 = warp_reduce_sum<8>((by1.x + by1.y) + (by1.z + by1.w));
            if (sub == 2 * c) {
                my_sy = sy0_32;
            }
            if (sub == 2 * c + 1) {
                my_sy = sy1_32;
            }

#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const float sc0 = __shfl_sync(0xffffffffu, d_sc[r], 2 * c,     8);
                const float sc1 = __shfl_sync(0xffffffffu, d_sc[r], 2 * c + 1, 8);
                float sx, sy;
                q4_k_dot8_bias_0(qs[r], by0, by1, sx, sy);
                if (valid) {
                    acc[r] = fmaf(sx, sc0, fmaf(sy, sc1, acc[r]));
                }
            }
        }

        if (valid) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                acc[r] = fmaf(d_m[r], my_sy, acc[r]);
            }
        }

        if (ib0 + 4 >= nb) {
            break;
        }
        x_ptr += 4 * sizeof(block_q4_K);
        y4    += 4 * (QK_K / 4);
    }

    const float sum = warp_reduce_sum_4rows_packed(acc[0], acc[1], acc[2], acc[3], lane);
    if (lane < ROWS_PER_WARP && first_row + lane < nrows) {
        float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
        dst_base[first_row + lane] = sum;
    }
}

template <int ROWS_PER_WARP, int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
static __global__ void mul_mat_vec_q4_K_kepler_swiz(
        const block_q4_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    static_assert(WARPS_PER_BLOCK == 4 || WARPS_PER_BLOCK == 8, "WARPS_PER_BLOCK must be 4 or 8");
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    constexpr int NTHREADS       = WARPS_PER_BLOCK * WARP_SIZE;

    const int block_row0 = blockIdx.x * ROWS_PER_BLOCK;
    if (block_row0 >= nrows) {
        return;
    }

    // Bank-conflict-free swizzled y tile (4 super-blocks = 1024 floats = 4 KB) + 32 sub-block sums.
    __shared__ float smem_y[8][4][WARP_SIZE];
    __shared__ float smem_sum_y[WARP_SIZE];

    const int lane      = threadIdx.x;
    const int tid       = threadIdx.y * WARP_SIZE + lane;
    const int first_row = block_row0 + threadIdx.y * ROWS_PER_WARP;

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q4_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float4     * y4_base = reinterpret_cast<const float4 *>(vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y);

    // 8 lanes/super-block x 4 super-blocks = 32 lanes; each lane also decodes 1 of the 32 sub-blocks.
    const int ib    = lane >> 3;
    const int sub   = lane & 7;
    const int chunk = sub >> 1;

    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        acc[r] = 0.0f;
    }

#pragma unroll 1
    for (int ib0 = 0; ib0 < nb; ib0 += 4) {
#pragma unroll
        for (int t = tid; t < 256; t += NTHREADS) {
            const int ib_t = t >> 6;
            const float4 yv = (ib0 + ib_t < nb)
                ? y4_base[static_cast<int64_t>(ib0 + ib_t) * 64 + (t & 63)]
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            const int c_t  = (t >> 4) & 3;
            const int lh_t = (t >> 3) & 1;
            const int hc_t = (t >> 2) & 1;
            const int k_t  = t & 3;
            const int dst_lane = (ib_t << 3) | (c_t << 1) | hc_t;
            const int step     = (k_t << 1) | lh_t;

            smem_y[step][0][dst_lane] = yv.x;
            smem_y[step][1][dst_lane] = yv.y;
            smem_y[step][2][dst_lane] = yv.z;
            smem_y[step][3][dst_lane] = yv.w;

            float sy = (yv.x + yv.y) + (yv.z + yv.w);
            sy = warp_reduce_sum<8>(sy);
            if ((t & 7) == 0) {
                smem_sum_y[t >> 3] = sy;
            }
        }
        __syncthreads();

        const bool lane_valid = (ib0 + ib < nb);
        const float my_sum_y  = smem_sum_y[lane];

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int row = (first_row + r < nrows) ? (first_row + r) : 0;
            const block_q4_K * blk = x_base + static_cast<int64_t>(row) * stride_row_x + (ib0 + ib);

            const uint4 qv  = lane_valid ? reinterpret_cast<const uint4 *>(blk->qs)[sub] : make_uint4(0u, 0u, 0u, 0u);
            const uint4 hdr = lane_valid ? *reinterpret_cast<const uint4 *>(blk)         : make_uint4(0u, 0u, 0u, 0u);

            float sx, sy;
            {
                const float4 y_lo = make_float4(smem_y[0][0][lane], smem_y[0][1][lane], smem_y[0][2][lane], smem_y[0][3][lane]);
                const float4 y_hi = make_float4(smem_y[1][0][lane], smem_y[1][1][lane], smem_y[1][2][lane], smem_y[1][3][lane]);
                q4_k_dot8_bias_0(qv.x, y_lo, y_hi, sx, sy);
            }
            {
                const float4 y_lo = make_float4(smem_y[2][0][lane], smem_y[2][1][lane], smem_y[2][2][lane], smem_y[2][3][lane]);
                const float4 y_hi = make_float4(smem_y[3][0][lane], smem_y[3][1][lane], smem_y[3][2][lane], smem_y[3][3][lane]);
                q4_k_dot8_bias_acc(qv.y, y_lo, y_hi, sx, sy);
            }
            {
                const float4 y_lo = make_float4(smem_y[4][0][lane], smem_y[4][1][lane], smem_y[4][2][lane], smem_y[4][3][lane]);
                const float4 y_hi = make_float4(smem_y[5][0][lane], smem_y[5][1][lane], smem_y[5][2][lane], smem_y[5][3][lane]);
                q4_k_dot8_bias_acc(qv.z, y_lo, y_hi, sx, sy);
            }
            {
                const float4 y_lo = make_float4(smem_y[6][0][lane], smem_y[6][1][lane], smem_y[6][2][lane], smem_y[6][3][lane]);
                const float4 y_hi = make_float4(smem_y[7][0][lane], smem_y[7][1][lane], smem_y[7][2][lane], smem_y[7][3][lane]);
                q4_k_dot8_bias_acc(qv.w, y_lo, y_hi, sx, sy);
            }

            half2 dm_h2;
            memcpy(&dm_h2, &hdr.x, sizeof(dm_h2));
            const float2 dm = __half22float2(dm_h2);

            const uint32_t sh = static_cast<uint32_t>((sub & 3) << 3);
            const uint32_t b0 = hdr.y >> sh;
            const uint32_t b1 = hdr.z >> sh;
            const uint32_t b2 = hdr.w >> sh;
            const bool is_hi  = (sub >= 4);
            const uint32_t sc_u = is_hi ? ((b2 & 0x0Fu) | ((b0 & 0xC0u) >> 2)) : (b0 & 0x3Fu);
            const uint32_t m_u  = is_hi ? (((b2 >> 4) & 0x0Fu) | ((b1 & 0xC0u) >> 2)) : (b1 & 0x3Fu);

            const float sc_f = __fadd_rn(__uint_as_float(sc_u | 0x4B000000u), -8388608.0f);
            const float m_f  = __fadd_rn(__uint_as_float(m_u  | 0x4B000000u), -8388608.0f);
            const float d_sc = dm.x * sc_f;

            const float d_sc0 = __shfl_sync(0xffffffffu, d_sc, 2 * chunk,     8);
            const float d_sc1 = __shfl_sync(0xffffffffu, d_sc, 2 * chunk + 1, 8);
            if (lane_valid) {
                acc[r] = fmaf(sx, d_sc0, fmaf(sy, d_sc1, fmaf(-(dm.y * m_f), my_sum_y, acc[r])));
            }
        }

        if (ib0 + 4 < nb) {
            __syncthreads();
        }
    }

    float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float sum = warp_reduce_sum<WARP_SIZE>(acc[r]);
        if (lane == 0 && first_row + r < nrows) {
            dst_base[first_row + r] = sum;
        }
    }
}

static __device__ __forceinline__ float dot4_u6_bias(const uint32_t q, const float4 y) {
    // q holds 4 unsigned 6-bit quants in bytes 0..3; set top byte to 0xFF once so bytes 0..2 use 1 AND each.
    float sum = __fadd_rn(__uint_as_float((q >> 24) | 0x4B000000u), -8388640.0f) * y.w;
    uint32_t q_ff;
    asm("or.b32 %0, %1, 0xff000000;" : "=r"(q_ff) : "r"(q));
    sum = fmaf(__fadd_rn(__uint_as_float(q_ff & 0x433F0000u), -160.0f),     y.z, sum);
    sum = fmaf(__fadd_rn(__uint_as_float(q_ff & 0x47003F00u), -32800.0f),   y.y, sum);
    sum = fmaf(__fadd_rn(__uint_as_float(q_ff & 0x4B00003Fu), -8388640.0f), y.x, sum);
    return sum;
}

static __device__ __forceinline__ float i8_to_f32_bias(const int8_t v) {
    const uint32_t u = static_cast<uint32_t>(static_cast<uint8_t>(v)) ^ 0x80u;
    return __fadd_rn(__uint_as_float(u | 0x4B000000u), -8388736.0f);
}

static __device__ __forceinline__ uint32_t load_u32_q6_k(
        const uint8_t * ptr, const int i4, const bool valid) {
    // block_q6_K is 210 bytes (2-byte aligned, not 4-byte aligned on odd blocks).
    // For any 32-byte slice (ql0, ql1, qh0), ptr + 2 + 4*i4 stays inside block_q6_K and is 4-byte aligned.
    const uint32_t mis    = valid ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) & 2u) : 0u;
    const uint32_t w_hi   = valid ? reinterpret_cast<const uint32_t *>(ptr + mis)[i4] : 0u;
    const uint32_t w_prev = __shfl_up_sync(0xffffffffu, w_hi, 1, 8);
    const uint32_t w_lo   = (valid && mis != 0u && i4 == 0)
        ? (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(ptr)) << 16)
        : w_prev;
    return mis ? __byte_perm(w_lo, w_hi, 0x5432u) : w_hi;
}

template <int THREADS_PER_ROW, int ROWS_PER_GROUP, int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
static __global__ void mul_mat_vec_q6_K_kepler(
        const block_q6_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    static_assert(THREADS_PER_ROW >= 16 && THREADS_PER_ROW <= WARP_SIZE && (THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0,
                  "THREADS_PER_ROW must be power of 2 in [16, 32]");
    static_assert(ROWS_PER_GROUP >= 1, "ROWS_PER_GROUP must be >= 1");

    constexpr int GROUPS_PER_WARP  = WARP_SIZE / THREADS_PER_ROW;
    constexpr int GROUPS_PER_BLOCK = WARPS_PER_BLOCK * GROUPS_PER_WARP;
    constexpr int ROWS_PER_BLOCK   = GROUPS_PER_BLOCK * ROWS_PER_GROUP;
    constexpr int BLOCKS_PER_ITER  = THREADS_PER_ROW / 16;

    const int warp_row0 = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * GROUPS_PER_WARP * ROWS_PER_GROUP;
    if (warp_row0 >= nrows) {
        return;
    }

    const int lane          = threadIdx.x;
    const int group_in_warp = lane / THREADS_PER_ROW;
    const int lane_in_group = lane % THREADS_PER_ROW;
    const int row0          = warp_row0 + group_in_warp * ROWS_PER_GROUP;

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q6_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    // 16 threads cooperate per block_q6_K: each handles 16 weights (4 quarters of 4 elements).
    const int blk_in_group = lane_in_group >> 4;
    const int sub          = lane_in_group & 15;
    const int h            = sub >> 3; // 0: elements 0..127, 1: elements 128..255
    const int i4           = sub & 7;  // 0..7 -> l0 = 4 * i4
    const int s_off        = 8 * h + (i4 >> 2);

    const bool lane_has_blk = (blk_in_group < nb);
    const block_q6_K * row_ptrs[ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        const int row = (row0 + r < nrows) ? (row0 + r) : 0;
        row_ptrs[r] = lane_has_blk ? (x_base + row * stride_row_x + blk_in_group) : nullptr;
    }

    const float4 * y4 = lane_has_blk
        ? reinterpret_cast<const float4 *>(y_base + static_cast<int64_t>(blk_in_group) * QK_K + 128 * h + 4 * i4)
        : nullptr;

    float acc[ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        acc[r] = 0.0f;
    }

    // Uniform loop bound across the warp so all lanes participate in __shfl_up_sync inside load_u32_q6_k.
    for (int ib_base = 0; ; ) {
        const bool valid = (ib_base + blk_in_group) < nb;

        uint32_t ql0[ROWS_PER_GROUP];
        uint32_t ql1[ROWS_PER_GROUP];
        uint32_t qh0[ROWS_PER_GROUP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r) {
            const block_q6_K * blk = valid ? row_ptrs[r] : nullptr;
            ql0[r] = load_u32_q6_k(valid ? (blk->ql + 64 * h)      : nullptr, i4, valid);
            ql1[r] = load_u32_q6_k(valid ? (blk->ql + 64 * h + 32) : nullptr, i4, valid);
            qh0[r] = load_u32_q6_k(valid ? (blk->qh + 32 * h)      : nullptr, i4, valid);
        }

        if (valid) {
            float blk_sum[ROWS_PER_GROUP];
            {
                const float4 y0 = y4[0];
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    const uint32_t q1 = (ql0[r] & 0x0F0F0F0Fu) | ((qh0[r] << 4) & 0x30303030u);
                    const float sc0 = i8_to_f32_bias(row_ptrs[r]->scales[s_off + 0]);
                    blk_sum[r] = dot4_u6_bias(q1, y0) * sc0;
                }
            }
            {
                const float4 y1 = y4[8];
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    const uint32_t q2 = (ql1[r] & 0x0F0F0F0Fu) | ((qh0[r] << 2) & 0x30303030u);
                    const float sc1 = i8_to_f32_bias(row_ptrs[r]->scales[s_off + 2]);
                    blk_sum[r] = fmaf(dot4_u6_bias(q2, y1), sc1, blk_sum[r]);
                }
            }
            {
                const float4 y2 = y4[16];
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    const uint32_t q3 = ((ql0[r] >> 4) & 0x0F0F0F0Fu) | (qh0[r] & 0x30303030u);
                    const float sc2 = i8_to_f32_bias(row_ptrs[r]->scales[s_off + 4]);
                    blk_sum[r] = fmaf(dot4_u6_bias(q3, y2), sc2, blk_sum[r]);
                }
            }
            {
                const float4 y3 = y4[24];
#pragma unroll
                for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                    const uint32_t q4 = ((ql1[r] >> 4) & 0x0F0F0F0Fu) | ((qh0[r] >> 2) & 0x30303030u);
                    const float sc3 = i8_to_f32_bias(row_ptrs[r]->scales[s_off + 6]);
                    blk_sum[r] = fmaf(dot4_u6_bias(q4, y3), sc3, blk_sum[r]);
                }
            }

#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                const float d = __half2float(row_ptrs[r]->d);
                acc[r] = fmaf(d, blk_sum[r], acc[r]);
            }
        }

        ib_base += BLOCKS_PER_ITER;
        if (ib_base >= nb) {
            break;
        }
        if (ib_base + blk_in_group < nb) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                row_ptrs[r] += BLOCKS_PER_ITER;
            }
            y4 += BLOCKS_PER_ITER * (QK_K / 4);
        }
    }

    const int lane_out = threadIdx.x;
    const int row0_out = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.y * GROUPS_PER_WARP + lane_out / THREADS_PER_ROW) * ROWS_PER_GROUP;
    const int lig_out  = lane_out % THREADS_PER_ROW;
    float * dst_base   = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        const float sum = warp_reduce_sum<THREADS_PER_ROW>(acc[r]);
        if (lig_out == 0 && row0_out + r < nrows) {
            dst_base[row0_out + r] = sum;
        }
    }
}

static __device__ __forceinline__ float dot4_u6_bias_tree(const uint32_t q, const float4 y) {
    const float q3 = __fadd_rn(__uint_as_float((q >> 24)         | 0x4B000000u), -8388640.0f);
    const float q2 = __fadd_rn(__uint_as_float((q & 0x003F0000u) | 0x43000000u), -160.0f);
    const float q1 = __fadd_rn(__uint_as_float((q & 0x00003F00u) | 0x47000000u), -32800.0f);
    const float q0 = __fadd_rn(__uint_as_float((q & 0x0000003Fu) | 0x4B000000u), -8388640.0f);
    return fmaf(q0, y.x, q1 * y.y) + fmaf(q2, y.z, q3 * y.w);
}

static __device__ __forceinline__ uint32_t load_u32_u16x2(const uint16_t * p) {
    const uint32_t lo = static_cast<uint32_t>(p[0]);
    const uint32_t hi = static_cast<uint32_t>(p[1]);
    return lo | (hi << 16);
}

template <int BYTE_IDX>
static __device__ __forceinline__ float unpack_u8_bias_f32(const uint32_t packed_u8_xor80) {
    if constexpr (BYTE_IDX == 0) {
        return __fadd_rn(__uint_as_float((packed_u8_xor80 & 0x000000FFu) | 0x4B000000u), -8388736.0f);
    } else if constexpr (BYTE_IDX == 1) {
        return __fadd_rn(__uint_as_float((packed_u8_xor80 & 0x0000FF00u) | 0x47000000u), -32896.0f);
    } else if constexpr (BYTE_IDX == 2) {
        return __fadd_rn(__uint_as_float(((packed_u8_xor80 >> 16) & 0x000000FFu) | 0x4B000000u), -8388736.0f);
    } else {
        return __fadd_rn(__uint_as_float((packed_u8_xor80 >> 24) | 0x4B000000u), -8388736.0f);
    }
}

static __device__ __forceinline__ uint32_t q6_k_stitch_uniform_mis(
        const uint32_t w_hi, const uint8_t * ptr, const int i4, const uint32_t mis, const bool valid) {
    const uint32_t w_prev = __shfl_up_sync(0xffffffffu, w_hi, 1, 8);
    const uint32_t w_lo   = (valid && mis != 0u && i4 == 0)
        ? (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(ptr)) << 16)
        : w_prev;
    return mis ? __byte_perm(w_lo, w_hi, 0x5432u) : w_hi;
}

__launch_bounds__(128)
static __global__ void mul_mat_vec_q6_K_kepler_vk(
        const block_q6_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int WARPS_PER_BLOCK = 4; // 128 threads/block
    constexpr int ROWS_PER_WARP   = 4; // 16 rows/block, 4-row y reuse, overlapped loads, shuffled scales
    constexpr int ROWS_PER_BLOCK  = WARPS_PER_BLOCK * ROWS_PER_WARP;
    constexpr int BLOCKS_PER_ITER = 2; // 16 lanes/block x 2 blocks/warp

    const int first_row_in = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    if (first_row_in >= nrows) {
        return;
    }

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q6_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    const int ix_init       = threadIdx.x >> 4;
    const bool lane_has_blk = (ix_init < nb);
    // A lane group with no block (nb == 1, second 16-lane group) still forms addresses below (stitch
    // arguments, the misalignment bit) even though every load is masked by `valid`. Point it at block 0
    // of its own row and at the start of y instead of nullptr: arithmetic on a null pointer is undefined
    // behaviour, while block 0 is a real 210-byte object and y holds at least QK_K floats.
    const int ix_safe       = lane_has_blk ? ix_init : 0;
    const uint8_t * row_ptrs[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = (first_row_in + r < nrows) ? (first_row_in + r) : 0;
        row_ptrs[r] = reinterpret_cast<const uint8_t *>(x_base + static_cast<int64_t>(row) * stride_row_x + ix_safe);
    }
    const float4 * y4 = reinterpret_cast<const float4 *>(
        y_base + static_cast<int64_t>(ix_safe) * QK_K + 128 * ((threadIdx.x >> 3) & 1) + 4 * (threadIdx.x & 7));

    float acc[ROWS_PER_WARP] = { 0.0f, 0.0f, 0.0f, 0.0f };

#pragma unroll 1
    for (int rem = nb; ; ) {
        const int ix     = threadIdx.x >> 4;
        const int itid   = threadIdx.x & 15;
        const int h      = itid >> 3;
        const int i4     = itid & 7;
        const int s_off  = (itid & 8) | ((itid >> 2) & 1);
        const bool valid = (ix < rem);

        uint32_t ql[ROWS_PER_WARP];
        uint32_t qh[ROWS_PER_WARP];
        uint8_t  sc_u8[ROWS_PER_WARP];

        // Phase A: issue 8 aligned plane loads + 4 scale loads independently, then pack scales and stitch.
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const uint8_t * base = row_ptrs[r];
            const uint32_t mis   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(base) & 2u);
            ql[r]    = valid ? reinterpret_cast<const uint32_t *>(base + 64 * h + mis)[i4]      : 0u;
            qh[r]    = valid ? reinterpret_cast<const uint32_t *>(base + 128 + 32 * h + mis)[i4] : 0u;
            sc_u8[r] = valid ? (base + 192)[itid] : uint8_t(0);
        }
        // Pack all 4 rows' scales into one uint32_t before stitching (saves 3 regs across stitch, 1 SHFL/quarter).
        const uint32_t sc_pack = (static_cast<uint32_t>(sc_u8[0])
                                | (static_cast<uint32_t>(sc_u8[1]) << 8)
                                | (static_cast<uint32_t>(sc_u8[2]) << 16)
                                | (static_cast<uint32_t>(sc_u8[3]) << 24)) ^ 0x80808080u;

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const uint8_t * base = row_ptrs[r];
            const uint32_t mis   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(base) & 2u);
            ql[r] = q6_k_stitch_uniform_mis(ql[r], base + 64 * h,       i4, mis, valid);
            qh[r] = q6_k_stitch_uniform_mis(qh[r], base + 128 + 32 * h, i4, mis, valid);
        }

        float blk_sum[ROWS_PER_WARP];
        {
            const uint32_t sc_q = __shfl_sync(0xffffffffu, sc_pack, s_off + 0, 16);
            if (valid) {
                const float4 y0 = y4[0];
                blk_sum[0] = dot4_u6_bias((ql[0] & 0x0F0F0F0Fu) | ((qh[0] << 4) & 0x30303030u), y0) * unpack_u8_bias_f32<0>(sc_q);
                blk_sum[1] = dot4_u6_bias((ql[1] & 0x0F0F0F0Fu) | ((qh[1] << 4) & 0x30303030u), y0) * unpack_u8_bias_f32<1>(sc_q);
                blk_sum[2] = dot4_u6_bias((ql[2] & 0x0F0F0F0Fu) | ((qh[2] << 4) & 0x30303030u), y0) * unpack_u8_bias_f32<2>(sc_q);
                blk_sum[3] = dot4_u6_bias((ql[3] & 0x0F0F0F0Fu) | ((qh[3] << 4) & 0x30303030u), y0) * unpack_u8_bias_f32<3>(sc_q);
            }
        }
        {
            const uint32_t sc_q = __shfl_sync(0xffffffffu, sc_pack, s_off + 4, 16);
            if (valid) {
                const float4 y2 = y4[16];
                blk_sum[0] = fmaf(dot4_u6_bias(((ql[0] >> 4) & 0x0F0F0F0Fu) | (qh[0] & 0x30303030u), y2), unpack_u8_bias_f32<0>(sc_q), blk_sum[0]);
                blk_sum[1] = fmaf(dot4_u6_bias(((ql[1] >> 4) & 0x0F0F0F0Fu) | (qh[1] & 0x30303030u), y2), unpack_u8_bias_f32<1>(sc_q), blk_sum[1]);
                blk_sum[2] = fmaf(dot4_u6_bias(((ql[2] >> 4) & 0x0F0F0F0Fu) | (qh[2] & 0x30303030u), y2), unpack_u8_bias_f32<2>(sc_q), blk_sum[2]);
                blk_sum[3] = fmaf(dot4_u6_bias(((ql[3] >> 4) & 0x0F0F0F0Fu) | (qh[3] & 0x30303030u), y2), unpack_u8_bias_f32<3>(sc_q), blk_sum[3]);
            }
        }

        // Phase B: ql0 is dead; issue 4 ql1 loads independently, then stitch.
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const uint8_t * base = row_ptrs[r];
            const uint32_t mis   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(base) & 2u);
            ql[r] = valid ? reinterpret_cast<const uint32_t *>(base + 64 * h + 32 + mis)[i4] : 0u;
        }
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const uint8_t * base = row_ptrs[r];
            const uint32_t mis   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(base) & 2u);
            ql[r] = q6_k_stitch_uniform_mis(ql[r], base + 64 * h + 32, i4, mis, valid);
        }
        {
            const uint32_t sc_q = __shfl_sync(0xffffffffu, sc_pack, s_off + 2, 16);
            if (valid) {
                const float4 y1 = y4[8];
                blk_sum[0] = fmaf(dot4_u6_bias((ql[0] & 0x0F0F0F0Fu) | ((qh[0] << 2) & 0x30303030u), y1), unpack_u8_bias_f32<0>(sc_q), blk_sum[0]);
                blk_sum[1] = fmaf(dot4_u6_bias((ql[1] & 0x0F0F0F0Fu) | ((qh[1] << 2) & 0x30303030u), y1), unpack_u8_bias_f32<1>(sc_q), blk_sum[1]);
                blk_sum[2] = fmaf(dot4_u6_bias((ql[2] & 0x0F0F0F0Fu) | ((qh[2] << 2) & 0x30303030u), y1), unpack_u8_bias_f32<2>(sc_q), blk_sum[2]);
                blk_sum[3] = fmaf(dot4_u6_bias((ql[3] & 0x0F0F0F0Fu) | ((qh[3] << 2) & 0x30303030u), y1), unpack_u8_bias_f32<3>(sc_q), blk_sum[3]);
            }
        }
        {
            const uint32_t sc_q = __shfl_sync(0xffffffffu, sc_pack, s_off + 6, 16);
            if (valid) {
                const float4 y3 = y4[24];
                blk_sum[0] = fmaf(dot4_u6_bias(((ql[0] >> 4) & 0x0F0F0F0Fu) | ((qh[0] >> 2) & 0x30303030u), y3), unpack_u8_bias_f32<0>(sc_q), blk_sum[0]);
                blk_sum[1] = fmaf(dot4_u6_bias(((ql[1] >> 4) & 0x0F0F0F0Fu) | ((qh[1] >> 2) & 0x30303030u), y3), unpack_u8_bias_f32<1>(sc_q), blk_sum[1]);
                blk_sum[2] = fmaf(dot4_u6_bias(((ql[2] >> 4) & 0x0F0F0F0Fu) | ((qh[2] >> 2) & 0x30303030u), y3), unpack_u8_bias_f32<2>(sc_q), blk_sum[2]);
                blk_sum[3] = fmaf(dot4_u6_bias(((ql[3] >> 4) & 0x0F0F0F0Fu) | ((qh[3] >> 2) & 0x30303030u), y3), unpack_u8_bias_f32<3>(sc_q), blk_sum[3]);

                // ql, qh, y3, sc_q are all dead; issue all 4 half d loads together.
                half d_h[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    d_h[r] = *reinterpret_cast<const half *>(row_ptrs[r] + 208);
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    acc[r] = fmaf(__half2float(d_h[r]), blk_sum[r], acc[r]);
                }
            }
        }

        rem -= BLOCKS_PER_ITER;
        if (rem <= 0) {
            break;
        }
        if (ix < rem) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                row_ptrs[r] += BLOCKS_PER_ITER * sizeof(block_q6_K);
            }
            y4 += BLOCKS_PER_ITER * (QK_K / 4);
        }
    }

    const int lane_out  = threadIdx.x;
    const int first_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    float * dst_base    = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float sum = warp_reduce_sum<WARP_SIZE>(acc[r]);
        if (lane_out == 0 && first_row + r < nrows) {
            dst_base[first_row + r] = sum;
        }
    }
}

static __device__ __forceinline__ void q6_k_smem_load_4rows_scales_d(
        const uint32_t * smem_u32,
        const int w_base_u32,
        const bool odd_w,
        const int lane,
        uint32_t & sc_xor80,
        uint32_t & qh62_raw,
        float & d_all) {
    // 1 LDS.32 loads all 16 scales + d for all 4 rows; lanes 8*r + 0..3 hold aligned scales, 8*r + 4 holds d.
    const int r_sc   = lane >> 3;
    const int w_sc   = lane & 7;
    const int sc_off = r_sc * 210 + w_base_u32 + 48 + ((w_sc < 5) ? w_sc : 0);
    const uint32_t raw_sc = smem_u32[sc_off];
    qh62_raw = raw_sc;

    const uint32_t next_sc = __shfl_down_sync(0xffffffffu, raw_sc, 1, 8);
    const uint32_t aligned = odd_w ? __byte_perm(raw_sc, next_sc, 0x5432u) : raw_sc;
    sc_xor80 = aligned ^ 0x80808080u;

    const uint16_t d_bits = odd_w ? static_cast<uint16_t>(raw_sc >> 16) : static_cast<uint16_t>(raw_sc & 0xFFFFu);
    half d_h;
    memcpy(&d_h, &d_bits, sizeof(d_h));
    d_all = __half2float(d_h);
}

static __device__ __forceinline__ float q6_k_smem_block_dot_v2(
        const uint32_t * b32,
        const int r,
        const int lane,
        const int h,
        const int i4,
        const int qh_idx,
        const int q_shift,
        const uint32_t sc_byte_shift,
        const bool odd_w,
        const uint32_t sc_xor80,
        const uint32_t qh62_raw,
        const float d_all,
        const float4 y_lo,
        const float4 y_hi) {
    // Only 2 LDS.32 per row (ql and qh) on both even and odd warps; scales and d come from registers via SHFL.
    const uint32_t ql_raw = b32[lane];
    const uint32_t qh_raw = b32[32 + qh_idx];

    uint32_t ql, qh;
    if (!odd_w) {
        ql = ql_raw;
        qh = qh_raw;
    } else {
        const uint32_t ql_dn  = __shfl_down_sync(0xffffffffu, ql_raw, 1);
        const uint32_t qh_l0  = __shfl_sync(0xffffffffu, qh_raw, 0);
        const uint32_t ql_hi  = (lane == 31) ? qh_l0 : ql_dn;
        ql = __byte_perm(ql_raw, ql_hi, 0x5432u);

        const uint32_t qh_dn  = __shfl_down_sync(0xffffffffu, qh_raw, 1, 8);
        const uint32_t qh_l16 = __shfl_sync(0xffffffffu, qh_raw, 16);
        const uint32_t qh_end = __shfl_sync(0xffffffffu, qh62_raw, 8 * r);
        const uint32_t qh_hi  = (i4 < 7) ? qh_dn : (h == 0 ? qh_l16 : qh_end);
        qh = __byte_perm(qh_raw, qh_hi, 0x5432u);
    }

    const uint32_t sw0 = __shfl_sync(0xffffffffu, sc_xor80, 8 * r + 2 * h);
    const uint32_t sw1 = __shfl_sync(0xffffffffu, sc_xor80, 8 * r + 2 * h + 1);
    const float d      = __shfl_sync(0xffffffffu, d_all,    8 * r + 4);

    const float sc0 = __fadd_rn(__uint_as_float(((sw0 >> sc_byte_shift) & 0xFFu) | 0x4B000000u), -8388736.0f);
    const float sc1 = __fadd_rn(__uint_as_float(((sw1 >> sc_byte_shift) & 0xFFu) | 0x4B000000u), -8388736.0f);

    const uint32_t qh_s = qh >> q_shift;
    const uint32_t q_lo = (ql & 0x0F0F0F0Fu) | ((qh_s << 4) & 0x30303030u);
    const uint32_t q_hi = ((ql >> 4) & 0x0F0F0F0Fu) | (qh_s & 0x30303030u);

    const float dot_lo = dot4_u6_bias(q_lo, y_lo);
    const float dot_hi = dot4_u6_bias(q_hi, y_hi);
    return fmaf(dot_lo, sc0, dot_hi * sc1) * d;
}

template <int ROWS_PER_BLOCK>
__launch_bounds__(128)
static __global__ void mul_mat_vec_q6_K_kepler_nb4(
        const block_q6_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nrows,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    static_assert((ROWS_PER_BLOCK % 4) == 0, "ROWS_PER_BLOCK must be multiple of 4");

    const int block_row0 = blockIdx.x * ROWS_PER_BLOCK;
    if (block_row0 >= nrows) {
        return;
    }

    // 4 rows x 4 blocks x 210 bytes = 3360 bytes = 840 uint32 (0 bank conflicts on STS.32).
    __shared__ uint32_t smem_buf[840];
    __shared__ float smem_red[4][4];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y; // 0..3 -> owns super-block w
    const int tid  = w * WARP_SIZE + lane;

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q6_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    const int h       = lane >> 4;
    const int q       = (lane >> 3) & 1;
    const int i4      = lane & 7;
    const int qh_idx  = (h << 3) | i4;
    const int q_shift = q << 1;
    const uint32_t sc_byte_shift = static_cast<uint32_t>(((q << 1) | (i4 >> 2)) << 3);
    const bool odd_w  = (w & 1) != 0;
    const int w_base_u32 = (w * 210 - (odd_w ? 2 : 0)) >> 2;

    // Load y once for block w and hold in registers across all ROWS_PER_BLOCK rows.
    const float4 * y4 = reinterpret_cast<const float4 *>(y_base + static_cast<int64_t>(w) * QK_K + 128 * h + 32 * q + 4 * i4);
    const float4 y_lo = y4[0];
    const float4 y_hi = y4[16];

    float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;

#pragma unroll 1
    for (int r_off = 0; r_off < ROWS_PER_BLOCK; r_off += 4) {
        const int r_base = block_row0 + r_off;
        if (r_base >= nrows) {
            break;
        }

        const uint32_t * g_src = reinterpret_cast<const uint32_t *>(x_base + static_cast<int64_t>(r_base) * 4);
        const bool full4 = (r_base + 4 <= nrows);
        if (full4) {
            // Split 4+3 to keep register peak <= 40 (75% occupancy on sm_30).
            const uint32_t v0 = g_src[tid];
            const uint32_t v1 = g_src[tid + 128];
            const uint32_t v2 = g_src[tid + 256];
            const uint32_t v3 = g_src[tid + 384];
            smem_buf[tid]       = v0;
            smem_buf[tid + 128] = v1;
            smem_buf[tid + 256] = v2;
            smem_buf[tid + 384] = v3;
            const uint32_t v4 = g_src[tid + 512];
            const uint32_t v5 = g_src[tid + 640];
            const uint32_t v6 = (tid < 72) ? g_src[tid + 768] : 0u;
            smem_buf[tid + 512] = v4;
            smem_buf[tid + 640] = v5;
            if (tid < 72) {
                smem_buf[tid + 768] = v6;
            }
        } else {
            const int n_words = (nrows - r_base) * 210;
            for (int i = tid; i < n_words; i += 128) {
                smem_buf[i] = g_src[i];
            }
        }
        __syncthreads();

        uint32_t sc_xor80, qh62_raw;
        float d_all;
        q6_k_smem_load_4rows_scales_d(smem_buf, w_base_u32, odd_w, lane, sc_xor80, qh62_raw, d_all);

        float rv[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float val = q6_k_smem_block_dot_v2(
                smem_buf + r * 210 + w_base_u32, r, lane, h, i4, qh_idx, q_shift, sc_byte_shift, odd_w,
                sc_xor80, qh62_raw, d_all, y_lo, y_hi);
            rv[r] = (full4 || r_base + r < nrows) ? val : 0.0f;
        }

        const float sum = warp_reduce_sum_4rows_packed(rv[0], rv[1], rv[2], rv[3], lane);
        if (lane < 4) {
            smem_red[w][lane] = sum;
        }
        __syncthreads();

        if (w == 0 && lane < 4 && r_base + lane < nrows) {
            dst_base[r_base + lane] = (smem_red[0][lane] + smem_red[1][lane]) + (smem_red[2][lane] + smem_red[3][lane]);
        }
    }
}

__launch_bounds__(128)
static __global__ void mul_mat_vec_q6_K_kepler_smem(
        const block_q6_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int ROWS_PER_BLOCK = 4;

    const int block_row0 = blockIdx.x * ROWS_PER_BLOCK;
    if (block_row0 >= nrows) {
        return;
    }

    __shared__ uint32_t smem_buf[ROWS_PER_BLOCK][210];
    __shared__ float smem_red[4][ROWS_PER_BLOCK];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y; // 0..3

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q6_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    const int h       = lane >> 4;
    const int q       = (lane >> 3) & 1;
    const int i4      = lane & 7;
    const int qh_idx  = (h << 3) | i4;
    const int q_shift = q << 1;
    const uint32_t sc_byte_shift = static_cast<uint32_t>(((q << 1) | (i4 >> 2)) << 3);
    const bool odd_w  = (w & 1) != 0;
    const int w_base_u32 = (w * 210 - (odd_w ? 2 : 0)) >> 2;

    const bool my_row_valid = (block_row0 + w < nrows);
    const uint32_t * my_row_src = my_row_valid
        ? reinterpret_cast<const uint32_t *>(x_base + static_cast<int64_t>(block_row0 + w) * stride_row_x)
        : nullptr;

    float acc[ROWS_PER_BLOCK] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const uint32_t * smem_flat = &smem_buf[0][0];

#pragma unroll 1
    for (int ib0 = 0; ib0 < nb; ib0 += 4) {
        const int n_blk = (nb - ib0 < 4) ? (nb - ib0) : 4;

        // Each warp w streams n_blk blocks of row w coalesced into smem_buf[w].
        if (my_row_valid) {
            const uint32_t * g_src = my_row_src + static_cast<int64_t>(ib0) * 52 + (ib0 >> 1);
            if (n_blk == 4) {
                const uint32_t v0 = g_src[lane];
                const uint32_t v1 = g_src[lane + 32];
                const uint32_t v2 = g_src[lane + 64];
                const uint32_t v3 = g_src[lane + 96];
                smem_buf[w][lane]      = v0;
                smem_buf[w][lane + 32] = v1;
                smem_buf[w][lane + 64] = v2;
                smem_buf[w][lane + 96] = v3;
                const uint32_t v4 = g_src[lane + 128];
                const uint32_t v5 = g_src[lane + 160];
                const uint32_t v6 = (lane < 18) ? g_src[lane + 192] : 0u;
                smem_buf[w][lane + 128] = v4;
                smem_buf[w][lane + 160] = v5;
                if (lane < 18) {
                    smem_buf[w][lane + 192] = v6;
                }
            } else {
                const int n_words = (n_blk * 210) >> 2;
                for (int i = lane; i < n_words; i += WARP_SIZE) {
                    smem_buf[w][i] = g_src[i];
                }
            }
        }
        __syncthreads();

        if (ib0 + w < nb) {
            const float4 * y4 = reinterpret_cast<const float4 *>(y_base + static_cast<int64_t>(ib0 + w) * QK_K + 128 * h + 32 * q + 4 * i4);
            const float4 y_lo = y4[0];
            const float4 y_hi = y4[16];

            uint32_t sc_xor80, qh62_raw;
            float d_all;
            q6_k_smem_load_4rows_scales_d(smem_flat, w_base_u32, odd_w, lane, sc_xor80, qh62_raw, d_all);

#pragma unroll
            for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
                if (block_row0 + r < nrows) {
                    acc[r] += q6_k_smem_block_dot_v2(
                        smem_flat + r * 210 + w_base_u32, r, lane, h, i4, qh_idx, q_shift, sc_byte_shift, odd_w,
                        sc_xor80, qh62_raw, d_all, y_lo, y_hi);
                }
            }
        }

        if (ib0 + 4 < nb) {
            __syncthreads();
        }
    }

    const float sum = warp_reduce_sum_4rows_packed(acc[0], acc[1], acc[2], acc[3], lane);
    if (lane < ROWS_PER_BLOCK) {
        smem_red[w][lane] = sum;
    }
    __syncthreads();

    float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
    if (w == 0 && lane < ROWS_PER_BLOCK && block_row0 + lane < nrows) {
        dst_base[block_row0 + lane] = (smem_red[0][lane] + smem_red[1][lane]) + (smem_red[2][lane] + smem_red[3][lane]);
    }
}

template <bool IS_ODD>
static __device__ __forceinline__ float q6_k_d32_dot_dsc_loaded(
        const uint32_t ql_raw,
        const uint32_t qh_raw,
        const int r,
        const float dsc_pair,
        const uint32_t qh62_raw,
        const float4 y_lo,
        const float4 y_hi,
        const float acc_in) {
    const int lane    = threadIdx.x;
    const int h       = lane >> 4;
    const int i4      = lane & 7;
    const int q_shift = (lane >> 2) & 2;
    const int s_lo    = ((lane & 16) >> 1) | ((lane >> 2) & 3);
    const int row_off = r << 4;

    uint32_t ql, qh;
    if constexpr (!IS_ODD) {
        ql = ql_raw;
        qh = qh_raw;
    } else {
        const uint32_t ql_dn  = __shfl_down_sync(0xffffffffu, ql_raw, 1);
        const uint32_t qh_l0  = __shfl_sync(0xffffffffu, qh_raw, 0);
        const uint32_t ql_hi  = (lane == 31) ? qh_l0 : ql_dn;
        ql = __byte_perm(ql_raw, ql_hi, 0x5432u);

        const uint32_t qh_dn  = __shfl_down_sync(0xffffffffu, qh_raw, 1, 8);
        const uint32_t qh_l16 = __shfl_sync(0xffffffffu, qh_raw, 16);
        const uint32_t qh_end = __shfl_sync(0xffffffffu, qh62_raw, row_off);
        const uint32_t qh_hi  = (i4 < 7) ? qh_dn : (h == 0 ? qh_l16 : qh_end);
        qh = __byte_perm(qh_raw, qh_hi, 0x5432u);
    }

    const float dsc0 = __shfl_sync(0xffffffffu, dsc_pair, row_off + s_lo);
    const float dsc1 = __shfl_sync(0xffffffffu, dsc_pair, row_off + s_lo + 4);

    const uint32_t qh_s = qh >> q_shift;
    const uint32_t q_lo = (ql & 0x0F0F0F0Fu) | ((qh_s << 4) & 0x30303030u);
    const uint32_t q_hi = ((ql >> 4) & 0x0F0F0F0Fu) | (qh_s & 0x30303030u);

    const float dot_lo = dot4_u6_bias(q_lo, y_lo);
    const float dot_hi = dot4_u6_bias(q_hi, y_hi);
    return fmaf(dot_lo, dsc0, fmaf(dot_hi, dsc1, acc_in));
}

template <bool FULL2>
__launch_bounds__(128)
static __global__ void mul_mat_vec_q6_K_kepler_d32(
        const block_q6_K * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int ROWS_PER_WARP   = 2; // 2 rows x 16 scales = 32 lanes -> 1 dsc reg, <= 40 regs -> 75% occupancy
    constexpr int ROWS_PER_BLOCK  = WARPS_PER_BLOCK * ROWS_PER_WARP;

    const int first_row_in = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    if (first_row_in >= nrows) {
        return;
    }

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);

    const block_q6_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;

    const uint8_t * x_ptr   = reinterpret_cast<const uint8_t *>(x_base + static_cast<int64_t>(first_row_in) * stride_row_x);
    const int64_t row_bytes = (FULL2 || first_row_in + 1 < nrows)
        ? (stride_row_x * static_cast<int64_t>(sizeof(block_q6_K)))
        : 0;
    const float4 * y4       = reinterpret_cast<const float4 *>(
        y_base + 128 * (threadIdx.x >> 4) + 32 * ((threadIdx.x >> 3) & 1) + 4 * (threadIdx.x & 7));

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Compile-time even/odd block pairs; both rows' ql/qh and scales+d issued together for full DRAM MLP.
#pragma unroll 1
    for (int ib = 0; ib < nb; ib += 2) {
        const int lane   = threadIdx.x;
        const int qh_idx = ((lane >> 4) << 3) | (lane & 7);
        const int r_sc   = lane >> 4; // 0 for lanes 0..15 (row 0), 1 for lanes 16..31 (row 1)
        const int s_idx  = lane & 15;
        const int sc_w   = (s_idx < 5) ? s_idx : 0;
        const uint32_t sh = static_cast<uint32_t>((s_idx & 3) << 3);

        // Even block (4-byte aligned, zero stitching).
        {
            const float4 y_lo = y4[0];
            const float4 y_hi = y4[16];

            const uint32_t * b0 = reinterpret_cast<const uint32_t *>(x_ptr);
            const uint32_t * b1 = reinterpret_cast<const uint32_t *>(x_ptr + row_bytes);
            const uint32_t * bs = r_sc ? b1 : b0;

            const uint32_t raw_sc = bs[48 + sc_w];
            const uint32_t ql0    = b0[lane];
            const uint32_t qh0    = b0[32 + qh_idx];
            const uint32_t ql1    = b1[lane];
            const uint32_t qh1    = b1[32 + qh_idx];

            const uint16_t d_bits = static_cast<uint16_t>(raw_sc & 0xFFFFu);
            half d_h;
            memcpy(&d_h, &d_bits, sizeof(d_h));
            const float d_all       = __half2float(d_h);
            const uint32_t sc_xor80 = raw_sc ^ 0x80808080u;

            const uint32_t w01 = __shfl_sync(0xffffffffu, sc_xor80, (r_sc << 4) | (s_idx >> 2));
            const float    d01 = __shfl_sync(0xffffffffu, d_all,    (r_sc << 4) | 4);
            const float   sc01 = __fadd_rn(__uint_as_float(((w01 >> sh) & 0xFFu) | 0x4B000000u), -8388736.0f);
            const float dsc_01 = d01 * sc01;

            acc0 = q6_k_d32_dot_dsc_loaded<false>(ql0, qh0, 0, dsc_01, 0u, y_lo, y_hi, acc0);
            acc1 = q6_k_d32_dot_dsc_loaded<false>(ql1, qh1, 1, dsc_01, 0u, y_lo, y_hi, acc1);
        }

        // Odd block (offset +208 is 4-byte aligned and inside block ib).
        {
            const float4 y_lo = y4[64];
            const float4 y_hi = y4[80];

            const uint32_t * b0 = reinterpret_cast<const uint32_t *>(x_ptr + 208);
            const uint32_t * b1 = reinterpret_cast<const uint32_t *>(x_ptr + row_bytes + 208);
            const uint32_t * bs = r_sc ? b1 : b0;

            const uint32_t raw_sc = bs[48 + sc_w];
            const uint32_t ql0    = b0[lane];
            const uint32_t qh0    = b0[32 + qh_idx];
            const uint32_t ql1    = b1[lane];
            const uint32_t qh1    = b1[32 + qh_idx];

            const uint32_t next_sc  = __shfl_down_sync(0xffffffffu, raw_sc, 1, 16);
            const uint32_t sc_xor80 = __byte_perm(raw_sc, next_sc, 0x5432u) ^ 0x80808080u;
            const uint16_t d_bits   = static_cast<uint16_t>(raw_sc >> 16);
            half d_h;
            memcpy(&d_h, &d_bits, sizeof(d_h));
            const float d_all = __half2float(d_h);

            const uint32_t w01 = __shfl_sync(0xffffffffu, sc_xor80, (r_sc << 4) | (s_idx >> 2));
            const float    d01 = __shfl_sync(0xffffffffu, d_all,    (r_sc << 4) | 4);
            const float   sc01 = __fadd_rn(__uint_as_float(((w01 >> sh) & 0xFFu) | 0x4B000000u), -8388736.0f);
            const float dsc_01 = d01 * sc01;

            acc0 = q6_k_d32_dot_dsc_loaded<true>(ql0, qh0, 0, dsc_01, raw_sc, y_lo, y_hi, acc0);
            acc1 = q6_k_d32_dot_dsc_loaded<true>(ql1, qh1, 1, dsc_01, raw_sc, y_lo, y_hi, acc1);
        }

        x_ptr += 2 * sizeof(block_q6_K);
        y4    += 2 * (QK_K / 4);
    }

    // Packed 2-row warp reduction (5 SHFLs instead of 10).
    const int lane_out  = threadIdx.x;
    const bool odd_l    = (lane_out & 1) != 0;
    float p = (odd_l ? acc1 : acc0) + __shfl_xor_sync(0xffffffffu, odd_l ? acc0 : acc1, 1);
    p += __shfl_xor_sync(0xffffffffu, p, 2);
    p += __shfl_xor_sync(0xffffffffu, p, 4);
    p += __shfl_xor_sync(0xffffffffu, p, 8);
    p += __shfl_xor_sync(0xffffffffu, p, 16);

    const int first_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    if (lane_out < ROWS_PER_WARP && (FULL2 || first_row + lane_out < nrows)) {
        float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
        dst_base[first_row + lane_out] = p;
    }
}

static void launch_mmvk_q4_k_vk2(
        const block_q4_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int rows_per_block = 16;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    if ((nb & 1) == 0 && (nrows & 3) == 0) {
        ggml_cuda_kernel_launch(
            mul_mat_vec_q4_K_kepler_vk2<4, true, true, false>,
            launch_params,
            src0_d, src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
    } else {
        ggml_cuda_kernel_launch(
            mul_mat_vec_q4_K_kepler_vk2<4, false, false, false>,
            launch_params,
            src0_d, src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
    }
}

template <int WARPS_PER_BLOCK>
static void launch_mmvk_q4_k_sy(
        const block_q4_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int rows_per_block = WARPS_PER_BLOCK * 4;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);
    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, WARPS_PER_BLOCK, 1);
    const size_t shared_bytes = static_cast<size_t>(nb * QK_K + nb * 8) * sizeof(float);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, shared_bytes, stream);

    if ((nb & 1) == 0 && (nrows & 3) == 0) {
        ggml_cuda_kernel_launch(
            mul_mat_vec_q4_K_kepler_vk2<WARPS_PER_BLOCK, true, true, true>,
            launch_params,
            src0_d, src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
    } else {
        ggml_cuda_kernel_launch(
            mul_mat_vec_q4_K_kepler_vk2<WARPS_PER_BLOCK, false, false, true>,
            launch_params,
            src0_d, src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
    }
}

static void launch_mmvk_q6_k_d32(
        const block_q6_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    // The d32 kernel always processes blocks in (even, odd) pairs (`ib += 2`) and has no per-block guard:
    // an odd nb would read y and weights past the end of the row. Every caller must check this; the
    // dispatcher does (q6_even_pairs_ok), and this assert keeps a future caller from launching it unguarded.
    GGML_ASSERT((nb & 1) == 0 && "mul_mat_vec_q6_K_kepler_d32 needs an even number of blocks per row");
    constexpr int rows_per_block = 8;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    if ((nrows & 1) == 0) {
        ggml_cuda_kernel_launch(
            mul_mat_vec_q6_K_kepler_d32<true>,
            launch_params,
            src0_d, src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
    } else {
        ggml_cuda_kernel_launch(
            mul_mat_vec_q6_K_kepler_d32<false>,
            launch_params,
            src0_d, src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
    }
}

static void launch_mmvk_q4_k_c8(
        const block_q4_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int rows_per_block = 16;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q4_K_kepler_c8,
        launch_params,
        src0_d, src1_d, dst_d,
        nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

template <int ROWS_PER_WARP, int WARPS_PER_BLOCK>
static void launch_mmvk_q4_k_swiz(
        const block_q4_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int rows_per_block = WARPS_PER_BLOCK * ROWS_PER_WARP;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, WARPS_PER_BLOCK, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q4_K_kepler_swiz<ROWS_PER_WARP, WARPS_PER_BLOCK>,
        launch_params,
        src0_d, src1_d, dst_d,
        nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

template <int ROWS_PER_BLOCK>
static void launch_mmvk_q6_k_nb4(
        const block_q6_K * src0_d, const float * src1_d, float * dst_d,
        const int nrows,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q6_K_kepler_nb4<ROWS_PER_BLOCK>,
        launch_params,
        src0_d, src1_d, dst_d,
        nrows,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

static void launch_mmvk_q6_k_smem(
        const block_q6_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int rows_per_block = 4;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q6_K_kepler_smem,
        launch_params,
        src0_d, src1_d, dst_d,
        nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

static __device__ __forceinline__ float q8_0_dot4_bias(const uint32_t q, const float4 y) {
    const uint32_t u = q ^ 0x80808080u;
    uint32_t u_ff;
    asm("or.b32 %0, %1, 0xff000000;" : "=r"(u_ff) : "r"(u));
    const float q0 = __fadd_rn(__uint_as_float(u_ff & 0x4B0000FFu), -8388736.0f);
    const float q1 = __fadd_rn(__uint_as_float(u_ff & 0x4700FF00u), -32896.0f);
    asm("or.b32 %0, %1, 0xff000000;" : "=r"(u_ff) : "r"(u >> 16));
    const float q2 = __fadd_rn(__uint_as_float(u_ff & 0x4B0000FFu), -8388736.0f);
    const float q3 = __fadd_rn(__uint_as_float(u_ff & 0x4700FF00u), -32896.0f);
    const float sum02 = fmaf(q2, y.z, q0 * y.x);
    const float sum13 = fmaf(q3, y.w, q1 * y.y);
    return sum02 + sum13;
}

static __device__ __forceinline__ float q8_0_scale(const block_q8_0 * block) {
    half d;
    memcpy(&d, &block->d, sizeof(d));
    return __half2float(d);
}

static __device__ __forceinline__ uint32_t q8_0_load_word_safe(const int8_t * qs, const int lane) {
    const uint16_t * words = reinterpret_cast<const uint16_t *>(qs + 4 * lane);
    return static_cast<uint32_t>(words[0]) | (static_cast<uint32_t>(words[1]) << 16);
}

template <bool ODD>
static __device__ __forceinline__ uint32_t q8_0_load_word_pair(const block_q8_0 * block, const int lane) {
    const int8_t * qs = block->qs;
    if constexpr (ODD) {
        return reinterpret_cast<const uint32_t *>(qs)[lane];
    } else {
        const uint32_t * words = reinterpret_cast<const uint32_t *>(qs - 2 + 4 * lane);
        return (words[0] >> 16) | (words[1] << 16);
    }
}

template <bool PAIRED>
__launch_bounds__(128)
static __global__ void mul_mat_vec_q8_0_kepler(
        const block_q8_0 * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int ROWS_PER_WARP = 4;
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;

    const int first_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y * ROWS_PER_WARP;
    if (first_row >= nrows) {
        return;
    }
    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x = fastdiv(blockIdx.z, sample_ratio);
    const block_q8_0 * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float * y_base = vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y;
    const int lane = threadIdx.x;
    const int row_lane = lane >> 3;
    const int lane_in_row = lane & 7;
    const int row = first_row + row_lane;
    const block_q8_0 * row_x = x_base + (row < nrows ? row : first_row) * stride_row_x;
    const float * y = y_base + 4 * lane_in_row;
    float acc_even = 0.0f;
    float acc_odd = 0.0f;

    if constexpr (PAIRED) {
        for (int ib = 0; ib < nb; ib += 2) {
            if (row < nrows) {
                const block_q8_0 * b0 = row_x + ib;
                const block_q8_0 * b1 = b0 + 1;
                const uint32_t q0 = q8_0_load_word_pair<false>(b0, lane_in_row);
                const uint32_t q1 = q8_0_load_word_pair<true>(b1, lane_in_row);
                const float4 y0 = *reinterpret_cast<const float4 *>(y + ib * QK8_0);
                const float4 y1 = *reinterpret_cast<const float4 *>(y + (ib + 1) * QK8_0);
                acc_even = fmaf(q8_0_scale(b0), q8_0_dot4_bias(q0, y0), acc_even);
                acc_odd = fmaf(q8_0_scale(b1), q8_0_dot4_bias(q1, y1), acc_odd);
            }
        }
    } else {
        for (int ib = 0; ib < nb; ++ib) {
            if (row < nrows) {
                const block_q8_0 * block = row_x + ib;
                const uint32_t q = q8_0_load_word_safe(block->qs, lane_in_row);
                const float4 y4 = *reinterpret_cast<const float4 *>(y + ib * QK8_0);
                if ((ib & 1) == 0) {
                    acc_even = fmaf(q8_0_scale(block), q8_0_dot4_bias(q, y4), acc_even);
                } else {
                    acc_odd = fmaf(q8_0_scale(block), q8_0_dot4_bias(q, y4), acc_odd);
                }
            }
        }
    }

    float acc = acc_even + acc_odd;
    for (int offset = 4; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset, 8);
    }
    if (lane_in_row == 0 && row < nrows) {
        float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
        dst_base[row] = acc;
    }
}

template <bool PAIRED>
static void launch_mmvk_q8_0(
        const block_q8_0 * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = 16;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q8_0_kepler<PAIRED>, launch_params,
        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
}

// Q8_0 "ws" (warp streams a row; y staged in shared memory). Written 2026-11-21 after nvprof on the
// pair kernel above: 4 rows x 8 lanes per warp gave 3.85 transactions per request, 0.73 replay overhead and
// 34% load efficiency (12 GB/s DRAM). Here a warp reads ONE row: lane = (block-in-group bo = lane / 8,
// 4-quant group g = lane % 8), so each step reads 4 consecutive blocks = 136 contiguous bytes, with U steps
// (4*U blocks) loaded before any use. 16-bit loads only, so any 2-byte alignment works (odd nb, offset
// views): no paired/safe split. y would be re-read per row from L2 (4 bytes of y per ~1 byte of weights),
// so each 128-thread block stages y in 4 KB tiles (32 Q8_0 blocks) in shared memory and its 4 warps
// stream 4 rows each through the tile. FP32 math, fixed summation order (bitwise repeatable).
// Tried and dropped: staging each step's 136 B through shared memory with aligned 32-bit loads
// (63 regs, 84 us vs 69 us at 1024x1024). MINB > 0 caps registers via __launch_bounds__(128, MINB).
template <int U, int MINB, bool SHFL>
__launch_bounds__(128, MINB)
static __global__ void mul_mat_vec_q8_0_kepler_ws(
        const block_q8_0 * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    constexpr int WARPS = 4;
    constexpr int RPW = 4;                 // rows per warp
    constexpr int RPB = WARPS * RPW;       // rows per thread block
    constexpr int TILE_B = 32;             // Q8_0 blocks of y per shared-memory tile (1024 floats)
    __shared__ float4 ys[TILE_B * QK8_0 / 4];

    const int tid  = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane = threadIdx.x;
    const int bo   = lane >> 3;            // block within a 4-block step
    const int g    = lane & 7;             // 4-quant group within the block

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);
    const block_q8_0 * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float4 * y4 = reinterpret_cast<const float4 *>(vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y);
    const int row0 = blockIdx.x * RPB + threadIdx.y * RPW;

    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        acc[r] = 0.0f;
    }

    for (int tb = 0; tb < nb; tb += TILE_B) {
        const int tn = min(TILE_B, nb - tb);
        __syncthreads();                   // every warp is done with the previous tile
        for (int i = tid; i < tn * (QK8_0 / 4); i += WARPS * WARP_SIZE) {
            ys[i] = y4[tb * (QK8_0 / 4) + i];
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = row0 + r;
            if (row < nrows) {             // warp-uniform; no barrier inside
                const block_q8_0 * xr = x_base + row * stride_row_x + tb;
                for (int ib = 0; ib < tn; ib += 4 * U) {
                    uint32_t q[U];
                    float    d[U];
                    if constexpr (SHFL) {
                        // rows 4-byte aligned, nb even: a step's 4 blocks are 34 aligned words; lane l loads
                        // word l (lanes 0-1 also word 32 + l), then each lane gathers its quant word and its
                        // block's scale by shuffle. One coalesced read per step instead of three overlapping
                        // 16-bit reads (nvprof: L2 traffic was ~4.6x DRAM).
                        uint32_t w0[U], w1[U];
#pragma unroll
                        for (int u = 0; u < U; ++u) {
                            const int b0 = ib + 4 * u;
                            const int nw = b0 < tn ? (min(4, tn - b0) * (int) sizeof(block_q8_0)) / 4 : 0;
                            const uint32_t * wsrc = reinterpret_cast<const uint32_t *>(xr + b0);
                            w0[u] = lane < nw ? wsrc[lane] : 0u;
                            w1[u] = lane + WARP_SIZE < nw ? wsrc[lane + WARP_SIZE] : 0u;
                        }
                        const int qo = (int) sizeof(block_q8_0) * bo + 2 + 4 * g;   // byte offset of my 4 quants
                        const int wi = qo >> 2;                                      // <= 33
                        const int dw = ((int) sizeof(block_q8_0) * bo) >> 2;         // word holding my scale
#pragma unroll
                        for (int u = 0; u < U; ++u) {
                            const uint32_t a0 = __shfl_sync(0xffffffffu, w0[u], wi & 31);
                            const uint32_t a1 = __shfl_sync(0xffffffffu, w1[u], wi & 31);
                            const uint32_t lo = wi < WARP_SIZE ? a0 : a1;
                            // the next word is only needed for even bo (quants start 2 bytes into a word); wi + 1 <= 25 there
                            const uint32_t hi = __shfl_sync(0xffffffffu, w0[u], (wi + 1) & 31);
                            q[u] = (bo & 1) ? lo : __byte_perm(lo, hi, 0x5432u);
                            const uint32_t dwv = __shfl_sync(0xffffffffu, w0[u], dw);
                            const uint16_t dbits = (bo & 1) ? static_cast<uint16_t>(dwv >> 16) : static_cast<uint16_t>(dwv & 0xffffu);
                            half dh;
                            memcpy(&dh, &dbits, sizeof(dh));
                            d[u] = __half2float(dh);
                        }
                    } else {
#pragma unroll
                        for (int u = 0; u < U; ++u) {
                            const int b = ib + 4 * u + bo;
                            if (b < tn) {
                                const uint16_t * p = reinterpret_cast<const uint16_t *>(xr + b);
                                d[u] = q8_0_scale(xr + b);
                                q[u] = static_cast<uint32_t>(p[1 + 2 * g]) | (static_cast<uint32_t>(p[2 + 2 * g]) << 16);
                            } else {
                                d[u] = 0.0f;
                                q[u] = 0u;
                            }
                        }
                    }
#pragma unroll
                    for (int u = 0; u < U; ++u) {
                        const int b = ib + 4 * u + bo;
                        if (b < tn) {
                            acc[r] = fmaf(d[u], q8_0_dot4_bias(q[u], ys[b * (QK8_0 / 4) + g]), acc[r]);
                        }
                    }
                }
            }
        }
    }

    float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset, WARP_SIZE);
        }
        if (lane == 0 && row0 + r < nrows) {
            dst_base[row0 + r] = v;
        }
    }
}

template <int U, int MINB, bool SHFL>
static void launch_mmvk_q8_0_ws(
        const block_q8_0 * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = 16;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q8_0_kepler_ws<U, MINB, SHFL>, launch_params,
        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
}

// Q5_0 (22-byte blocks: half d, 32 high bits qh, 16 bytes of nibbles; weight j < 16 is the low nibble of
// qs[j], weight j + 16 the high nibble, high bit of weight w is bit w of qh; value = (5-bit - 16) * d).
// Same structure as Q8_0 ws (2026-11-21, Claude): a warp streams one row, 4 consecutive blocks (88 B) per
// step, U steps in flight, y in 4 KB shared tiles. Lane (bo = lane / 8, g = lane % 8) handles weights
// 4g..4g+3 of block bo: nibble word qs[4(g&3)..] (low nibbles for g < 4, high for g >= 4) and qh bits 4g..4g+3.
// SHFL (rows 4-byte aligned, nb even): a step is 22 aligned words, one load per lane (lanes 0-21), fields
// gathered by shuffle; otherwise 16-bit loads (any 2-byte alignment).
static __device__ __forceinline__ float q5_0_dot4_bias(const uint32_t q, const float4 y) {
    // q holds four unsigned 5-bit values (0..31), one per byte; value - 16 via the magic-number float trick
    uint32_t u_ff;
    asm("or.b32 %0, %1, 0xff000000;" : "=r"(u_ff) : "r"(q));
    const float q0 = __fadd_rn(__uint_as_float(u_ff & 0x4B0000FFu), -8388624.0f);   // 2^23 + 16
    const float q1 = __fadd_rn(__uint_as_float(u_ff & 0x4700FF00u), -32784.0f);     // 2^15 + 16
    asm("or.b32 %0, %1, 0xff000000;" : "=r"(u_ff) : "r"(q >> 16));
    const float q2 = __fadd_rn(__uint_as_float(u_ff & 0x4B0000FFu), -8388624.0f);
    const float q3 = __fadd_rn(__uint_as_float(u_ff & 0x4700FF00u), -32784.0f);
    const float sum02 = fmaf(q2, y.z, q0 * y.x);
    const float sum13 = fmaf(q3, y.w, q1 * y.y);
    return sum02 + sum13;
}

static __device__ __forceinline__ uint32_t q5_0_quads(const uint32_t qsw, const uint32_t qh, const int g) {
    const uint32_t nib = (g < 4 ? qsw : (qsw >> 4)) & 0x0F0F0F0Fu;
    const uint32_t h = (qh >> (4 * g)) & 0xFu;
    const uint32_t hb = ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
    return nib | hb;
}

template <int U, bool SHFL>
__launch_bounds__(128, 1)
static __global__ void mul_mat_vec_q5_0_kepler_ws(
        const block_q5_0 * __restrict__ vx,
        const float * __restrict__ vy,
        float * __restrict__ dst,
        const int nb,
        const int nrows,
        const int64_t stride_row_x,
        const uint3 channel_ratio,
        const int64_t stride_channel_x,
        const int64_t stride_channel_y,
        const int64_t stride_channel_dst,
        const uint3 sample_ratio,
        const int64_t stride_sample_x,
        const int64_t stride_sample_y,
        const int64_t stride_sample_dst) {
    static_assert(sizeof(block_q5_0) == 22, "unexpected block_q5_0 size");
    constexpr int WARPS = 4;
    constexpr int RPW = 4;
    constexpr int RPB = WARPS * RPW;
    constexpr int TILE_B = 32;
    constexpr int BS = (int) sizeof(block_q5_0);
    __shared__ float4 ys[TILE_B * QK5_0 / 4];

    const int tid  = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane = threadIdx.x;
    const int bo   = lane >> 3;
    const int g    = lane & 7;

    const uint32_t channel_x = fastdiv(blockIdx.y, channel_ratio);
    const uint32_t sample_x  = fastdiv(blockIdx.z, sample_ratio);
    const block_q5_0 * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float4 * y4 = reinterpret_cast<const float4 *>(vy + blockIdx.z * stride_sample_y + blockIdx.y * stride_channel_y);
    const int row0 = blockIdx.x * RPB + threadIdx.y * RPW;

    // byte offsets inside a 4-block step for my fields (SHFL path)
    const int o_d  = BS * bo;                  // scale (2 B)
    const int o_qh = BS * bo + 2;              // high bits (4 B)
    const int o_qs = BS * bo + 6 + 4 * (g & 3);// my nibble word (4 B)

    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        acc[r] = 0.0f;
    }

    for (int tb = 0; tb < nb; tb += TILE_B) {
        const int tn = min(TILE_B, nb - tb);
        __syncthreads();
        for (int i = tid; i < tn * (QK5_0 / 4); i += WARPS * WARP_SIZE) {
            ys[i] = y4[tb * (QK5_0 / 4) + i];
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = row0 + r;
            if (row < nrows) {
                const block_q5_0 * xr = x_base + row * stride_row_x + tb;
                for (int ib = 0; ib < tn; ib += 4 * U) {
                    uint32_t q[U];
                    float    d[U];
                    if constexpr (SHFL) {
                        uint32_t w[U];
#pragma unroll
                        for (int u = 0; u < U; ++u) {
                            const int b0 = ib + 4 * u;
                            const int nw = b0 < tn ? (min(4, tn - b0) * BS) / 4 : 0;   // 11 or 22 words
                            w[u] = lane < nw ? reinterpret_cast<const uint32_t *>(xr + b0)[lane] : 0u;
                        }
#pragma unroll
                        for (int u = 0; u < U; ++u) {
                            const uint32_t dwv = __shfl_sync(0xffffffffu, w[u], o_d >> 2);
                            const uint32_t h0  = __shfl_sync(0xffffffffu, w[u], o_qh >> 2);
                            const uint32_t h1  = __shfl_sync(0xffffffffu, w[u], ((o_qh >> 2) + 1) & 31);
                            const uint32_t s0  = __shfl_sync(0xffffffffu, w[u], o_qs >> 2);
                            const uint32_t s1  = __shfl_sync(0xffffffffu, w[u], ((o_qs >> 2) + 1) & 31);
                            const bool mis = (o_qs & 2) != 0;    // even bo: qh and qs start 2 bytes into a word
                            const uint32_t qh  = mis ? __byte_perm(h0, h1, 0x5432u) : h0;
                            const uint32_t qsw = mis ? __byte_perm(s0, s1, 0x5432u) : s0;
                            const uint16_t dbits = (o_d & 2) ? static_cast<uint16_t>(dwv >> 16) : static_cast<uint16_t>(dwv & 0xffffu);
                            half dh;
                            memcpy(&dh, &dbits, sizeof(dh));
                            d[u] = __half2float(dh);
                            q[u] = q5_0_quads(qsw, qh, g);
                        }
                    } else {
#pragma unroll
                        for (int u = 0; u < U; ++u) {
                            const int b = ib + 4 * u + bo;
                            if (b < tn) {
                                const uint16_t * p = reinterpret_cast<const uint16_t *>(xr + b);
                                half dh;
                                memcpy(&dh, p, sizeof(dh));
                                d[u] = __half2float(dh);
                                const uint32_t qh  = static_cast<uint32_t>(p[1]) | (static_cast<uint32_t>(p[2]) << 16);
                                const uint32_t qsw = static_cast<uint32_t>(p[3 + 2 * (g & 3)]) | (static_cast<uint32_t>(p[4 + 2 * (g & 3)]) << 16);
                                q[u] = q5_0_quads(qsw, qh, g);
                            } else {
                                d[u] = 0.0f;
                                q[u] = 0u;
                            }
                        }
                    }
#pragma unroll
                    for (int u = 0; u < U; ++u) {
                        const int b = ib + 4 * u + bo;
                        if (b < tn) {
                            acc[r] = fmaf(d[u], q5_0_dot4_bias(q[u], ys[b * (QK5_0 / 4) + g]), acc[r]);
                        }
                    }
                }
            }
        }
    }

    float * dst_base = dst + blockIdx.z * stride_sample_dst + blockIdx.y * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset, WARP_SIZE);
        }
        if (lane == 0 && row0 + r < nrows) {
            dst_base[row0 + r] = v;
        }
    }
}

template <bool SHFL>
static void launch_mmvk_q5_0_ws(
        const block_q5_0 * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = 16;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q5_0_kepler_ws<SHFL ? 2 : 4, SHFL>, launch_params,   // SHFL at U 4 spilled 32 B
        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
}

static void launch_mmvk_q4_k_vk(
        const block_q4_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int rows_per_block = 16;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q4_K_kepler_vk,
        launch_params,
        src0_d, src1_d, dst_d,
        nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

static void launch_mmvk_q6_k_vk(
        const block_q6_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int rows_per_block = 16;
    const unsigned int nblocks_x = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, 4, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q6_K_kepler_vk,
        launch_params,
        src0_d, src1_d, dst_d,
        nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

template <int THREADS_PER_ROW, int ROWS_PER_GROUP, int WARPS_PER_BLOCK>
static void launch_mmvk_q4_k(
        const block_q4_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int groups_per_block = WARPS_PER_BLOCK * (WARP_SIZE / THREADS_PER_ROW);
    constexpr int rows_per_block   = groups_per_block * ROWS_PER_GROUP;
    const unsigned int nblocks_x   = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, WARPS_PER_BLOCK, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q4_K_kepler<THREADS_PER_ROW, ROWS_PER_GROUP, WARPS_PER_BLOCK>,
        launch_params,
        src0_d, src1_d, dst_d,
        nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

template <int THREADS_PER_ROW, int ROWS_PER_GROUP, int WARPS_PER_BLOCK>
static void launch_mmvk_q6_k(
        const block_q6_K * src0_d, const float * src1_d, float * dst_d,
        const int nb, const int nrows, const int64_t stride_row_x,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst,
        const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    constexpr int groups_per_block = WARPS_PER_BLOCK * (WARP_SIZE / THREADS_PER_ROW);
    constexpr int rows_per_block   = groups_per_block * ROWS_PER_GROUP;
    const unsigned int nblocks_x   = static_cast<unsigned int>((static_cast<int64_t>(nrows) + rows_per_block - 1) / rows_per_block);

    const dim3 block_nums(nblocks_x, static_cast<unsigned int>(ne2), static_cast<unsigned int>(ne3));
    const dim3 block_dims(WARP_SIZE, WARPS_PER_BLOCK, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(
        mul_mat_vec_q6_K_kepler<THREADS_PER_ROW, ROWS_PER_GROUP, WARPS_PER_BLOCK>,
        launch_params,
        src0_d, src1_d, dst_d,
        nb, nrows, stride_row_x,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Q4_K `sy` keeps the whole y row (nb*256 floats) plus 8 sub-block sums per block in shared memory; sm_30 gives a
// block at most 48 KB, so rows longer than nb = 45 (k = 11520) cannot use it.
static constexpr size_t MMVK_SY_SMEM_LIMIT = 48 * 1024;
static void mmvk_warn_fallback_once(const char * env_name, const char * requested, const char * used, const int nb) {
    static std::atomic<bool> warned{false};   // only Q4_K `sy` uses this today
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: %s=%s cannot run this shape (nb=%d blocks per row needs more than 48 KB of shared memory); "
                      "using %s for such shapes\n", __func__, env_name, requested, nb, used);
    }
}

// An explicitly requested Q6_K variant that cannot run for a shape falls back to another kernel; say so
// once per requested variant, so a benchmark of `d32`/`smem`/`nb4` that actually ran `vk` is visible.
static void mmvk_q6_k_warn_fallback(const char * requested, const char * used, const int nb) {
    static std::atomic<bool> warned_nb4{false}, warned_smem{false}, warned_d32{false};
    std::atomic<bool> & warned = requested[0] == 'n' ? warned_nb4 : (requested[0] == 's' ? warned_smem : warned_d32);
    if (!warned.exchange(true)) {
        GGML_LOG_WARN("%s: GGML_CUDA_KEPLER_MMV_Q6K=%s cannot run this shape (nb=%d blocks per row or "
                      "unaligned rows); using %s for such shapes\n", __func__, requested, nb, used);
    }
}
#endif

// GGML_CUDA_KEPLER_MMV: unset or any other value = on; "0", "off", "false", "no" (any case) = off.
// Read once per process.
static bool ggml_cuda_kepler_mmv_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_MMV");
        if (env == nullptr) {
            return true;
        }
        std::string v(env);
        for (char & c : v) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return !(v == "0" || v == "off" || v == "false" || v == "no");
    }();
    return enabled;
}

template <typename T>
static __global__ void mul_mat_vec_kepler_attn_kq(
        const T * x, const float * y, float * dst,
        const int k, const int rows,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb12, const int64_t nb13, const int64_t nb02_dst, const int64_t nb03_dst,
        const int64_t dst_nb0, const int head_ratio, const int batch_ratio) {
    __shared__ float query[256];
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += 128) {
        const char * query_ptr = reinterpret_cast<const char *>(y)
            + (int64_t) blockIdx.y * nb12 + (int64_t) blockIdx.z * nb13 + (int64_t) i * sizeof(float);
        query[i] = *reinterpret_cast<const float *>(query_ptr);
    }
    __syncthreads();

    const int lane = tid & 15;
    const int row = blockIdx.x * 8 + tid / 16;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    float sum = 0.0f;
    if (row < rows) {
        const char * row_ptr = reinterpret_cast<const char *>(x)
            + (int64_t) row * nb01 + (int64_t) (head / head_ratio) * nb02 + (int64_t) (batch / batch_ratio) * nb03;
        for (int i = lane; i < k; i += 16) {
            const T value = *reinterpret_cast<const T *>(row_ptr + (int64_t) i * nb00);
            sum = fmaf(static_cast<float>(value), query[i], sum);
        }
    }
    for (int offset = 8; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 16);
    }
    if (lane == 0 && row < rows) {
        char * out = reinterpret_cast<char *>(dst) + (int64_t) head * nb02_dst + (int64_t) batch * nb03_dst;
        *reinterpret_cast<float *>(out + (int64_t) row * dst_nb0) = sum;
    }
}

template <typename T>
static __global__ void mul_mat_vec_kepler_attn_kqv(
        const T * x, const float * y, float * dst,
        const int k,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb12, const int64_t nb13, const int64_t nb02_dst, const int64_t nb03_dst,
        const int64_t dst_nb0, const int head_ratio, const int batch_ratio) {
    __shared__ float partial[4];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const char * x_head = reinterpret_cast<const char *>(x) + (int64_t) (head / head_ratio) * nb02 + (int64_t) (batch / batch_ratio) * nb03;
    const float * q = reinterpret_cast<const float *>(reinterpret_cast<const char *>(y)
        + (int64_t) head * nb12 + (int64_t) batch * nb13);
    float sum = 0.0f;
    for (int i = tid; i < k; i += 128) {
        const T value = *reinterpret_cast<const T *>(x_head + (int64_t) i * nb00 + (int64_t) row * nb01);
        sum = fmaf(static_cast<float>(value), q[i], sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        partial[warp] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        const float total = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        char * out = reinterpret_cast<char *>(dst) + (int64_t) head * nb02_dst + (int64_t) batch * nb03_dst;
        *reinterpret_cast<float *>(out + (int64_t) row * dst_nb0) = total;
    }
}

template <typename T>
static __global__ void mul_mat_vec_kepler_attn_kqv_gqa2(
        const T * x, const float * y, float * dst,
        const int k,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb12, const int64_t nb13, const int64_t nb02_dst, const int64_t nb03_dst,
        const int64_t dst_nb0, const int batch_ratio) {
    __shared__ float partial[2][4];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x;
    const int head0 = blockIdx.y * 2;
    const int batch = blockIdx.z;
    const char * x_head = reinterpret_cast<const char *>(x)
        + (int64_t) blockIdx.y * nb02 + (int64_t) (batch / batch_ratio) * nb03;
    const char * y_base = reinterpret_cast<const char *>(y) + (int64_t) batch * nb13;
    const float * q0 = reinterpret_cast<const float *>(y_base + (int64_t) head0 * nb12);
    const float * q1 = reinterpret_cast<const float *>(y_base + (int64_t) (head0 + 1) * nb12);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int i = tid; i < k; i += 128) {
        const T value = *reinterpret_cast<const T *>(x_head + (int64_t) i * nb00 + (int64_t) row * nb01);
        const float v = static_cast<float>(value);
        sum0 = fmaf(v, q0[i], sum0);
        sum1 = fmaf(v, q1[i], sum1);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (lane == 0) {
        partial[0][warp] = sum0;
        partial[1][warp] = sum1;
    }
    __syncthreads();
    if (tid == 0) {
        char * out = reinterpret_cast<char *>(dst) + (int64_t) batch * nb03_dst;
        const float total0 = (partial[0][0] + partial[0][1]) + (partial[0][2] + partial[0][3]);
        const float total1 = (partial[1][0] + partial[1][1]) + (partial[1][2] + partial[1][3]);
        *reinterpret_cast<float *>(out + (int64_t) row * dst_nb0 + (int64_t) head0 * nb02_dst) = total0;
        *reinterpret_cast<float *>(out + (int64_t) row * dst_nb0 + (int64_t) (head0 + 1) * nb02_dst) = total1;
    }
}

static bool ggml_cuda_kepler_attn_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_ATTN");
        if (env == nullptr) {
            return true;
        }
        std::string value(env);
        for (char & c : value) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return !(value == "0" || value == "off" || value == "false" || value == "no");
    }();
    return enabled;
}

static bool ggml_cuda_mmvk_shape_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_Q8_0 &&
            src0->type != GGML_TYPE_Q5_0) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];

    if (ne11 != 1 || ne1 != 1 || ne10 != ne00 || ne0 != ne01 || ne2 != ne12 || ne3 != ne13) {
        return false;
    }
    const int64_t block_size = ggml_blck_size(src0->type);
    if (ne00 <= 0 || (ne00 % block_size) != 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0 || ne2 <= 0 || ne3 <= 0) {
        return false;
    }
    if ((ne2 % ne02) != 0 || (ne3 % ne03) != 0) {
        return false;
    }
    // sm_30 gridDim.y and gridDim.z hardware limits
    if (ne2 > 65535 || ne3 > 65535 || ne01 > INT32_MAX || ne00 > INT32_MAX) {
        return false;
    }

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_f32  = sizeof(float);

    if (src0->nb[0] != ts_src0 || src1->nb[0] != ts_f32 || dst->nb[0] != ts_f32) {
        return false;
    }
    if ((src0->nb[1] % ts_src0) != 0 || (src0->nb[2] % ts_src0) != 0 || (src0->nb[3] % ts_src0) != 0) {
        return false;
    }
    if ((src1->nb[2] % ts_f32) != 0 || (src1->nb[3] % ts_f32) != 0) {
        return false;
    }
    if ((dst->nb[2] % ts_f32) != 0 || (dst->nb[3] % ts_f32) != 0) {
        return false;
    }

    // float4 loads on src1 require 16-byte alignment for base and active batch strides
    if ((reinterpret_cast<uintptr_t>(src1->data) % 16) != 0) {
        return false;
    }
    if ((ne12 > 1 && (src1->nb[2] % 16) != 0) || (ne13 > 1 && (src1->nb[3] % 16) != 0)) {
        return false;
    }

    // Q4_K uses uint4 (16-byte aligned; sizeof(block_q4_K) == 144 is multiple of 16).
    // Q6_K, Q8_0 and Q5_0 use 2-byte-aligned block loads.
    const size_t req_align_src0 = (src0->type == GGML_TYPE_Q4_K) ? 16 : 2;
    if ((reinterpret_cast<uintptr_t>(src0->data) % req_align_src0) != 0) {
        return false;
    }

    return true;
}

bool ggml_cuda_should_use_mmvk(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
#else
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_KEPLER || cc >= 500) {
        return false;
    }
    if (!ggml_cuda_kepler_mmv_enabled()) {
        return false;
    }
    return ggml_cuda_mmvk_shape_supported(src0, src1, dst);
#endif
}

bool ggml_cuda_mul_mat_vec_kepler(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
#else
    // Exported entry point: re-check the device here too (not only in ggml_cuda_should_use_mmvk), so a
    // new caller can never launch these sm_30 kernels on another architecture.
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_KEPLER || cc >= 500) {
        return false;
    }
    if (!ggml_cuda_mmvk_shape_supported(src0, src1, dst)) {
        return false;
    }

    const int block_size = ggml_blck_size(src0->type);
    const int nb    = static_cast<int>(src0->ne[0] / block_size);
    const int nrows = static_cast<int>(src0->ne[1]);

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_f32  = sizeof(float);

    const int64_t stride_row_x       = static_cast<int64_t>(src0->nb[1] / ts_src0);
    const int64_t stride_channel_x   = static_cast<int64_t>(src0->nb[2] / ts_src0);
    const int64_t stride_sample_x    = static_cast<int64_t>(src0->nb[3] / ts_src0);

    const int64_t stride_channel_y   = static_cast<int64_t>(src1->nb[2] / ts_f32);
    const int64_t stride_sample_y    = static_cast<int64_t>(src1->nb[3] / ts_f32);

    const int64_t stride_channel_dst = static_cast<int64_t>(dst->nb[2] / ts_f32);
    const int64_t stride_sample_dst  = static_cast<int64_t>(dst->nb[3] / ts_f32);

    const uint3 channel_ratio = init_fastdiv_values(static_cast<uint64_t>(dst->ne[2] / src0->ne[2]));
    const uint3 sample_ratio  = init_fastdiv_values(static_cast<uint64_t>(dst->ne[3] / src0->ne[3]));

    const float * src1_d = static_cast<const float *>(src1->data);
    float       * dst_d  = static_cast<float *>(dst->data);
    cudaStream_t  stream = ctx.stream();

    if (src0->type == GGML_TYPE_Q4_K) {
        const block_q4_K * src0_d = static_cast<const block_q4_K *>(src0->data);
        const bool q4_sy_fits = static_cast<size_t>(nb) * (QK_K + 8) * sizeof(float) <= MMVK_SY_SMEM_LIMIT;
        switch (ggml_cuda_kepler_mmv_q4k_variant()) {
            case MMVK_Q4_K_32x1x4:
                launch_mmvk_q4_k<32, 1, 4>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_32x1x8:
                launch_mmvk_q4_k<32, 1, 8>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_32x2x4:
                launch_mmvk_q4_k<32, 2, 4>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_8x1x4:
                launch_mmvk_q4_k<8, 1, 4>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_VK:
                launch_mmvk_q4_k_vk(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_SWIZ:
                launch_mmvk_q4_k_swiz<2, 8>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_C8:
                launch_mmvk_q4_k_c8(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_SY:
                if (q4_sy_fits) {
                    launch_mmvk_q4_k_sy<16>(
                        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                        dst->ne[2], dst->ne[3], stream);
                    break;
                }
                mmvk_warn_fallback_once("GGML_CUDA_KEPLER_MMV_Q4K", "sy", "vk2", nb);
                launch_mmvk_q4_k_vk2(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q4_K_AUTO:
                // `sy` (y in shared memory) measured faster than vk2 for long rows (1024x4608, nb 18: 98.9 vs
                // 109.7 us) and for 512x1024; slower for the other k = 1024 shapes and the 65536-row head.
                if (q4_sy_fits && (nb >= 16 || (nb == 4 && nrows == 512))) {
                    launch_mmvk_q4_k_sy<16>(
                        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                        dst->ne[2], dst->ne[3], stream);
                } else {
                    launch_mmvk_q4_k_vk2(
                        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                        dst->ne[2], dst->ne[3], stream);
                }
                break;
            case MMVK_Q4_K_VK2:
            default:
                launch_mmvk_q4_k_vk2(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
        }
        return true;
    }

    if (src0->type == GGML_TYPE_Q6_K) {
        const block_q6_K * src0_d = static_cast<const block_q6_K *>(src0->data);
        const bool q6_nb4_ok = (nb == 4)
            && (stride_row_x == 4)
            && ((reinterpret_cast<uintptr_t>(src0_d) & 3u) == 0u)
            && (src0->ne[2] == 1 || (stride_channel_x & 1) == 0)
            && (src0->ne[3] == 1 || (stride_sample_x & 1) == 0);
        // Gate for the kernels that walk blocks in (even, odd) pairs, i.e. both `smem` and `d32`: an even
        // number of blocks per row, and row/channel/sample starts that keep the pairs 4-byte aligned.
        // Shapes that fail it go to the `vk` kernel, which handles any nb.
        const bool q6_even_pairs_ok = ((nb & 1) == 0)
            && ((stride_row_x & 1) == 0)
            && ((reinterpret_cast<uintptr_t>(src0_d) & 3u) == 0u)
            && (src0->ne[2] == 1 || (stride_channel_x & 1) == 0)
            && (src0->ne[3] == 1 || (stride_sample_x & 1) == 0);

        switch (ggml_cuda_kepler_mmv_q6k_variant()) {
            case MMVK_Q6_K_32x2x4:
                launch_mmvk_q6_k<32, 2, 4>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q6_K_32x1x4:
                launch_mmvk_q6_k<32, 1, 4>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q6_K_32x1x8:
                launch_mmvk_q6_k<32, 1, 8>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q6_K_16x1x4:
                launch_mmvk_q6_k<16, 1, 4>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q6_K_VK:
                launch_mmvk_q6_k_vk(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q6_K_NB4:
                if (q6_nb4_ok) {
                    launch_mmvk_q6_k_nb4<8>(
                        src0_d, src1_d, dst_d, nrows,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                        dst->ne[2], dst->ne[3], stream);
                    break;
                }
                mmvk_q6_k_warn_fallback("nb4", q6_even_pairs_ok ? "smem" : "vk", nb);
                [[fallthrough]];
            case MMVK_Q6_K_SMEM:
                if (q6_even_pairs_ok) {
                    launch_mmvk_q6_k_smem(
                        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                        dst->ne[2], dst->ne[3], stream);
                    break;
                }
                mmvk_q6_k_warn_fallback("smem", "vk", nb);
                launch_mmvk_q6_k_vk(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
            case MMVK_Q6_K_D32:
            case MMVK_Q6_K_AUTO:
            default:
                if (!q6_even_pairs_ok && ggml_cuda_kepler_mmv_q6k_variant() == MMVK_Q6_K_D32) {
                    mmvk_q6_k_warn_fallback("d32", "vk", nb);
                }
                if (q6_even_pairs_ok) {
                    launch_mmvk_q6_k_d32(
                        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                        dst->ne[2], dst->ne[3], stream);
                } else {
                    launch_mmvk_q6_k_vk(
                        src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                        dst->ne[2], dst->ne[3], stream);
                }
                break;
        }
        return true;
    }

    if (src0->type == GGML_TYPE_Q5_0) {
        const block_q5_0 * src0_d = static_cast<const block_q5_0 *>(src0->data);
        const bool q5_even_pairs_ok = (nb > 0)
            && ((nb & 1) == 0)
            && ((stride_row_x & 1) == 0)
            && ((reinterpret_cast<uintptr_t>(src0_d) & 3u) == 0u)
            && (src0->ne[2] == 1 || (stride_channel_x & 1) == 0)
            && (src0->ne[3] == 1 || (stride_sample_x & 1) == 0);
        static const bool q5_noshfl = std::getenv("GGML_CUDA_KEPLER_MMV_Q5_NOSHFL") != nullptr;
        if (q5_even_pairs_ok && !q5_noshfl) {
            launch_mmvk_q5_0_ws<true>(
                src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                dst->ne[2], dst->ne[3], stream);
        } else {
            launch_mmvk_q5_0_ws<false>(
                src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                dst->ne[2], dst->ne[3], stream);
        }
        return true;
    }

    if (src0->type == GGML_TYPE_Q8_0) {
        const block_q8_0 * src0_d = static_cast<const block_q8_0 *>(src0->data);
        const bool q8_even_pairs_ok = (nb > 0)
            && ((nb & 1) == 0)
            && ((stride_row_x & 1) == 0)
            && ((reinterpret_cast<uintptr_t>(src0_d) & 3u) == 0u)
            && (src0->ne[2] == 1 || (stride_channel_x & 1) == 0)
            && (src0->ne[3] == 1 || (stride_sample_x & 1) == 0);
        const mmvk_q8_0_variant variant = ggml_cuda_kepler_mmv_q8_variant();
        const bool use_pair = (variant == MMVK_Q8_0_PAIR) && q8_even_pairs_ok;
        if (variant == MMVK_Q8_0_AUTO || variant == MMVK_Q8_0_WS) {
            // GGML_CUDA_KEPLER_MMV_Q8_CFG (sweep, 2026-11-21): U (4-block steps in flight) and register cap
            // cfg 0 (default): shuffle-gather on 4-byte-aligned rows with even nb, 16-bit loads otherwise.
            // cfg 1: 16-bit loads everywhere; cfg 2/3: shuffle-gather with U 2 / 8 (aligned rows only).
#define Q8WS(U_, MINB_, SH_) launch_mmvk_q8_0_ws<U_, MINB_, SH_>(src0_d, src1_d, dst_d, nb, nrows, stride_row_x, \
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, \
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, dst->ne[2], dst->ne[3], stream)
            static const int q8cfg = []() {
                const char * c = std::getenv("GGML_CUDA_KEPLER_MMV_Q8_CFG");
                return c ? std::atoi(c) : 0;
            }();
            if (!q8_even_pairs_ok || q8cfg == 1) {
                Q8WS(4, 1, false);
            } else if (q8cfg == 2) {
                Q8WS(2, 1, true);
            } else if (q8cfg == 3) {
                Q8WS(8, 1, true);
            } else {
                Q8WS(4, 1, true);
            }
#undef Q8WS
        } else if (use_pair) {
            launch_mmvk_q8_0<true>(
                src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                dst->ne[2], dst->ne[3], stream);
        } else {
            launch_mmvk_q8_0<false>(
                src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
                dst->ne[2], dst->ne[3], stream);
        }
        return true;
    }

    return false;
#endif
}

bool ggml_cuda_mul_mat_vec_kepler_attn(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
#else
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_KEPLER || cc >= 500 || !ggml_cuda_kepler_attn_enabled()) {
        return false;
    }
    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32)
            || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32
            || src1->ne[1] != 1 || src0->ne[0] != src1->ne[0]
            || src0->ne[1] != dst->ne[0] || dst->ne[1] != 1
            || src1->ne[2] != dst->ne[2] || src1->ne[3] != dst->ne[3]
            || src0->ne[2] <= 0 || src0->ne[3] <= 0
            || dst->ne[2] % src0->ne[2] != 0 || dst->ne[3] % src0->ne[3] != 0
            || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    // r13 (review r9): attention operands only. Model weights (e.g. a vision projector with a 64..256 dim)
    // live in a weights buffer and go to the normal matvec path.
    if (src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    // r13: non-empty, 32-bit row/column counts (with room for the (rows + 7) / 8 grid), sm_30 grid y/z
    // limit, element-aligned pointers and strides (the kernels index through typed element pointers).
    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src0->ne[0] > INT32_MAX - 8 || src0->ne[1] > INT32_MAX - 8
            || dst->ne[2] > 65535 || dst->ne[3] > 65535) {
        return false;
    }
    {
        const size_t ts = ggml_type_size(src0->type);
        if ((uintptr_t) src0->data % ts != 0 || (uintptr_t) src1->data % sizeof(float) != 0
                || (uintptr_t) dst->data % sizeof(float) != 0) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (src0->nb[i] % ts != 0 || src1->nb[i] % sizeof(float) != 0 || dst->nb[i] % sizeof(float) != 0) {
                return false;
            }
        }
    }

    const bool dim0_head = src0->ne[0] == 64 || src0->ne[0] == 80 || src0->ne[0] == 96 || src0->ne[0] == 128 || src0->ne[0] == 256;
    const bool dim1_head = src0->ne[1] == 64 || src0->ne[1] == 80 || src0->ne[1] == 96 || src0->ne[1] == 128 || src0->ne[1] == 256;
    bool kq = dim0_head && !dim1_head;
    bool kqv = dim1_head && !dim0_head;
    if (dim0_head && dim1_head) {
        kq = src0->nb[0] < src0->nb[1];
        kqv = src0->nb[1] < src0->nb[0];
    }
    if (!kq && !kqv) {
        return false;
    }
    const int rows = static_cast<int>(src0->ne[1]);
    const int k = static_cast<int>(src0->ne[0]);
    const int head_ratio = static_cast<int>(dst->ne[2] / src0->ne[2]);
    const int batch_ratio = static_cast<int>(dst->ne[3] / src0->ne[3]);
    const bool grouped_kqv = kqv && head_ratio == 2;
    const dim3 grid(kq ? (rows + 7) / 8 : rows,
                    static_cast<unsigned int>(grouped_kqv ? dst->ne[2] / 2 : dst->ne[2]),
                    static_cast<unsigned int>(dst->ne[3]));
    const dim3 block(128, 1, 1);
    const size_t shared_bytes = 0; // r13: the attention kernels declare static __shared__ arrays
    const ggml_cuda_kernel_launch_params launch_params(grid, block, shared_bytes, ctx.stream());
    const size_t nb02_dst = dst->nb[2];
    const size_t nb03_dst = dst->nb[3];

    if (src0->type == GGML_TYPE_F16) {
        const half * x = static_cast<const half *>(src0->data);
        if (kq) {
            ggml_cuda_kernel_launch(mul_mat_vec_kepler_attn_kq<half>, launch_params, x,
                static_cast<const float *>(src1->data), static_cast<float *>(dst->data), k, rows,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                nb02_dst, nb03_dst, dst->nb[0], head_ratio, batch_ratio);
        } else if (grouped_kqv) {
            ggml_cuda_kernel_launch(mul_mat_vec_kepler_attn_kqv_gqa2<half>, launch_params, x,
                static_cast<const float *>(src1->data), static_cast<float *>(dst->data), k,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                nb02_dst, nb03_dst, dst->nb[0], batch_ratio);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_kepler_attn_kqv<half>, launch_params, x,
                static_cast<const float *>(src1->data), static_cast<float *>(dst->data), k,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                nb02_dst, nb03_dst, dst->nb[0], head_ratio, batch_ratio);
        }
    } else {
        const float * x = static_cast<const float *>(src0->data);
        if (kq) {
            ggml_cuda_kernel_launch(mul_mat_vec_kepler_attn_kq<float>, launch_params, x,
                static_cast<const float *>(src1->data), static_cast<float *>(dst->data), k, rows,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                nb02_dst, nb03_dst, dst->nb[0], head_ratio, batch_ratio);
        } else if (grouped_kqv) {
            ggml_cuda_kernel_launch(mul_mat_vec_kepler_attn_kqv_gqa2<float>, launch_params, x,
                static_cast<const float *>(src1->data), static_cast<float *>(dst->data), k,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                nb02_dst, nb03_dst, dst->nb[0], batch_ratio);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_kepler_attn_kqv<float>, launch_params, x,
                static_cast<const float *>(src1->data), static_cast<float *>(dst->data), k,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3],
                nb02_dst, nb03_dst, dst->nb[0], head_ratio, batch_ratio);
        }
    }
    return true;
#endif
}
