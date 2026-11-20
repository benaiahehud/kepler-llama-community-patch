#include "mmvk.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static_assert(sizeof(block_q4_K) == 144, "unexpected block_q4_K size");
static_assert(offsetof(block_q4_K, scales) == 4, "unexpected block_q4_K scales offset");
static_assert(offsetof(block_q4_K, qs) == 16, "unexpected block_q4_K qs offset");

static_assert(sizeof(block_q6_K) == 210, "unexpected block_q6_K size");
static_assert(offsetof(block_q6_K, ql) == 0, "unexpected block_q6_K ql offset");
static_assert(offsetof(block_q6_K, qh) == 128, "unexpected block_q6_K qh offset");
static_assert(offsetof(block_q6_K, scales) == 192, "unexpected block_q6_K scales offset");
static_assert(offsetof(block_q6_K, d) == 208, "unexpected block_q6_K d offset");

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Selectable Kepler (sm_30) launch geometries "<THREADS_PER_ROW>x<ROWS_PER_GROUP>x<WARPS_PER_BLOCK>":
// - GGML_CUDA_KEPLER_MMV_Q4K: "32x1x4" (default), "32x1x8", "32x2x4", "8x1x4"
// - GGML_CUDA_KEPLER_MMV_Q6K: "32x2x4" (default), "32x1x4", "32x1x8", "16x1x4"
enum mmvk_q4_k_variant {
    MMVK_Q4_K_32x1x4 = 0,
    MMVK_Q4_K_32x1x8,
    MMVK_Q4_K_32x2x4,
    MMVK_Q4_K_8x1x4,
};

enum mmvk_q6_k_variant {
    MMVK_Q6_K_32x2x4 = 0,
    MMVK_Q6_K_32x1x4,
    MMVK_Q6_K_32x1x8,
    MMVK_Q6_K_16x1x4,
};

static mmvk_q4_k_variant ggml_cuda_kepler_mmv_q4k_variant() {
    static const mmvk_q4_k_variant variant = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_MMV_Q4K");
        if (env == nullptr || env[0] == '\0') {
            return MMVK_Q4_K_32x1x4;
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
        return MMVK_Q4_K_32x1x4;
    }();
    return variant;
}

static mmvk_q6_k_variant ggml_cuda_kepler_mmv_q6k_variant() {
    static const mmvk_q6_k_variant variant = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_MMV_Q6K");
        if (env == nullptr || env[0] == '\0') {
            return MMVK_Q6_K_32x2x4;
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
        return MMVK_Q6_K_32x2x4;
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

static __device__ __forceinline__ float dot4_u6_bias(const uint32_t q, const float4 y) {
    // q holds 4 unsigned 6-bit quants in bytes 0..3 (bits 6..7 of each byte are 0).
    // Bias subtraction also folds the -32 zero-point.
    float sum = __fadd_rn(__uint_as_float((q >> 24)         | 0x4B000000u), -8388640.0f) * y.w;
    sum = fmaf(__fadd_rn(__uint_as_float((q & 0x003F0000u) | 0x43000000u), -160.0f),     y.z, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((q & 0x00003F00u) | 0x47000000u), -32800.0f),   y.y, sum);
    sum = fmaf(__fadd_rn(__uint_as_float((q & 0x0000003Fu) | 0x4B000000u), -8388640.0f), y.x, sum);
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

static bool ggml_cuda_kepler_mmv_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_CUDA_KEPLER_MMV");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ggml_cuda_mmvk_shape_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K) {
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
    if (ne00 <= 0 || (ne00 % QK_K) != 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0 || ne2 <= 0 || ne3 <= 0) {
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
    // Q6_K uses 2-byte-aligned block loads (sizeof(block_q6_K) == 210 is multiple of 2).
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
    if (!ggml_cuda_mmvk_shape_supported(src0, src1, dst)) {
        return false;
    }

    const int nb    = static_cast<int>(src0->ne[0] / QK_K);
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
        switch (ggml_cuda_kepler_mmv_q4k_variant()) {
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
            case MMVK_Q4_K_32x1x4:
            default:
                launch_mmvk_q4_k<32, 1, 4>(
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
        switch (ggml_cuda_kepler_mmv_q6k_variant()) {
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
            case MMVK_Q6_K_32x2x4:
            default:
                launch_mmvk_q6_k<32, 2, 4>(
                    src0_d, src1_d, dst_d, nb, nrows, stride_row_x,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst,
                    dst->ne[2], dst->ne[3], stream);
                break;
        }
        return true;
    }

    return false;
#endif
}
