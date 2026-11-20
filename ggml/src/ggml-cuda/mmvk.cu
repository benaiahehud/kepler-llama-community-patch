#include "mmvk.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Tunable launch geometry for Kepler (sm_30).
// THREADS_PER_ROW must be a power of 2 <= WARP_SIZE so reduction uses warp shuffles only.
static constexpr int MMVK_Q4_K_THREADS_PER_ROW = 32; // must be multiple of 8 (8 threads per super-block)
static constexpr int MMVK_Q4_K_ROWS_PER_GROUP  = 2;
static constexpr int MMVK_Q4_K_WARPS_PER_BLOCK = 4;

static constexpr int MMVK_Q6_K_THREADS_PER_ROW = 32; // must be multiple of 16 (16 threads per super-block)
static constexpr int MMVK_Q6_K_ROWS_PER_GROUP  = 2;
static constexpr int MMVK_Q6_K_WARPS_PER_BLOCK = 4;

static_assert(sizeof(block_q4_K) == 144, "unexpected block_q4_K size");
static_assert(offsetof(block_q4_K, scales) == 4, "unexpected block_q4_K scales offset");
static_assert(offsetof(block_q4_K, qs) == 16, "unexpected block_q4_K qs offset");

static_assert(sizeof(block_q6_K) == 210, "unexpected block_q6_K size");
static_assert(offsetof(block_q6_K, ql) == 0, "unexpected block_q6_K ql offset");
static_assert(offsetof(block_q6_K, qh) == 128, "unexpected block_q6_K qh offset");
static_assert(offsetof(block_q6_K, scales) == 192, "unexpected block_q6_K scales offset");
static_assert(offsetof(block_q6_K, d) == 208, "unexpected block_q6_K d offset");

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static constexpr __host__ __device__ int mmvk_min_blocks_per_sm(int warps_per_block) {
    // 1024 threads/SMX caps register target at 63/thread on sm_30 without forcing spills.
    return (1024 / (warps_per_block * WARP_SIZE)) > 0 ? (1024 / (warps_per_block * WARP_SIZE)) : 1;
}

static __device__ __forceinline__ void q4_k_dot8_bias(
        const uint32_t u, const float4 y_lo, const float4 y_hi,
        float & sum_q0, float & sum_q1) {
    // One shift extracts the upper 16 bits for all 8 weights.
    // Exponents 0x4B/0x47 place bits 0..3 / 8..11 at integer scale; 0x49/0x45 fold the 1/16 of high nibbles.
    const uint32_t u_hi = u >> 16;

    const float ql0 = __fadd_rn(__uint_as_float((u    & 0x000Fu) | 0x4B000000u), -8388608.0f);
    const float ql1 = __fadd_rn(__uint_as_float((u    & 0x0F00u) | 0x47000000u), -32768.0f);
    const float ql2 = __fadd_rn(__uint_as_float((u_hi & 0x000Fu) | 0x4B000000u), -8388608.0f);
    const float ql3 = __fadd_rn(__uint_as_float((u_hi & 0x0F00u) | 0x47000000u), -32768.0f);

    sum_q0 = fmaf(ql0, y_lo.x, fmaf(ql1, y_lo.y, fmaf(ql2, y_lo.z, fmaf(ql3, y_lo.w, sum_q0))));

    const float qh0 = __fadd_rn(__uint_as_float((u    & 0x00F0u) | 0x49000000u), -524288.0f);
    const float qh1 = __fadd_rn(__uint_as_float((u    & 0xF000u) | 0x45000000u), -2048.0f);
    const float qh2 = __fadd_rn(__uint_as_float((u_hi & 0x00F0u) | 0x49000000u), -524288.0f);
    const float qh3 = __fadd_rn(__uint_as_float((u_hi & 0xF000u) | 0x45000000u), -2048.0f);

    sum_q1 = fmaf(qh0, y_hi.x, fmaf(qh1, y_hi.y, fmaf(qh2, y_hi.z, fmaf(qh3, y_hi.w, sum_q1))));
}

template <int THREADS_PER_ROW, int ROWS_PER_GROUP, int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE, mmvk_min_blocks_per_sm(WARPS_PER_BLOCK))
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

    constexpr int GROUPS_PER_WARP  = WARP_SIZE / THREADS_PER_ROW;
    constexpr int GROUPS_PER_BLOCK = WARPS_PER_BLOCK * GROUPS_PER_WARP;
    constexpr int ROWS_PER_BLOCK   = GROUPS_PER_BLOCK * ROWS_PER_GROUP;
    constexpr int BLOCKS_PER_ITER  = THREADS_PER_ROW / 8;

    const int lane          = threadIdx.x;
    const int group_in_warp = lane / THREADS_PER_ROW;
    const int lane_in_group = lane % THREADS_PER_ROW;
    const int group_id      = threadIdx.y * GROUPS_PER_WARP + group_in_warp;
    const int row0          = blockIdx.x * ROWS_PER_BLOCK + group_id * ROWS_PER_GROUP;

    if constexpr (GROUPS_PER_WARP == 1) {
        if (row0 >= nrows) {
            return;
        }
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst,  sample_ratio);

    const block_q4_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + sample_dst * stride_sample_y + channel_dst * stride_channel_y;

    // 8 threads cooperate per block_q4_K: each loads one uint4 (16 bytes = 32 weights) of qs.
    const int blk_in_group = lane_in_group >> 3;
    const int sub          = lane_in_group & 7;
    const int chunk        = sub >> 1; // 0..3 -> 64-element chunk
    const int half_chunk   = sub & 1;  // 0..1 -> first or second 16 bytes of chunk

    const uint32_t sc_shift = static_cast<uint32_t>((chunk & 1) << 4);
    const uint32_t hi_mask  = static_cast<uint32_t>(-static_cast<int32_t>(chunk >= 2));

    const int y4_off_lo = 16 * chunk + 4 * half_chunk;
    const int y4_off_hi = y4_off_lo + 8;

    const block_q4_K * row_ptrs[ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        const int row = (row0 + r < nrows) ? (row0 + r) : 0;
        row_ptrs[r] = x_base + row * stride_row_x + blk_in_group;
    }

    const float4 * y4 = reinterpret_cast<const float4 *>(y_base + static_cast<int64_t>(blk_in_group) * QK_K);

    float acc[ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        acc[r] = 0.0f;
    }

    for (int ib = blk_in_group; ib < nb; ib += BLOCKS_PER_ITER) {
        uint4 qs_v[ROWS_PER_GROUP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r) {
            qs_v[r] = reinterpret_cast<const uint4 *>(row_ptrs[r]->qs)[sub];
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

        {
            const float4 y_lo = y4[y4_off_lo + 0];
            const float4 y_hi = y4[y4_off_hi + 0];
            sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
            sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                q4_k_dot8_bias(qs_v[r].x, y_lo, y_hi, sum_q0[r], sum_q1[r]);
            }
        }
        {
            const float4 y_lo = y4[y4_off_lo + 1];
            const float4 y_hi = y4[y4_off_hi + 1];
            sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
            sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                q4_k_dot8_bias(qs_v[r].y, y_lo, y_hi, sum_q0[r], sum_q1[r]);
            }
        }
        {
            const float4 y_lo = y4[y4_off_lo + 2];
            const float4 y_hi = y4[y4_off_hi + 2];
            sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
            sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                q4_k_dot8_bias(qs_v[r].z, y_lo, y_hi, sum_q0[r], sum_q1[r]);
            }
        }
        {
            const float4 y_lo = y4[y4_off_lo + 3];
            const float4 y_hi = y4[y4_off_hi + 3];
            sum_y0 += (y_lo.x + y_lo.y) + (y_lo.z + y_lo.w);
            sum_y1 += (y_hi.x + y_hi.y) + (y_hi.z + y_hi.w);
#pragma unroll
            for (int r = 0; r < ROWS_PER_GROUP; ++r) {
                q4_k_dot8_bias(qs_v[r].w, y_lo, y_hi, sum_q0[r], sum_q1[r]);
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r) {
            const uint4 hdr = *reinterpret_cast<const uint4 *>(row_ptrs[r]);

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

            const float sc0 = __fadd_rn(__uint_as_float((sc_pair & 0x003Fu) | 0x4B000000u), -8388608.0f);
            const float sc1 = __fadd_rn(__uint_as_float((sc_pair & 0x3F00u) | 0x47000000u), -32768.0f);
            const float m0  = __fadd_rn(__uint_as_float((m_pair  & 0x003Fu) | 0x4B000000u), -8388608.0f);
            const float m1  = __fadd_rn(__uint_as_float((m_pair  & 0x3F00u) | 0x47000000u), -32768.0f);

            const float d0   = dm.x * sc0;
            const float d1   = dm.x * sc1;
            const float min0 = dm.y * m0;
            const float min1 = dm.y * m1;

            acc[r] = fmaf(d0, sum_q0[r], fmaf(-min0, sum_y0, acc[r]));
            acc[r] = fmaf(d1, sum_q1[r], fmaf(-min1, sum_y1, acc[r]));

            row_ptrs[r] += BLOCKS_PER_ITER;
        }
        y4 += BLOCKS_PER_ITER * (QK_K / 4);
    }

    float * dst_base = dst + sample_dst * stride_sample_dst + channel_dst * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        const float sum = warp_reduce_sum<THREADS_PER_ROW>(acc[r]);
        if (lane_in_group == 0 && row0 + r < nrows) {
            dst_base[row0 + r] = sum;
        }
    }
}

static __device__ __forceinline__ float dot4_u6_bias(const uint32_t q, const float4 y) {
    // q holds 4 unsigned 6-bit quants in bytes 0..3 (bits 6..7 of each byte are 0).
    // Bias subtraction also folds the -32 zero-point.
    const float f0 = __fadd_rn(__uint_as_float((q & 0x0000003Fu) | 0x4B000000u), -8388640.0f);
    const float f1 = __fadd_rn(__uint_as_float((q & 0x00003F00u) | 0x47000000u), -32800.0f);
    const float f2 = __fadd_rn(__uint_as_float((q & 0x003F0000u) | 0x43000000u), -160.0f);
    const float f3 = __fadd_rn(__uint_as_float((q >> 24)         | 0x4B000000u), -8388640.0f);
    return fmaf(f0, y.x, fmaf(f1, y.y, fmaf(f2, y.z, f3 * y.w)));
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
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE, mmvk_min_blocks_per_sm(WARPS_PER_BLOCK))
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

    constexpr int GROUPS_PER_WARP  = WARP_SIZE / THREADS_PER_ROW;
    constexpr int GROUPS_PER_BLOCK = WARPS_PER_BLOCK * GROUPS_PER_WARP;
    constexpr int ROWS_PER_BLOCK   = GROUPS_PER_BLOCK * ROWS_PER_GROUP;
    constexpr int BLOCKS_PER_ITER  = THREADS_PER_ROW / 16;

    const int lane          = threadIdx.x;
    const int group_in_warp = lane / THREADS_PER_ROW;
    const int lane_in_group = lane % THREADS_PER_ROW;
    const int group_id      = threadIdx.y * GROUPS_PER_WARP + group_in_warp;
    const int row0          = blockIdx.x * ROWS_PER_BLOCK + group_id * ROWS_PER_GROUP;

    if constexpr (GROUPS_PER_WARP == 1) {
        if (row0 >= nrows) {
            return;
        }
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst,  sample_ratio);

    const block_q6_K * x_base = vx + sample_x * stride_sample_x + channel_x * stride_channel_x;
    const float      * y_base = vy + sample_dst * stride_sample_y + channel_dst * stride_channel_y;

    // 16 threads cooperate per block_q6_K: each handles 16 weights (4 quarters of 4 elements).
    const int blk_in_group = lane_in_group >> 4;
    const int sub          = lane_in_group & 15;
    const int h            = sub >> 3; // 0: elements 0..127, 1: elements 128..255
    const int i4           = sub & 7;  // 0..7 -> l0 = 4 * i4
    const int s_off        = 8 * h + (i4 >> 2);

    const block_q6_K * row_ptrs[ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        const int row = (row0 + r < nrows) ? (row0 + r) : 0;
        row_ptrs[r] = x_base + row * stride_row_x + blk_in_group;
    }

    const float4 * y4 = reinterpret_cast<const float4 *>(y_base + static_cast<int64_t>(blk_in_group) * QK_K + 128 * h + 4 * i4);

    float acc[ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        acc[r] = 0.0f;
    }

    // Uniform loop bound across the warp so all lanes participate in __shfl_up_sync inside load_u32_q6_k.
    for (int ib_base = 0; ib_base < nb; ib_base += BLOCKS_PER_ITER) {
        const bool valid = (ib_base + blk_in_group) < nb;

        uint32_t ql0[ROWS_PER_GROUP];
        uint32_t ql1[ROWS_PER_GROUP];
        uint32_t qh0[ROWS_PER_GROUP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r) {
            const block_q6_K * blk = row_ptrs[r];
            ql0[r] = load_u32_q6_k(blk->ql + 64 * h,      i4, valid);
            ql1[r] = load_u32_q6_k(blk->ql + 64 * h + 32, i4, valid);
            qh0[r] = load_u32_q6_k(blk->qh + 32 * h,      i4, valid);
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

#pragma unroll
        for (int r = 0; r < ROWS_PER_GROUP; ++r) {
            row_ptrs[r] += BLOCKS_PER_ITER;
        }
        y4 += BLOCKS_PER_ITER * (QK_K / 4);
    }

    float * dst_base = dst + sample_dst * stride_sample_dst + channel_dst * stride_channel_dst;
#pragma unroll
    for (int r = 0; r < ROWS_PER_GROUP; ++r) {
        const float sum = warp_reduce_sum<THREADS_PER_ROW>(acc[r]);
        if (lane_in_group == 0 && row0 + r < nrows) {
            dst_base[row0 + r] = sum;
        }
    }
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
        constexpr int groups_per_block = MMVK_Q4_K_WARPS_PER_BLOCK * (WARP_SIZE / MMVK_Q4_K_THREADS_PER_ROW);
        constexpr int rows_per_block   = groups_per_block * MMVK_Q4_K_ROWS_PER_GROUP;
        const int     nblocks_x        = (nrows + rows_per_block - 1) / rows_per_block;

        const dim3 block_nums(nblocks_x, static_cast<unsigned int>(dst->ne[2]), static_cast<unsigned int>(dst->ne[3]));
        const dim3 block_dims(WARP_SIZE, MMVK_Q4_K_WARPS_PER_BLOCK, 1);
        const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

        ggml_cuda_kernel_launch(
            mul_mat_vec_q4_K_kepler<MMVK_Q4_K_THREADS_PER_ROW, MMVK_Q4_K_ROWS_PER_GROUP, MMVK_Q4_K_WARPS_PER_BLOCK>,
            launch_params,
            static_cast<const block_q4_K *>(src0->data), src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
        return true;
    }

    if (src0->type == GGML_TYPE_Q6_K) {
        constexpr int groups_per_block = MMVK_Q6_K_WARPS_PER_BLOCK * (WARP_SIZE / MMVK_Q6_K_THREADS_PER_ROW);
        constexpr int rows_per_block   = groups_per_block * MMVK_Q6_K_ROWS_PER_GROUP;
        const int     nblocks_x        = (nrows + rows_per_block - 1) / rows_per_block;

        const dim3 block_nums(nblocks_x, static_cast<unsigned int>(dst->ne[2]), static_cast<unsigned int>(dst->ne[3]));
        const dim3 block_dims(WARP_SIZE, MMVK_Q6_K_WARPS_PER_BLOCK, 1);
        const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);

        ggml_cuda_kernel_launch(
            mul_mat_vec_q6_K_kepler<MMVK_Q6_K_THREADS_PER_ROW, MMVK_Q6_K_ROWS_PER_GROUP, MMVK_Q6_K_WARPS_PER_BLOCK>,
            launch_params,
            static_cast<const block_q6_K *>(src0->data), src1_d, dst_d,
            nb, nrows, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio,  stride_sample_x,  stride_sample_y,  stride_sample_dst);
        return true;
    }

    return false;
#endif
}
