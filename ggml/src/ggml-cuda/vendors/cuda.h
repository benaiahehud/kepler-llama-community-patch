#pragma once
/*
!defined(__CUDA_ARCH__) == __host__
 defined(__CUDA_ARCH__) == __device__
*/

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#if CUDART_VERSION >= 7050
  #include <cuda_fp16.h>
#else //TODO test CUDA 6 / Tesla arch
	#define __half2float(...) (...)
	#define __float2half(...) (...)
	#define half float
	#define half2 float2
	#define CUDA_R_16F CUDA_R_32F
#endif // CUDART_VERSION >= 7050

// BF16: (CUDART_VERSION >= 11000 && CUDART_VERSION < 12020) most requirements only __CUDA_ARCH__ >= Ampere available, but CUDART_VERSION >= 12020 open all __CUDA_ARCH__ available
#if CUDART_VERSION >= 11000
  #include <cuda_bf16.h>
  #if CUDART_VERSION < 12020 && __CUDA_ARCH__ < 800
	  #define __bfloat1622float2(x) make_float2(__bfloat162float(x.x),__bfloat162float(x.y))
  #endif
#else
	#define __bfloat162float __half2float
	#define __float2bfloat16 __float2half
	#define __bfloat1622float2 __half22float2
	#define nv_bfloat16 half
	#define nv_bfloat162 half2
	#define CUDA_R_16BF CUDA_R_16F
#endif

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL


// === Lib func

#if CUDART_VERSION < 9010 //ggml-cuda.cu:only FP32 compute
	#define cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo) cublasSgemmBatched(handle, transa, transb, m, n, k, (const float *)alpha, (const float **)Aarray, lda, (const float **)Barray, ldb, (const float *)beta, (float **)Carray, ldc, batchCount)
	#define cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo) cublasSgemmStridedBatched(handle, transa, transb, m, n, k, (const float *)alpha, (const float *)A, lda, strideA, (const float *)B, ldb, strideB, (const float *)beta, (float *)C, ldc, strideC, batchCount)
#endif // CUDART_VERSION < 9010

#if CUDART_VERSION < 9000
	#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS /*mode==CUBLAS_DEFAULT_MATH?CUBLAS_STATUS_SUCCESS:CUBLAS_STATUS_INVALID_VALUE*/ // reference from hip.h
	#define CUBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DFALT
#endif // CUDART_VERSION < 9000

#if CUDART_VERSION < 11010
  // cudaStreamWaitEvent(stream, event) suggest legacy coding style: cudaStreamWaitEvent(stream, event, 0)
#endif


// === Inst func

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl
#if (CUDART_VERSION >= 9000) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300)
  // native __*_sync
#elif (CUDART_VERSION >= 4010) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 700))
  #define __shfl_sync(mask,var,srcLane)           __shfl(var,srcLane)
  #define __shfl_sync(mask,var,srcLane,width)     __shfl(var,srcLane,width)
  #define __shfl_xor_sync(mask,var,srcLane)       __shfl_xor(var,srcLane)
  #define __shfl_xor_sync(mask,var,srcLane,width) __shfl_xor(var,srcLane,width)
  #define __shfl_up_sync(mask,var,srcLane)        __shfl_up(var,srcLane)
  #define __shfl_up_sync(mask,var,srcLane,width)  __shfl_up(var,srcLane,width)
#else
  #define __shfl_sync(mask,var,...)     var
  #define __shfl_xor_sync(mask,var,...) var
  #define __shfl_up_sync(mask,var,...)  var
  #pragma message("maybe in run using '__shfl' can unexpected logical error")
#endif
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote
#if CUDART_VERSION >= 9000
  // native __*_sync
#elif (CUDART_VERSION >= 2000) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700)
  #define __all_sync(mask,var) __all(var)
  #define __any_sync(mask,var) __any(var)
#else
  #define __all_sync(mask,var) var
  #define __any_sync(mask,var) var
  #pragma message("maybe in run using '__all'/'__any' can unexpected logical error")
#endif
#if (CUDART_VERSION >= 9000) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200)
  // native __*_sync
#elif (CUDART_VERSION >= 2000) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 200 && __CUDA_ARCH__ < 700))
  #define __ballot_sync(mask,var) __ballot(var)
#else
  #define __ballot_sync(mask,var) var
  #pragma message("maybe in run using '__ballot' can logical error")
#endif

#if CUDART_VERSION <= 8000
static __host__ __forceinline__ __half __float2half(const float a){ // fill.cu:(void __cdecl ggml_cuda_op_fill(struct ggml_backend_cuda_context &, struct ggml_tensor *))
  __half ret;
  #if __cplusplus > 202002L //C++23
    ret.x = std::bit_cast<unsigned short>(static_cast<std::float16_t>(a));
  #else
    // TODO __float2ushort
    #pragma message("CUDA Toolkit <= v8.0 must 'CMAKE_CXX_COMPILER' of support C++ 23 or hight (e.g.: -DCMAKE_CXX_COMPILER=clang)")
  #endif
  return ret;
}
// __half2 == int x
static __device__ __forceinline__ __half2 make_half2(float l, float h)
{
  __half2 t; t.x = (__float2half_rn(l) & 0xFFFF) | (__float2half_rn(h) << 16); return t;
}
/*
CUDA v9.1(unknow min. ver.) is available
// __half2 == short x,y
__half2 make_half2(__half x, __half y)
{
    __half2 t; t.x = x; t.y = y; return t;
}*/
#endif
