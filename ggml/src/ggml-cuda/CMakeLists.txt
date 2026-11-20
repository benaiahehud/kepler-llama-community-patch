cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

# ARM64EC uses the x64 ABI, so it needs the x64 import libraries. The arm64 ones
# hold native ARM64 symbols and cannot satisfy EC references at link time.
# FindCUDAToolkit picks the library dir from the host arch, not the target, so
# anchor its search at lib/x64 here. It derives everything else from CUDA_CUDART.
string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" GGML_CUDA_PLATFORM_LWR)
if (MSVC AND GGML_CUDA_PLATFORM_LWR STREQUAL "arm64ec")
    find_library(CUDA_CUDART
        NAMES cudart
        HINTS ${CUDAToolkit_ROOT} ENV CUDA_PATH
        PATH_SUFFIXES lib/x64
    )
endif()

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")

    # unconfirmed patch: EOL ver. maybe 11... maybe 10... maybe 9...
    #if (CUDAToolkit_VERSION VERSION_LESS "12")
    #    message(WARNING "CUDA Toolkit < v12 is EOL! Although it compiles normally, normal operation is not guaranteed, and issues within the EOL scope may not be accepted for submission.")
    #endif()
    if (${CMAKE_CUDA_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Cooperative Groups support of only nvcc, use by softmax.cu file of 'soft_max_f32_parallelize_cols_single_row' function")
        if (CUDAToolkit_VERSION VERSION_LESS "4.1")
            message(FATAL_ERROR "Clang must CUDA Toolkit >= v4.1") # PTX ISA 3.0(ptx30)
        endif()
    endif()

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 20     == Fermi, __ballot (Warp Vote Functions), LLVM-IR NVPTX (-DCMAKE_CUDA_COMPILER=clang) compiler support, end of CUDA v8.0
        #  Thx very much <LLVM Project> and <Kitware CMake> always keep compatibility with the old NVIDIA architecture :-)
        #  https://llvm.org/docs/CodeGenerator.html#the-nvptx-backend
        #  https://llvm.org/docs/NVPTXUsage.html#nvptx-architecture-hierarchy-and-ordering
        # 30     == Kepler, __shfl instruction (shuffle), end of CUDA v10.2
        # 50     == Maxwell, lowest CUDA 12 standard
        # 53     == Maxwell3, FP16 CUDA intrinsics, but it's Tegra, so not in "if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)"
        # 60     == P100, Cooperative Groups support
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        # 80     == Ampere, asynchronous data loading, faster tensor core instructions
        # 86     == RTX 3000, needs CUDA v11.1
        # 89     == RTX 4000, needs CUDA v11.8
        # 90     == Hopper H100/200, needs CUDA v11.8
        # 120    == Blackwell, needs CUDA v12.8, FP4 tensor cores
        #
        # XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
        # XX-real    == compile CUDA code as device code for this specific architecture
        # no suffix  == compile as both PTX and device code
        #
        # The default behavior for a non-native is to build virtual architectures as needed to cover all features needed
        #     for best performance and to also build real architectures for the most commonly used GPUs.
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            # EOL
            if (NOT ${CMAKE_CUDA_COMPILER_ID} MATCHES "Clang")#TODO => clang: error: unsupported CUDA gpu architecture: sm_1Y
                if (CUDAToolkit_VERSION VERSION_LESS_EQUAL "6.0") #sm_10 = Tesla G80
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 10-real)
                endif()
                if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "2.0" AND CUDAToolkit_VERSION VERSION_LESS "7") #Tesla, end of CUDA v6.5
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 11-real 12-real 13-real)
                endif()
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "3.0" AND CUDAToolkit_VERSION VERSION_LESS "9") #sm_20 = Fermi, end of CUDA v8.0
                if (${CMAKE_CUDA_COMPILER_ID} MATCHES "Clang")
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 20-real 21-real)#because 'Clang doesn't support disabling CUDA real code generation.'
                else()
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 20-virtual)#nvcc unsupport compile sm_21
                endif()
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "4.2" AND CUDAToolkit_VERSION VERSION_LESS "11") #sm_30 = Kepler old, end of CUDA v10.2
                list(APPEND CMAKE_CUDA_ARCHITECTURES 30-virtual)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "5.0" AND CUDAToolkit_VERSION VERSION_LESS "12") #sm_35 = Kepler new, end of CUDA v11.8
                list(APPEND CMAKE_CUDA_ARCHITECTURES 35-virtual)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "6.0" AND CUDAToolkit_VERSION VERSION_LESS "13") #Maxwell~Volta, end of CUDA v12.9.2
                list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual)
                if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "8.0") #sm_61 = Pascal, __dp4a instruction (per-byte integer dot product)
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 61-virtual)
                endif()
                if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "9.0")# AND CUDAToolkit_VERSION VERSION_EQUAL "9.1" AND ${CMAKE_CUDA_COMPILER_ID} MATCHES "Clang")
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 70-virtual) #sm_70 = Volta, FP16 tensor cores
                endif()
            endif()

            # Updateing... / MOL / TL;DR
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.0") #sm_75 = Turing, int8 tensor cores
                list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual)
                if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0") #sm_80 = Ampere, asynchronous data loading, faster tensor core instructions
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 80-virtual)
                endif()
                if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1") #sm_86 = RTX 3000
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
                endif()
                if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8") #sm_89 = RTX 4000
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
                    list(APPEND CMAKE_CUDA_ARCHITECTURES 90-virtual)
                endif()
            endif()

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                # The CUDA architecture 120f-virtual would in principle work for Blackwell support
                #     but the newly added "f" suffix conflicted with a preexising regex for validating CUDA architectures in CMake.
                # So either a recent CMake version or one with the backported fix is needed.
                # The following versions should work:
                #   - CMake >= v3.31.8 && CMake < v4.0.0
                #   - CMake >= v4.0.2
                # This is NOT documented in the CMake release notes,
                #     check Modules/Internal/CMakeCUDAArchitecturesValidate.cmake in the CMake git repository instead.
                # However, the architectures 120a-real and 121a-real should work with basically any CMake version and
                #     until the release of e.g. Rubin there is no benefit to shipping virtual architectures for Blackwell.
                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
            endif()
        endif()
    endif()

    enable_language(CUDA)

    if (GGML_CUDA_CCCL_VERSION)
        include(FetchContent)

        FetchContent_Declare(
            CCCL
            GIT_REPOSITORY https://github.com/nvidia/cccl.git
            GIT_TAG        "${GGML_CUDA_CCCL_VERSION}"
            GIT_SHALLOW    TRUE
        )

        FetchContent_MakeAvailable(CCCL)
    endif()

    # Replace any plain 12X CUDA architectures with their "architecture-specific" equivalents 12Xa.
    # 12X is forwards-compatible, 12Xa is not.
    # Notably the Blackwell FP4 tensor core instructions are not forwards compatible and therefore need 12Xa.
    # But while 12X vs. 12Xa can be checked in device code there is (to my knowledge) no easy way to do the same check in host code.
    # So for now just replace all instances of 12X with 12Xa, this should be fine until Rubin is released.
    foreach(ARCHS IN ITEMS CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(FIXED_ARCHS "")
        foreach(ARCH IN LISTS ${ARCHS})
            if (ARCH MATCHES "^12[0-9](-real|-virtual)?$")
                string(REGEX REPLACE "^(12[0-9])((-real|-virtual)?)$" "\\1a\\2" FIXED_ARCH ${ARCH})
                message(STATUS "Replacing ${ARCH} in ${ARCHS} with ${FIXED_ARCH}")
                list(APPEND FIXED_ARCHS "${FIXED_ARCH}")
            else()
                list(APPEND FIXED_ARCHS "${ARCH}")
            endif()
        endforeach()
        set(${ARCHS} ${FIXED_ARCHS})
    endforeach()

    # If we try to compile a "native" build it will use the 12X architectures and fail.
    # So we should instead use the native architectures as determined by CMake after replacing 12X with 12Xa.
    # But if at the time of the build no GPUs are connected at all CMAKE_CUDA_ARCHITECTURES will contain garbage that we should not use.
    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?|;)*$")
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} CMAKE_CUDA_ARCHITECTURES_NATIVE=${CMAKE_CUDA_ARCHITECTURES_NATIVE}")

    file(GLOB   GGML_HEADERS_CUDA "*.cuh")
    list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")

    file(GLOB   GGML_SOURCES_CUDA "*.cu")
    file(GLOB   SRCS "template-instances/fattn-tile*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmf*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    ggml_cuda_fattn_vec_instances(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    ggml_add_backend_library(ggml-cuda
                             ${GGML_HEADERS_CUDA}
                             ${GGML_SOURCES_CUDA}
                            )

    if (GGML_CUDA_GRAPHS)
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.0")
            add_compile_definitions(GGML_CUDA_USE_GRAPHS)
        else()
            message(WARNING "CUDA Graphs support requirements >= CUDA Toolkit 10.0")
        endif()
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    elseif(CUDAToolkit_VERSION VERSION_LESS "10.2")
        add_compile_definitions(GGML_CUDA_NO_VMM)
        message(WARNING "CUDA Virtual Memory Management support requirements >= CUDA Toolkit 10.2")
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (GGML_CUDA_CCCL_VERSION)
        target_link_libraries(ggml-cuda PRIVATE CCCL::CCCL)
    endif()

    if (GGML_STATIC)
        if (WIN32)
            # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
            target_link_libraries(ggml-cuda PRIVATE CUDA::cudart_static CUDA::cublas)
        else ()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.1")
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            else()
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static)
            endif()
        endif()
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    if (GGML_CUDA_NO_VMM)
        # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)
    endif()

    if (GGML_CUDA_NCCL)
        find_package(NCCL)
        if (NCCL_FOUND)
            add_compile_definitions(GGML_USE_NCCL)
            target_link_libraries(ggml-cuda PRIVATE NCCL::NCCL)
        else()
            message(STATUS "Warning: NCCL not found, performance for multiple CUDA GPUs will be suboptimal")
        endif()
    endif()

    if (NOT DEFINED CUDA_CXX_FLAGS)
        set(CUDA_CXX_FLAGS "")
    endif()

    if (${CMAKE_CUDA_COMPILER_ID} MATCHES "Clang")#LLVM-IR NVPTX
        list(APPEND CUDA_CXX_FLAGS -Wno-unknown-cuda-version --no-cuda-version-check -D__NV_NO_HOST_COMPILER_CHECK=1 -Wno-all -Wno-extra)
        if (NOT WIN32)
            list(APPEND CUDA_CXX_FLAGS -fPIC)
        endif()
    else()#nvcc
        #--compiler-options -fPIC # Explicitly set PIC flags for CUDA targets.
        #-D_FORCE_INLINES # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
        #-Wno-deprecated-gpu-targets # Do not show warnings if the architectures are deprecated.
        set(CUDA_FLAGS -w -use_fast_math -D_FORCE_INLINES -Wno-deprecated-gpu-targets)
        if (CUDAToolkit_VERSION VERSION_EQUAL "7.0")
            list(APPEND CUDA_FLAGS -relaxed-constexpr)
            #list(APPEND CUDA_FLAGS --extended-lambda)#nvcc.exe v7.0 is closed
        endif()
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "7.5")#nvcc.exe v9.0 open -expt-extended-lambda actual effect
            list(APPEND CUDA_FLAGS -expt-relaxed-constexpr -expt-extended-lambda)
        endif()
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "9.0")
            list(APPEND CUDA_FLAGS -Wno-deprecated-declarations)# Do not show warnings if cuda library functions are deprecated.
        endif()
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.1.2" AND NOT APPLE)
            list(APPEND CUDA_FLAGS -allow-unsupported-compiler)
        endif()
        list(APPEND CUDA_FLAGS -D__NV_NO_HOST_COMPILER_CHECK=1)# Some older versions of the CUDA Toolkit need to continue doing this to affect crt/host_config.h

        if (GGML_CUDA_DEBUG)
            list(APPEND CUDA_FLAGS -lineinfo)
            add_compile_definitions(GGML_CUDA_DEBUG)
        endif()

        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            # Options are:
            # - none (not recommended)
            # - speed (nvcc's default)
            # - balance
            # - size
            list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
        endif()

        if (GGML_FATAL_WARNINGS)
            list(APPEND CUDA_FLAGS -Werror all-warnings)
        endif()

        if (GGML_ALL_WARNINGS AND NOT MSVC)
            set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
            if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
                list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
            endif()

            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler --version
                OUTPUT_VARIABLE CUDA_CCFULLVER
                ERROR_QUIET
            )

            if (NOT CUDA_CCFULLVER MATCHES clang)
                set(CUDA_CCID "GNU")
                execute_process(
                    COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                    OUTPUT_VARIABLE CUDA_CCVER
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )
            else()
                if (CUDA_CCFULLVER MATCHES Apple)
                    set(CUDA_CCID "AppleClang")
                else()
                    set(CUDA_CCID "Clang")
                endif()
                string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
            endif()

            message(STATUS "CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

            ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
            list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
        endif()

        if (NOT MSVC)
            list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
        else()
            # CCCL 3.2 onwards will require a cpp-standard-compliant preprocessor for MSVC
            # https://github.com/NVIDIA/cccl/pull/6827
	    
            #TODO _MSC_VER unknow
            #list(APPEND CUDA_CXX_FLAGS /W0 /Gm- /EHsc- /D_HAS_EXCEPTIONS=0
            #                           /GS- /Zi- /Z7-)# /FdNUL
            # https://learn.microsoft.com/zh-cn/cpp/build/reference/zc-preprocessor?view=msvc-170
            #if (_MSC_VER GREATER_EQUAL 1650)
                list(APPEND CUDA_CXX_FLAGS /Zc:preprocessor)
            #elseif(_MSC_VER GREATER_EQUAL 1580)
            #    list(APPEND CUDA_CXX_FLAGS /experimental:preprocessor)
            #endif()
        endif()
    endif()

    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        if (${CMAKE_CUDA_COMPILER_ID} MATCHES "NVIDIA")#nvcc
            list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
        else()#Clang / PGI / HPC SDK ...
            list(APPEND CUDA_FLAGS ${CUDA_CXX_FLAGS_JOINED})
        endif()
    endif()

    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
else()
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
