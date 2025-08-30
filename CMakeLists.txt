cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)  

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
project(TensorRT-YOLO
    VERSION 6.3.0
    LANGUAGES CXX CUDA
    DESCRIPTION "🚀 Easier & Faster YOLO Deployment Toolkit for NVIDIA 🛠️"
)

# clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include")

# undefined symbol: fatbinData
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# -------------------------------------------------------------------------------
# CUDA
# -------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# get_cmake_property(_vars VARIABLES)
# foreach(v ${_vars})
#     if(v MATCHES "CUDA")  # chỉ in biến có chữ CUDAToolkit
#         message(STATUS "${v} = ${${v}}")
#     endif()
# endforeach()

list(APPEND CMAKE_CUDA_ARCHITECTURES 72 75 80 86 87 89 90)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
endif()
message(STATUS "CUDA code for default SMs: ${CMAKE_CUDA_ARCHITECTURES}")

# TensorRT
set(TRT_PATH "" CACHE PATH "Path to TensorRT installation (e.g., /usr/local/tensorrt)")
if(NOT TRT_PATH)
    message(FATAL_ERROR "TensorRT path must be specified with -DTRT_PATH=/path/to/tensorrt")
endif()

set(TRT_LIB_DIR "${TRT_PATH}/lib")
if(MSVC AND EXISTS "${TRT_LIB_DIR}/nvinfer_10.dll")
    set(TRT_LIBS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
else()
    set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
endif()

message(STATUS "TRT_PATH: ${TRT_PATH}")

# Python
option(BUILD_PYTHON "Build Python bindings with pybind11" OFF)
if(BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED QUIET)
    if(NOT pybind11_FOUND)
        message(WARNING "pybind11 not found. Python bindings will not be built.")
        set(BUILD_PYTHON OFF)
    endif()
endif()

# get_cmake_property(_vars VARIABLES)
# foreach(v ${_vars})
#     if(v MATCHES "pybind11")  # chỉ in biến có chữ CUDAToolkit
#         message(STATUS "${v} = ${${v}}")
#     endif()
# endforeach()

# message(STATUS "${TRT_LIBS}")
# message(STATUS "${BUILD_PYTHON}")

# #-------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------
function(set_target_compile_options target)
    # MSVC 
    if(MSVC)
        # C++
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /Oi /fp:fast>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Ob0 /Zi /RTC1>
        )

        # CUDA
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2 -O3>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Od -g -G>
        )

        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY
            "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>"
        )

    # GCC/Clang 
    else()
        # C++
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native -flto=auto -DNDEBUG>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g3 -fno-omit-frame-pointer -fno-inline>
        )

        # CUDA
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        )

        target_link_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -flto=auto>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g3>
        )
    endif()

    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    )
endfunction()

#-------------------------------------------------------------------------------
# TensorRT/CUDA 
#-------------------------------------------------------------------------------
function(configure_cuda_trt_target target)
    target_include_directories(${target} PRIVATE
        ${TRT_PATH}/include
    )

    target_link_directories(${target} PRIVATE
        ${TRT_LIB_DIR}
    )

    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        ${TRT_LIBS}
    )

    # CUDA 
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
add_subdirectory(modules/trtyolo)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
add_subdirectory(modules/plugin)