cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fft_release LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(HE_NVCC_ARCHS ${HE_NVCC_ARCHS_SUPPORTED} CACHE STRING "The SM architectures requested.")
set(HE_NVCC_ARCHS_ENABLED ${HE_NVCC_ARCHS} CACHE STRING "The SM architectures to build code for.")
foreach(ARCH ${HE_NVCC_ARCHS_ENABLED})
  string(APPEND NVCC_GENCODE_FLAGS " -gencode=arch=compute_${ARCH},code=compute_${ARCH}")
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -Xcompiler=-Wall,-fopenmp -lcudart ")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -g -O0 -lcudart")

add_executable(fft_release 
  src/ffthelper.cu
  src/cuda_helper.cu
  test/ffttest.cu
)

set_target_properties(fft_release PROPERTIES CUDA_ARCHITECTURES 70 72 75)

target_include_directories(fft_release
  INTERFACE include
  PRIVATE include/fft
)

target_link_libraries(fft_release
  PUBLIC cudart cufft CUDA::nvToolsExt)
