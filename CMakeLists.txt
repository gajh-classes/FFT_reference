cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(cucrt LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
find_package(CUDA REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(HE_NVCC_ARCHS_SUPPORTED "")
if (NOT CUDA_VERSION VERSION_LESS 7.5)
  list(APPEND HE_NVCC_ARCHS_SUPPORTED 50)
endif()
if (NOT CUDA_VERSION VERSION_LESS 8.0)
  list(APPEND HE_NVCC_ARCHS_SUPPORTED 60 61)
endif()
if (NOT CUDA_VERSION VERSION_LESS 9.0)
  list(APPEND HE_NVCC_ARCHS_SUPPORTED 70)
endif()
if (NOT CUDA_VERSION VERSION_LESS 9.2)
  list(APPEND HE_NVCC_ARCHS_SUPPORTED 72)
endif()
if (NOT CUDA_VERSION VERSION_LESS 10.0)
  list(APPEND HE_NVCC_ARCHS_SUPPORTED 75)
endif()
set(HE_NVCC_ARCHS ${HE_NVCC_ARCHS_SUPPORTED} CACHE STRING "The SM architectures requested.")
set(HE_NVCC_ARCHS_ENABLED ${HE_NVCC_ARCHS} CACHE STRING "The SM architectures to build code for.")
foreach(ARCH ${HE_NVCC_ARCHS_ENABLED})
  string(APPEND NVCC_GENCODE_FLAGS " -gencode=arch=compute_${ARCH},code=compute_${ARCH}")
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -Xcompiler=-Wall,-fopenmp ${NVCC_GENCODE_FLAGS} -lcudart -std=c++11")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0 -lcudart")

add_executable(fft_release 
  src/ffthelper.cu
  src/cuda_helper.cu
  ffttest.cu
)

target_include_directories(fft_release
  INTERFACE include
  PRIVATE include/fft
)

target_link_libraries(fft_release
  PUBLIC cudart)
