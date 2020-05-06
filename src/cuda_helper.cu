#include "cuda_helper.h"

#include <assert.h>
#include <time.h>
#include <iostream>
#include <complex>
#include <fstream>
#include <vector>
#include <omp.h>

namespace refft {

#define CUDA_ERROR_CHECK
#define CUDA_CALL(x)                                                 \
  do {                                                               \
    if ((x) != cudaSuccess) {                                        \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                \
      printf("error: %s\n", cudaGetErrorString(cudaGetLastError())); \
      abort();                                                       \
    }                                                                \
  } while (0)

void __cudaCheckError(const char* file, const int line) {
  cudaError err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    std::cerr << "cudaCheckError failed at " << file << " " << line << "\n";
    std::cerr << "error: " << cudaGetErrorString(err) << "\n";
    exit(-1);
  }
  return;
}

void CudaHostSync() { cudaDeviceSynchronize(); }

void* DeviceMalloc(size_t len) {
  void* data;
  CUDA_CALL(cudaMalloc(&data, len));
  return data;
}

void* DeviceMalloc(const std::vector<std::complex<float>> input) {
  void* dst = DeviceMalloc(input.size() * sizeof(std::complex<float>));
  CUDA_CALL(cudaMemcpy(dst, input.data(), input.size() * sizeof(std::complex<float>),
                       cudaMemcpyHostToDevice));
  return dst;
}

std::vector<std::complex<float>> D2H(const std::complex<float>* input, size_t elems) {
  std::vector<std::complex<float>> host(elems);
  CUDA_CALL(cudaMemcpy(host.data(), input, elems * sizeof(std::complex<float>),
                       cudaMemcpyDeviceToHost));
  return host;
}


void DeviceFree(uint32_t* p) { CUDA_CALL(cudaFree((void*)p)); }
void DeviceFree(bool* p) { CUDA_CALL(cudaFree((void*)p)); }
void DeviceFree(std::complex<float>* p) { CUDA_CALL(cudaFree((void*)p)); }

}  // namespace cucrt
