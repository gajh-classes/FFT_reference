#pragma once

#include <complex>
#include <iostream>
#include <vector>

#ifndef NDEBUG
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
#else
#define CudaCheckError()
#endif

namespace refft {
using HostVec = std::vector<uint32_t>;
using ComplexVec = std::vector<std::complex<float>>;

extern int blocksize;
extern int gridsize;


void __cudaCheckError(const char* file, const int line);
// A wrapper of cudaDeviceSynchronize() to be called from the host side.
void CudaHostSync();
void* DeviceMalloc(size_t len);
void* DeviceMalloc(const std::vector<std::complex<float>> input);
// copy device input to vector
std::vector<std::complex<float>> D2H(const std::complex<float>* input,
                                     size_t elems);
// Convert device pointer to ZZs or a ZZ.
void DeviceFree(uint32_t* p);
void DeviceFree(bool* p);
void DeviceFree(std::complex<float>* p);

}  // namespace refft
