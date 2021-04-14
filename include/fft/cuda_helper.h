#pragma once

#include <complex>
#include <iostream>
#include <vector>
#include <chrono>

#ifndef NDEBUG
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
#else
#define CudaCheckError()
#endif

namespace refft {
using HostVec = std::vector<uint32_t>;
using ComplexVec = std::vector<std::complex<float>>;

extern int FFTblocksize;
extern int iFFTblocksize;

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
void CudaNvtxStart(std::string msg = "");
void CudaNvtxStop();
class CudaTimer {                                                                                                                                            
 public:
  CudaTimer(std::string name, bool sync = true, bool print = true)                                                                                           
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()),                                                                                                    
        sync_(sync),
        print_(print) {
    CudaNvtxStart(m_name);                                                                                                                                   
  }
  ~CudaTimer() {                                                                                                                                             
    if (sync_)
      CudaHostSync();                                                                                                                                        
    CudaNvtxStop();                                                                                                                                          
    if (print_) {
      auto end = std::chrono::high_resolution_clock::now();                                                                                                  
      auto dur =
          std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);                                                                                
      std::cout << m_name + ", " + std::to_string(dur.count()) + "us\n";                                                                                     
    }                                                                                                                                                        
  }
 private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;                                                                                         
  bool sync_;
  bool print_;                                                                                                                                               
};
}  // namespace refft
