#pragma once
#include "cuda_helper.h"
#include "ffthelper.h"
#include "utils.h"

#include "cuComplex.h"
#include <cuda_runtime.h>
#include <math.h>
#include <cufft.h>
#include <cufftXt.h>

#include <vector>
#include <complex>


using ComplexVec = std::vector<std::complex<float>>;

namespace refft{

// Modular multiplication a * N mod p
// In: a[np][N]

__device__ cuFloatComplex twiddle(const float expr){
  cuFloatComplex res;
  float s, c;
  sincosf(expr, &s,&c);
  res.x = c;
  res.y = s;
  return res;
}

__device__ void butt_fft(cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex w){
  cuFloatComplex U = cuCmulf(*b, w);
  *b = cuCsubf(*a,U);
  *a = cuCaddf(*a,U);
}

__global__ void Fft(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2) ;
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(-M_PI * (double) t_idx / (double) m);   
    butt_fft(a_x, a_y, w);
  }
}

__global__ void FftStudent(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2) * 1 ;
       i += blockDim.x * gridDim.x) {
   // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(-M_PI * (double) t_idx / (double) m);   
    butt_fft(a_x, a_y, w);
  }
}

__device__ void butt_ifft(cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex w){
  cuFloatComplex T = cuCsubf(*a,*b);
  *a = cuCaddf(*a,*b);
  (*a).x /= 2.0;
  (*a).y /= 2.0;
  *b = cuCmulf(T,w);
  (*b).x /= 2.0;
  (*b).y /= 2.0;
}

__global__ void Ifft(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2) ;
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
     // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(M_PI * (double) t_idx / (double) m);   
    butt_ifft(a_x, a_y, w);
  }
}

__global__ void IfftStudent(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2) ;
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
     // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(M_PI * (double) t_idx / (double) m);   
    butt_ifft(a_x, a_y, w);
  }
}

__global__ void bitReverse(std::complex<float> *a, int N){ 
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); 
       i += blockDim.x * gridDim.x) {
    int logN = __log2f(N);
    int N_idx = i % N;
    std::complex<float> *a_x = a;
    int revN = __brev(N_idx)>>(32-logN);
    if(revN > N_idx){
      std::complex<float> temp = a_x[N_idx];
      a_x[N_idx] = a_x[revN];
      a_x[revN] = temp;
    }
  }
}

__device__ cuFloatComplex Cmul(cuFloatComplex a, cuFloatComplex b)
{
  float temp = double(a.x)*b.x - double(a.y)*b.y;
  float temp2 = double(a.x)*b.y + double(a.y)*b.x;
  cuFloatComplex res;
  res.x = temp;
  res.y = temp2;
  return res;
}

__global__ void Hadamard(cuFloatComplex *a, cuFloatComplex *b, int N){ 
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); 
       i += blockDim.x * gridDim.x) {
    int N_idx = i % N;
    cuFloatComplex *a_x = a;
    cuFloatComplex *b_x = b;
    a_x[N_idx] = Cmul(a_x[N_idx],b_x[N_idx]);
  }
}

void ExecFft(std::complex<float> *a, int N) {
  dim3 blockDim(256);
  dim3 gridDim(N/2/256);
  bitReverse<<<gridDim, blockDim>>>(a,N);
  for (int i = 1; i < N; i *= 2) {
    Fft<<<gridDim, blockDim>>>((cuFloatComplex*)a,i, N);
    CudaCheckError();
  }
  CudaCheckError();
}

void ExecStudentFft(std::complex<float> *a, int N) {
  dim3 blockDim(256);
  dim3 gridDim(N/2/256);
  bitReverse<<<gridDim, blockDim>>>(a,N);
  for (int i = 1; i < N; i *= 2) {
    std::cout << i << std::endl;
    FftStudent<<<gridDim, blockDim>>>((cuFloatComplex*)a,i, N);
    CudaCheckError();
  }
  CudaCheckError();
}

void ExecIfft(std::complex<float> *a, int N) {
  dim3 blockDim(1024);
  dim3 gridDim(N/2/1024);
  for (int i = N / 2; i > 0; i >>= 1) {
    Ifft<<<gridDim, blockDim>>>((cuFloatComplex*)a,i, N);
  }
  bitReverse<<<gridDim, blockDim>>>(a,N);
  CudaCheckError();
}

void ExecStudentIfft(std::complex<float> *a, int N) {
  dim3 blockDim(1024);
  dim3 gridDim(N/2/1024);
  for (int i = N / 2; i > 0; i >>= 1) {
    IfftStudent<<<gridDim, blockDim>>>((cuFloatComplex*)a,i, N);
  }
  bitReverse<<<gridDim, blockDim>>>(a,N);
  CudaCheckError();
}

void Mult(std::complex<float> *a, std::complex<float> *b, int N) {
  dim3 blockDim(1024);
  dim3 gridDim(N/1024);
  Hadamard<<<gridDim, blockDim>>>((cuFloatComplex*)a,(cuFloatComplex*)b, N);  
  CudaCheckError();
}
}  // namespace 
