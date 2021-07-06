#include "cuda_helper.h"
#include "ffthelper.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <math.h>
#include "cuComplex.h"

#include <complex>
#include <vector>

using ComplexVec = std::vector<std::complex<float>>;

namespace refft {

// Modular multiplication a * N mod p
// In: a[np][N]

__device__ cuFloatComplex twiddle(const float expr) {
  cuFloatComplex res;
  float s, c;
  sincosf(expr, &s, &c);
  res.x = c;
  res.y = s;
  return res;
}

__device__ void butt_fft(cuFloatComplex *a, cuFloatComplex *b,
                         cuFloatComplex w) {
  cuFloatComplex U = cuCmulf(*b, w);
  *b = cuCsubf(*a, U);
  *a = cuCaddf(*a, U);
}

__global__ void Fft(cuFloatComplex *a, const int m, const int N, const int num_images) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2) * num_images;
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    int image_idx = i / (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a + image_idx * N;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(-M_PI * (double)t_idx / (double)m);
    butt_fft(a_x, a_y, w);
  }
}

__device__ void butt_ifft(cuFloatComplex *a, cuFloatComplex *b,
                          cuFloatComplex w) {
  cuFloatComplex T = cuCsubf(*a, *b);
  *a = cuCaddf(*a, *b);
  (*a).x /= 2.0;
  (*a).y /= 2.0;
  *b = cuCmulf(T, w);
  (*b).x /= 2.0;
  (*b).y /= 2.0;
}

__global__ void Ifft(cuFloatComplex *a, const int m, const int N, const int num_images) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2) * num_images;
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    int image_idx = i / (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a + image_idx * N;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(M_PI * (double)t_idx / (double)m);
    butt_ifft(a_x, a_y, w);
  }
}

__global__ void bitReverse(std::complex<float> *a, int N, int num_images) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N) * num_images;
       i += blockDim.x * gridDim.x) {
    int logN = __log2f(N);
    int N_idx = i % N;
    int image_idx = i / N;
    std::complex<float> *a_x = a + N * image_idx;
    int revN = __brev(N_idx) >> (32 - logN);
    if (revN > N_idx) {
      std::complex<float> temp = a_x[N_idx];
      a_x[N_idx] = a_x[revN];
      a_x[revN] = temp;
    }
  }
}

__device__ cuFloatComplex Cmul(cuFloatComplex a, cuFloatComplex b) {
  float temp = double(a.x) * b.x - double(a.y) * b.y;
  float temp2 = double(a.x) * b.y + double(a.y) * b.x;
  cuFloatComplex res;
  res.x = temp;
  res.y = temp2;
  return res;
}

__global__ void Hadamard(cuFloatComplex *a, cuFloatComplex *b, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N);
       i += blockDim.x * gridDim.x) {
    int N_idx = i % N;
    cuFloatComplex *a_x = a;
    cuFloatComplex *b_x = b;
    a_x[N_idx] = Cmul(a_x[N_idx], b_x[N_idx]);
  }
}

void FftHelper::ExecFft(std::complex<float> *a, const int N, const int num_images) {
  dim3 blockDim(refft::FFTblocksize);
  dim3 gridDim(N/2/refft::FFTblocksize);
  {
    CudaHostSync();
    CudaTimer t("bitReversing_ours");
    bitReverse<<<gridDim, blockDim>>>(a,N, num_images);
  }
  {
    CudaHostSync();
    CudaTimer t("Butterfly_ours");
    for (int i = 1; i < N; i *= 2) {
      Fft<<<gridDim, blockDim>>>((cuFloatComplex *)a, i, N, num_images);
      CudaCheckError();
    }
  }
  CudaCheckError();
}

void FftHelper::ExecIfft(std::complex<float> *a, const int N, const int num_images) {
  dim3 blockDim(refft::iFFTblocksize);
  dim3 gridDim(N/2/refft::iFFTblocksize);
  for (int i = N / 2; i > 0; i >>= 1) {
    Ifft<<<gridDim, blockDim>>>((cuFloatComplex *)a, i, N, num_images);
  }
  bitReverse<<<gridDim, blockDim>>>(a, N, num_images);
  CudaCheckError();
}

void FftHelper::Mult(std::complex<float> *a, std::complex<float> *b, int N) {
  dim3 blockDim(refft::iFFTblocksize);
  dim3 gridDim(N/refft::iFFTblocksize);
  Hadamard<<<gridDim, blockDim>>>((cuFloatComplex*)a,(cuFloatComplex*)b, N);  
  CudaCheckError();
}

void FftHelper::ExecCUFFT(std::complex<float> *a, const int N, const int num_images) {
  cufftHandle plan;
  if(cufftPlan1d(&plan,N,CUFFT_C2C,num_images)!=CUFFT_SUCCESS){
    std::cout << "CUFFT ERROR : PLAN ERROR" << std::endl;
    return;
  }
  {
    CudaTimer t("CUFFT");
  if(cufftExecC2C(plan,(cuComplex *) a, (cuComplex *) a, CUFFT_FORWARD)!=CUFFT_SUCCESS){
    std::cout << "CUFFT ERROR : CUFFT ERROR" << std::endl;
    return;
  }
  }
}

void FftHelper::ExecCUIFFT(std::complex<float> *a, const int N, const int num_images) {
  cufftHandle plan;
  if(cufftPlan1d(&plan,N,CUFFT_C2C,num_images)!=CUFFT_SUCCESS){
    std::cout << "CUFFT ERROR : PLAN ERROR" << std::endl;
    return;
  }
  if(cufftExecC2C(plan,(cuComplex *) a, (cuComplex *) a, CUFFT_INVERSE)!=CUFFT_SUCCESS){
    std::cout << "CUFFT ERROR : CUFFT ERROR" << std::endl;
    return;
  }
}
}  // namespace refft
