#pragma once

#include <stdint.h>

#include <complex>
#include <vector>

namespace refft {
class FftHelper {
 public:
  static void ExecFft(std::complex<float> *a, const int N, const int num_images);
  static void ExecIfft(std::complex<float> *a, const int N, const int num_images);
  static void ExecCUFFT(std::complex<float> *a, const int N, const int num_images);
  static void ExecCUIFFT(std::complex<float> *a, const int N, const int num_images);
  static void Mult(std::complex<float> *a, std::complex<float> *b, int N);
};
}  // namespace refft
