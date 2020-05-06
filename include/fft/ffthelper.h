#pragma once

#include <stdint.h>

#include <complex>
#include <vector>

namespace refft {
  class FftHelper{
    public:
    static void ExecFft(std::complex<float> *a, int N);
    static void ExecStudentFft(std::complex<float> *a, int N);
    static void ExecIfft(std::complex<float> *a, int N);
    static void ExecStudentIfft(std::complex<float> *a, int N);
    static void Mult(std::complex<float> *a, std::complex<float> *b, int N);
 };
}  // namespace refft 
