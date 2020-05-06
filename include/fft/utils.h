#pragma once

#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <complex>
#include <string>
#include <vector>

namespace refft {

namespace utils {

using namespace std;
using HostVec = std::vector<uint32_t>;
using ComplexVec = std::vector<std::complex<float>>;
static std::random_device rd;
static std::mt19937 fre(rd());
static std::uniform_real_distribution<float> fi(0.0,1.0);
static auto genComp = [&]() { return std::complex<float>(fi(fre),fi(fre)); };
static auto genZero = [&]() { return std::complex<float>(0,0); };

template <typename T>
void Print(T x, int num, std::string msg = "") {
  std::cout << msg << "\n";
  for (int i = 0; i < num; i++) {
    std::cout << std::hex << std::setfill('0') << std::setw(8) << x[i] << " ";
  }
  std::cout << "\n";
}

template <typename T>
void Print(std::vector<T> x, std::string msg = "") {
  Print(x, x.size(), msg);
}

}  // namespace utils
}  // namespace refft 
