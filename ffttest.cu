#define MAX(X,Y) ((X) > (Y) ? (X) : (Y)) 
#include "cuda_helper.h"
#include "ffthelper.h"
#include "utils.h"


//#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>


using HostVec = std::vector<uint32_t>;
using ComplexVec = std::vector<std::complex<float>>;
using refft::utils::genComp;
using refft::utils::genZero;

void ReadFile(ComplexVec a, std::string file_name){
  std::ifstream input;
  input.open(file_name);
  if(!input.good()){
    std::cout << "Could not open " << file_name <<std::endl;
    std::exit(0);
  }
  float real;
  float imag; 
  for(int i = 0;i<a.size();i++){
    input >> real;
    input >> imag;
    a[i] = {real,imag};
  }
  input.close();

}

int main(){
  int N = 32768; 
  for(int iter = 0; iter < 1; iter++){
    ComplexVec h_a(N);
    ComplexVec res_ref(N);
    ReadFile(h_a, "Polynomial_Coeff.txt");
    ReadFile(res_ref, "Output_Coeff.txt");
    std::complex<float> * d_alpha = (std::complex<float>*)refft::DeviceMalloc(h_a);

    auto start = clock();
    refft::FftHelper::ExecStudentFft(d_alpha, N);
    refft::CudaHostSync(); 
    auto end = clock();
    double elapsed_sec = double(end - start) / CLOCKS_PER_SEC;

    ComplexVec res = refft::D2H(d_alpha , N);
    for(unsigned int i = 0; i< res.size(); i++){
      if(!(abs(res_ref[i].real() - res[i].real()) < 0.001))std::cout << "wrong " <<std::endl;
    }
    refft::DeviceFree(d_alpha);
    std::cout << "Duration of StudentFft : "<< elapsed_sec << "(sec)"<< std::endl;
  }
  return 0;
}

