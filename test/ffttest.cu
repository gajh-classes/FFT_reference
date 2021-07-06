#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#include "cuda_helper.h"
#include "ffthelper.h"
#include "utils.h"

//#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <getopt.h>
#include <random>
#include <vector>

using HostVec = std::vector<uint32_t>;
using ComplexVec = std::vector<std::complex<float>>;
using refft::utils::genComp;
using refft::utils::genZero;

static void parse_opt(int argc, char **argv, int& N, int &num_images, int &num_iters);
static void print_help(const char* prog_name);
void ReadFile(ComplexVec & a, std::string file_name);

ComplexVec GetRandomComplexVec(int N){
  ComplexVec randTemp(N);
  for(int i = 0; i < N; i++){
    randTemp[i] = {static_cast<float>(rand())/ static_cast<float>(RAND_MAX), static_cast<float>(rand())/ static_cast<float>(RAND_MAX)};
  }
  return randTemp;
}

int main(int argc, char **argv){
  int N = 32768; 
  int num_images = 32;
  int num_iter = 1;
	parse_opt(argc, argv, N, num_images, num_iter);
  int input_size = N * num_images;
  std::cout << "N :" << N  << std::endl;
  std::cout << "num_images :" << num_images  << std::endl;
  std::cout << "num_iter :" << num_iter  << std::endl;
  for(int iter = 0; iter < num_iter; iter++){
    auto h_a = GetRandomComplexVec(N * num_images);    
    std::complex<float>* d_alpha =
        (std::complex<float>*)refft::DeviceMalloc(h_a); 
    std::complex<float>* d_alpha_ref =
        (std::complex<float>*)refft::DeviceMalloc(h_a); 
    refft::CudaHostSync();
    refft::FftHelper::ExecFft(d_alpha, N, num_images);
    refft::FftHelper::ExecCUFFT(d_alpha_ref, N, num_images);
    refft::CudaHostSync();

    ComplexVec res_ref = refft::D2H(d_alpha, input_size);
    ComplexVec res_cufft = refft::D2H(d_alpha_ref, input_size);
    refft::CudaHostSync();
    refft::FftHelper::ExecIfft(d_alpha, N, num_images);
    refft::FftHelper::ExecCUIFFT(d_alpha_ref, N, num_images);
    refft::CudaHostSync();

    ComplexVec ires_ref = refft::D2H(d_alpha, N);
    ComplexVec ires_cufft = refft::D2H(d_alpha_ref, N);
    for (unsigned int i = 0; i < res_ref.size(); i++) {
      if (!(abs(res_ref[i].real() - res_cufft[i].real()) < MAX(abs(0.001*res_ref[i].real()),0.001))) {
        std::cout << "Wrong real value in index " << i << std::endl;
        std::cout << "Reference : " << res_ref[i].real() << std::endl;
        std::cout << "Calculated : " << res_cufft[i].real() << std::endl;
        //std::exit(0);
      }
      if (!(abs(res_ref[i].imag() - res_cufft[i].imag()) < MAX(abs(0.001 *res_ref[i].imag()),0.001))) {
        std::cout << abs(res_ref[i].imag() - res_cufft[i].imag()) <<std::endl;
        std::cout << "Wrong imag value in index " << i << std::endl;
        std::cout << "Reference : " << res_ref[i].imag() << std::endl;
        std::cout << "Calculated : " << res_cufft[i].imag() << std::endl;
        //std::exit(0);
      }
    }
    refft::DeviceFree(d_alpha);
    refft::DeviceFree(d_alpha_ref);
  }
  return 0;
}

static void parse_opt(int argc, char **argv, int& N, int &num_images, int &num_iters) {
  int c;
  //int digit_optind = 0;
	while (1) {
		//int this_option_optind = optind ? optind : 1;
		int option_index = 0;
		static struct option long_options[] = {
			{"fftb",		required_argument, 0, '0'},
			{"ifftb",   required_argument, 0, '2'},
			{"numimages",   required_argument, 0, 'm'},
			{"N",   required_argument, 0, 'N'},
			{"numiter",   required_argument, 0, 'i'},
			{"help",  required_argument, 0, 'h'}, 
			{0,       0,                 0,  0 }
		};
		c = getopt_long(argc, argv, "0:2:h:N:i:h",
				long_options, &option_index);
		if (c == -1)
		break; // while(1) break;

		switch (c) {
		case 0:
			printf("option %s", long_options[option_index].name);
			if (optarg)
				 printf(" with arg %s", optarg);
			printf("\n");
			break;
		case '0':
      refft::FFTblocksize = atoi(optarg);
			break;
    case '2':
      refft::iFFTblocksize = atoi(optarg);
      break;
    case 'm':
      num_images = atoi(optarg);
      break;
    case 'N':
      N = atoi(optarg);
      break;
    case 'i':
      num_iters = atoi(optarg);
      break;
		case 'h':
		default:
			print_help(argv[0]);
			exit(0);
		}
	}
}

static void print_help(const char* prog_name) {
  printf("Usage: %s [--fb FFTblocksize] [--ifb iFFTblocksize]\n", prog_name);
  printf("Options:\n");
  printf("  --fftb:  FFT  blocksize (default: 256 )\n");
  printf("  --ifftb: iFFT blocksize (default: 1024)\n");
  printf("  ( c.f. gridsize = N/2/blocksize )\n");
  printf("  --numiter : number of iteration for performance comparison\n");
  printf("  --N : length of input sequence\n");
  printf("  --numimags : the number of batching of input sequence\n\n");
  printf("  --help:  print help page \n");
}

void ReadFile(ComplexVec &a, std::string file_name) {
  std::ifstream input;
  input.open(file_name);
  if (!input.good()) {
    std::cout << "Could not open " << file_name << std::endl;
    std::exit(0);
  }
  float real;
  float imag;
  for (unsigned int i = 0; i < a.size(); i++) {
    input >> real;
    input >> imag;
    a[i] = {real, imag};
  }
  input.close();
}

