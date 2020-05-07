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

static void parse_opt(int argc, char **argv);
static void print_help(const char* prog_name);
void ReadFile(ComplexVec a, std::string file_name);

int main(int argc, char **argv){
	parse_opt(argc, argv);

  int N = 32768; 
  for(int iter = 0; iter < 1; iter++){
    ComplexVec h_a(N);
    ComplexVec res_ref(N);
    ReadFile(h_a, "Polynomial_Coeff.txt");
    ReadFile(res_ref, "Output_Coeff.txt");
    std::complex<float>* d_alpha =
        (std::complex<float>*)refft::DeviceMalloc(h_a);

    refft::FftHelper::ExecStudentFft(d_alpha, N);
    refft::CudaHostSync();

    ComplexVec res = refft::D2H(d_alpha, N);
    for (unsigned int i = 0; i < res.size(); i++) {
      if (!(abs(res_ref[i].real() - res[i].real()) < 0.001)) {
        std::cout << "Wrong value in index " << i << std::endl;
        std::cout << "Reference : " << res_ref[i].real() << std::endl;
        std::cout << "Calculated : " << res[i].real() << std::endl;
        std::exit(0);
      }
      if (!(abs(res_ref[i].imag() - res[i].imag()) < 0.001)) {
        std::cout << "Wrong value in index " << i << std::endl;
        std::cout << "Reference : " << res_ref[i].imag() << std::endl;
        std::cout << "Calculated : " << res_ref[i].imag() << std::endl;
        std::exit(0);
      }
    }
    refft::DeviceFree(d_alpha);
  }
  return 0;
}

static void parse_opt(int argc, char **argv) {
  int c;
  //int digit_optind = 0;
	while (1) {
		//int this_option_optind = optind ? optind : 1;
		int option_index = 0;
		static struct option long_options[] = {
			{"fftb",		required_argument, 0, '0'},
			{"ifftb",   required_argument, 0, '2'},
			{"help",  required_argument, 0, 'h'},
			{0,       0,                 0,  0 }
		};

		c = getopt_long(argc, argv, "0:2:h",
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
  printf("  ( c.f. gridsize = N/2/blocksize )\n\n");
  printf("  --help:  print help page \n");
}

void ReadFile(ComplexVec a, std::string file_name) {
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

