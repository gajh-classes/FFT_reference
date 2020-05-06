# CUDA FFT Reference code

This is the reference fast fourier transform(fft) code for computer architecture class 2020.\
Find "ExecFft" function in src/ffthelper.cu, where the radix-2 colley tukey algorithm is implemented as a reference code. You are asked to change the function to optimize for the GPU architecture, using the gpgpu-sim simulator.

## Overview

In this project, we will use docker to setup gpgpu-sim environment. 

1. Pull docker images.
2. Build FFT.
3. set environmental variables.
4. run 

## Docker images 

The following instructions are based on the Ubuntu 18.04 LTS. 

```
$ apt-get install docker.io
$ docker pull michael604/scal2020_gpgpu_sim
$ docker run -it michael604/scal2020_gpgpu_sim /bin/bash
```

## How to build
in */fft_release* directory,
```bash
mkdir build&&cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
make -j
```
Binary file named "*fft_release*" will be generated in {BUILD_DIRECTORY}

## How to Run
* You first need to set environmental variables correctly.

  * in */gpgpu-sim_distribution* directory,
```bash
source setup_environment {BUILD_TYPE}
```
* You need to locate "config_ARCH_islip.cnt" & "gpgpusim.config" files from */gpgpu-sim_distribute/configs/tested-cfgs*,
to the directory that you are going to run the application.

  * in */fft_release* directory,
```bash
mkdir run&&cd run
cp {gpgpu-sim_distrubute directory}/configs/tested-cfgs/{SM_NUM}/* ./
```

* You can then start the application binary just as you do normally.
```bash
./../fft_release
```

## To do

* Make your ***fft / ifft*** code that performs better (leading to short execution time).\
You can modify every part of the code and add any other custom function. But it is necessary to explain the implementation.

* When you run the application binary with gpgpu-sim, execution time of the code will be displayed. The execution time of your optimized code and application will be evaluated using the time displayed at the end.

* If it's necessary, you are allowed to modify the CMakeLists.txt. However, do not modify any important flags, such as optimizaiton level flags. 

# Reporting Content

* Describe what you have done.

## Contact

If you find some errors or issues in the code, fill free to e-mail us. Our e-mails are posted in our class discription page.

Michael Jaemin Kim: michael604@scale.snu.ac.kr\
Sangpyo Kim: spkim@scale.snu.ac.kr\
[Class Page Link](https://scale-snu.github.io/jekyll/update/2020/03/16/aca2020-lecture-01.html)

