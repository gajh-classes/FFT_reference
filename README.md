# CUDA FFT Reference code

This is the reference fast fourier transform(fft) code for computer architecture class 2020.\
Find "ExecFft" function in src/ffthelper.cu, where the radix-2 colley tukey algorithm is implemented as a reference code. You are asked to change the function to optimize for the GPU architecture, using the gpgpu-sim simulator.



## Overview

In this project, we will use docker to setup gpgpu-sim environment. 

1. Pull docker images.
2. Build FFT.
3. set environmental variables.
4. run 



## Install Docker & Pull Image

The following instructions are based on the Ubuntu 18.04 LTS. 

```
$ apt-get install docker.io
$ docker pull michael604/scal2020_gpgpu_sim
$ docker pull michael604/comparch_430.636_spring2020
$ docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined michael604/comparch_430.636_spring2020 --name {NAME}
```



## How to Build & Run
This part will explain necessary procedures to properly build FFT_release CDUA application and how to attach it to GPGPU-sim when executing the binary.\
We also provide automated script file `makestep.sh`, so you may also use the script.\
(Directory names in the following explanation is based on the [docker image][docker_image] that is provided.

### Build
in `/root/FFT_release` directory,
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
make -j
```
Binary file named `fft_release` will be generated in `/root/FFT_release/build`

### Run
You need to set correct environment variable in order to properly attach the CUDA application to GPGPU-sim.You also have to locate the correct GPGPU-sim configuration files and matrix data.

##### Set Environmental Variable
You first need to set environmental variables correctly.\
In `/root/gpgpu-sim_distribution` directory,
```bash
$ source setup_environment {BUILD_TYPE}
```
The default `BUILD_TYPE` in the docker [image][docker_image] is release. 

##### Locate Configuration Files & Matrix Data
You need to locate `config_ARCH_islip.cnt` & `gpgpusim.config` files from `/gpgpu-sim_distribute/configs/tested-cfgs`, to the directory that you are going to run the application.

In `/root/FFT_release` directory,
```bash
$ mkdir build/run && cd build/run
$ cp /root/gpgpu-sim_distrubute/configs/tested-cfgs/{SM_NUM}/* ./
$ cp /root/FFT_release/test/*.txt ./
```
We will use `SM2_GTX480` configuration as default machine configuration for GPGPU-sim. We do not need to run power simulation, so you need to change the `power_simulation_enabled` value to 0 in `gpgpusim.config` file.

##### Run CUDA Application
Normally execute the application binary, and it will attach to GPGPU-sim properly.
```bash
cd /root/FFT_release/build/run
./../fft_release
```

##### makestep.sh
in `/root/FFT_release`, automated script is provided. You may use the script to skip the above procedures.



## To do

* Make your ***fft / ifft*** code that performs better (leading to short execution time).\
You can modify every part of the code and add any other custom function. But it is necessary to explain the implementation.

* When you run the application binary with gpgpu-sim, execution time of the code will be displayed. The execution time of your optimized code and application will be evaluated using the time displayed at the end.

* If it's necessary, you are allowed to modify the CMakeLists.txt. However, do not modify any important flags, such as optimizaiton level flags. 



# Report

* Describe what you have done.

## Contact

If you find some errors or issues in the code, fill free to e-mail us. Our e-mails are posted in our class discription page.

Michael Jaemin Kim: michael604@scale.snu.ac.kr\
Sangpyo Kim: spkim@scale.snu.ac.kr\
[Class Page Link](https://scale-snu.github.io/jekyll/update/2020/03/16/aca2020-lecture-01.html)

[docker_image]: https://hub.docker.com/repository/docker/michael604/comparch_scale2020
