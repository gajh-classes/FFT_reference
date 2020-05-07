#!/bin/bash

# when building the FFT_release, refernce to the actual location of the libcudart.so
# reference the GPGPU-sim's libcudart.so only when executing the binary
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

rm -rf build&&mkdir build;
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 
make -j

mkdir run 
cp ../SM6_TITANX/* ./run/
cp /root/FFT_reference/test/*.txt ./run/

#mkdir sm2
#mkdir sm6
#mkdir sm7
#cp /root/gpgpu-sim_distribution/configs/tested-cfgs/SM2_GTX480/* ./sm2/
#cp /root/gpgpu-sim_distribution/configs/tested-cfgs/SM6_TITANX/* ./sm6/
#cp /root/gpgpu-sim_distribution/configs/tested-cfgs/SM7_TITANV/* ./sm7/
#sed -i 's/power_simulation_enabled\ 1/power_simulation_enabled\ 0/g' sm2/gpgpusim.config
#cp /root/FFT_reference/test/*.txt ./sm2/
#cp /root/FFT_reference/test/*.txt ./sm6/
#cp /root/FFT_reference/test/*.txt ./sm7/
