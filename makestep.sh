#!/bin/bash

## This is meant to be used in docker image, "michael604/comparch_430.636_spring2020"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

rm -rf build&&mkdir build;
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 
make -j
mkdir sm2
mkdir sm6
mkdir sm7
cp /root/gpgpu-sim_distribution/configs/tested-cfgs/SM2_GTX480/* ./sm2/
cp /root/gpgpu-sim_distribution/configs/tested-cfgs/SM6_TITANX/* ./sm6/
cp /root/gpgpu-sim_distribution/configs/tested-cfgs/SM7_TITANV/* ./sm7/

sed -i 's/power_simulation_enabled\ 1/power_simulation_enabled\ 0/g' sm2/gpgpusim.config

cp /root/FFT_reference/test/*.txt ./sm2/
cp /root/FFT_reference/test/*.txt ./sm6/
cp /root/FFT_reference/test/*.txt ./sm7/

