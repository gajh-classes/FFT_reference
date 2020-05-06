export LD_LIBRARY_PATH=/usr/local/cuda/lib64

rm -rf build&&mkdir build;
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 
make -j
#mkdir sm2
#mkdir sm7
#cp ../SM2/* ./sm2/
#cp ../SM7/* ./sm7/

