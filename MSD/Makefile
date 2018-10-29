CFLAGS=-lboost_program_options 
CUDAFLAGS=-arch=sm_35


all: full

full: msdcuda.o 
	nvcc $(CFLAGS) $(CUDAFLAGS) msdcuda.o -o msdcuda.x

msdcuda.o: CUDA_MSD.cu  cudaerr.h  dcdread-s.h
	nvcc $(CUDAFLAGS) -c CUDA_MSD.cu -o msdcuda.o
    
clean:
	rm -rvf msdcuda.o msdcuda.x


