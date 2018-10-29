CFLAGS=-lboost_program_options 
CUDAFLAGS=-arch=sm_35


all: full

full: rdfcuda.o 
	nvcc $(CFLAGS) $(CUDAFLAGS) rdfcuda.o -o rdfcuda.x

rdfcuda.o: rdf.cu  cudaerr.h  dcdread.h
	nvcc $(CUDAFLAGS) -c rdf.cu -o rdfcuda.o
    
clean:
	rm -rvf rdfcuda.o rdfcuda.x



