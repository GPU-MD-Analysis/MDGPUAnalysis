CFLAGS=-lboost_program_options -lgfortran
CUDAFLAGS=-arch=sm_35 -I/opt/ohpc/pub/libs/gnu/openmpi/boost/1.63.0/include -L/opt/ohpc/pub/libs/gnu/openmpi/boost/1.63.0/lib

all: full

full:  trip.gen.o 
	nvcc $(CFLAGS) $(CUDAFLAGS) trip.gen.o -o trip.x

trip.gen.o: trip.gen.cu trip.h
	nvcc $(CUDAFLAGS) -c trip.gen.cu -o trip.gen.o
    
clean:
	rm -rvf  trip.gen.o trip.x

