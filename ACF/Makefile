all: full

full: acf-fort.o acf.o
	gfortran -L/home/soft/cuda-7.0/lib64 -lstdc++ -lcudart -lcuda -lgfortran acf-fort.o acf.o -o acf.out

acf-fort.o: acf-fort.f
	gfortran -c acf-fort.f -o acf-fort.o

acf.o: acf.cu
	nvcc -c -arch=sm_35 acf.cu -o acf.o

clean:
	rm -rvf acf-fort.o acf.o acf.out

