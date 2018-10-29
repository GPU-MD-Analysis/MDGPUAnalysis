all: full

full: mainqtet.o qtet.o
	gfortran -L/home/soft/cuda-7.0/lib64 -lstdc++ -lcudart -lcuda -lgfortran mainqtet.o  qtet.o -o qtet.x

mainqtet.o: mainqtet.f
	gfortran -mcmodel=medium -Wunused -c mainqtet.f -o mainqtet.o 
qtet.o: qtet.cu
	nvcc -c -arch=sm_35 qtet.cu -o qtet.o

clean:
	rm -rvf mainqtet.o qtet.o qtet.x

