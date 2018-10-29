#include<iostream>
#include<cstdio>
#include<fstream>
#include<cmath>
#include<string>
#include<iomanip>
#include<assert.h>
#include"cudaerr.h"

using namespace std;

__global__  void calcacf(double* d_fluxx, double* d_fluxy, double* d_fluxz,
    double* d_acf,int tcor, int nconf)
{ 
  double a0[3],acc=0.0l;
  int dt=blockDim.x*blockIdx.x+threadIdx.x;
  if (dt<=tcor) {
    for(int t0=0;t0<nconf-dt;t0++){
      a0[0]=d_fluxx[t0];
      a0[1]=d_fluxy[t0];
      a0[2]=d_fluxz[t0];
      acc=acc+a0[0]*d_fluxx[t0+dt]+a0[1]*d_fluxy[t0+dt]+a0[2]*d_fluxz[t0+dt];
    }
    d_acf[dt]=acc;
  }
}

extern "C"
{
  void calcacf_(double *h_fluxx,double *h_fluxy,double *h_fluxz,
  double *h_acf,
  int *fnconf, int *ftcor)
  { 
    int nconf=*fnconf,tcor=*ftcor+1;
    double *d_fluxx,*d_fluxy,*d_fluxz;
    double *d_acf;

    int nthreads=128;
    int near2;

     unsigned long long int sized=nconf*sizeof(double);
     unsigned long long int sizef=tcor*sizeof(double);

     memset(h_acf,0,sizef);
     
     ERR(cudaMalloc((void**)&d_fluxx,sized));
     ERR(cudaMalloc((void**)&d_fluxy,sized));
     ERR(cudaMalloc((void**)&d_fluxz,sized));
     ERR(cudaMalloc((void**)&d_acf,sizef));
   
     HANDLE_ERROR ( cudaGetLastError());
     ERR(cudaMemset(d_acf,0,sizef));
     memset(h_acf,0,sizef);
     
     near2=nthreads*(int(tcor/nthreads)+1);

     ERR(cudaMemcpy(d_fluxx,h_fluxx,sized,cudaMemcpyHostToDevice) );
     ERR(cudaMemcpy(d_fluxy,h_fluxy,sized,cudaMemcpyHostToDevice) );
     ERR(cudaMemcpy(d_fluxz,h_fluxz,sized,cudaMemcpyHostToDevice) );

     HANDLE_ERROR ( cudaGetLastError());
     calcacf<<<near2/nthreads,nthreads>>>
     (d_fluxx,d_fluxy,d_fluxz,d_acf,tcor,nconf);
     
     cudaDeviceSynchronize();

     ERR(cudaMemcpy(h_acf,d_acf,sizef,cudaMemcpyDeviceToHost) );
     cudaDeviceSynchronize();
     ERR(cudaFree(d_fluxx));
     ERR(cudaFree(d_fluxy));
     ERR(cudaFree(d_fluxz));
     ERR(cudaFree(d_acf));
     ERR(cudaGetLastError());
  }
}
