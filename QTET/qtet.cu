#include<iostream>
#include<cstdio>
#include<fstream>
#include<cmath>
#include<string>
#include<iomanip>
#include<assert.h>
#include"cudaerr.h"
#include"dcdread.h"

using namespace std;

__global__  void qtetgpu(double* d_x, double* d_y, double* d_z,
double *d_dx, double *d_dy, double *d_dz,double * d_dr,
double* d_qtet,double xbox, double ybox, double zbox, int numatm) {
 double r,dx,dy,dz,cosab,cosabacc=0.0f;
 int d_neigh,j;
 int numatmby10=numatm/10;
 double box;
 box=min(xbox,ybox);
 box=min(box,zbox);

 double cut=box*0.2f;
 double r38=(3.0l/8.0l);
 double r13=(1.0l/3.0l);
 double ab,rj,rk;
 double n1,n2,n3,n4;
 n1=xbox;
 n2=xbox;
 n3=xbox;
 n4=xbox;

 unsigned long long int i = blockIdx.x * blockDim.x + threadIdx.x;
 if ( i < numatm ) {
    d_neigh=0;
    for (j=0;j<numatm;j++){
      if (i!=j) {
        dx=d_x[i]-d_x[j];
        dy=d_y[i]-d_y[j];
        dz=d_z[i]-d_z[j];
        dx=dx-xbox*(round(dx/xbox));
        dy=dy-ybox*(round(dy/ybox));
        dz=dz-zbox*(round(dz/zbox));

        r=sqrt(dx*dx+dy*dy+dz*dz);
        if (r<cut) {
            d_dx[i*numatmby10+d_neigh]=dx;
            d_dy[i*numatmby10+d_neigh]=dy;
            d_dz[i*numatmby10+d_neigh]=dz;
            d_dr[i*numatmby10+d_neigh]=r;
            d_neigh++;
        }
      }
     }
double r1,r2,tmp;
 for (int k=0;k<5;k++) {
  for (int j=d_neigh-1;j>0;j--) {
     r1=d_dr[i*numatmby10+j];
     r2=d_dr[i*numatmby10+j-1];
     if (r2>r1){
       tmp=d_dr[i*numatmby10+j];
       d_dr[i*numatmby10+j]=d_dr[i*numatmby10+j-1];
       d_dr[i*numatmby10+j-1]=tmp;

       tmp=d_dx[i*numatmby10+j];
       d_dx[i*numatmby10+j]=d_dx[i*numatmby10+j-1];
       d_dx[i*numatmby10+j-1]=tmp;

       tmp=d_dy[i*numatmby10+j];
       d_dy[i*numatmby10+j]=d_dy[i*numatmby10+j-1];
       d_dy[i*numatmby10+j-1]=tmp;

       tmp=d_dz[i*numatmby10+j];
       d_dz[i*numatmby10+j]=d_dz[i*numatmby10+j-1];
       d_dz[i*numatmby10+j-1]=tmp;
    }
   }
  }

   cosabacc=0.0f;
   ab=0.0f;
   for (j=0;j<3;j++) {
     rj=d_dr[i*numatmby10+j];
     for (int k=j+1;k<4;k++) {
       ab =d_dx[i*numatmby10+j]*d_dx[i*numatmby10+k];
       ab+=d_dy[i*numatmby10+j]*d_dy[i*numatmby10+k];
       ab+=d_dz[i*numatmby10+j]*d_dz[i*numatmby10+k];

       rk=d_dr[i*numatmby10+k];
       cosab=ab/(rj*rk);
       cosabacc+=(cosab+r13)*(cosab+r13);
     }
   }
   d_qtet[i]=1.0f-(r38*cosabacc);
 }
} 

extern "C"
  void calcqtet_(double *h_x,double *h_y,double *h_z,
  double *h_qtet,double *h_xbox, double *h_ybox, double *h_zbox, int *h_numatm) { 
    double *d_x,*d_y,*d_z;
    double *d_dx,*d_dy,*d_dz,*d_dr;
    double *d_qtet;
    double xbox=*h_xbox;
    double ybox=*h_ybox;
    double zbox=*h_zbox;
    int numatm=*h_numatm;
    int nthreads=128;
    int near2;
    
    numatm/(xbox*ybox*zbox);
   

     unsigned long long int sized1=numatm*sizeof(double);
     unsigned long long int sized2=numatm*(numatm/10)*sizeof(double);

     ERR(cudaSetDevice(0));
     ERR(cudaMalloc((void**)&d_x,sized1));
     ERR(cudaMalloc((void**)&d_y,sized1));
     ERR(cudaMalloc((void**)&d_z,sized1));
     ERR(cudaMalloc((void**)&d_dx,sized2));
     ERR(cudaMalloc((void**)&d_dy,sized2));
     ERR(cudaMalloc((void**)&d_dz,sized2));
     ERR(cudaMalloc((void**)&d_dr,sized2));
     ERR(cudaMalloc((void**)&d_qtet,sized1));
   

     HANDLE_ERROR(cudaGetLastError());
     
     near2=nthreads*(int(numatm/nthreads)+1);

     ERR(cudaMemcpy(d_x,h_x,sized1,cudaMemcpyHostToDevice) );
     ERR(cudaMemcpy(d_y,h_y,sized1,cudaMemcpyHostToDevice) );
     ERR(cudaMemcpy(d_z,h_z,sized1,cudaMemcpyHostToDevice) );

     HANDLE_ERROR ( cudaGetLastError());
     qtetgpu<<<near2/nthreads,nthreads>>>
     (d_x,d_y,d_z,
     d_dx,d_dy,d_dz,d_dr,
     d_qtet,xbox,ybox,zbox,numatm);
     
     cudaDeviceSynchronize();

     ERR(cudaMemcpy(h_qtet,d_qtet,sized1,cudaMemcpyDeviceToHost) );
     cudaDeviceSynchronize();

     ERR(cudaFree(d_x));
     ERR(cudaFree(d_y));
     ERR(cudaFree(d_z));
     ERR(cudaFree(d_dx));
     ERR(cudaFree(d_dy));
     ERR(cudaFree(d_dz));
     ERR(cudaFree(d_dr));
     ERR(cudaFree(d_qtet));



     ERR(cudaGetLastError());
  }
