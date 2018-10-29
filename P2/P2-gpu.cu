#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <cstdio>



using namespace std;


__global__ void p2_calc_gpu(float* d_x, float* d_y, float* d_z, float* d_ans, int* d_count, int* d_status, unsigned long long int numatm,float* d_xbox,float* d_ybox,float* d_zbox, int numfrm)
{
 int numcount;
 float r,cut,dx,dy,dz;
 float px,py,pz,nx,ny,nz,rp,rn,rpn,vcos,accu;
 cut=8.75;
 int conf,nat;
 unsigned long long int i = blockIdx.x * blockDim.x + threadIdx.x;
    
 if ( i < numatm*numfrm) {
    conf=int(i/numatm);
    nat=int(i%numatm);
    d_ans[i]=0;   
    d_count[i]=0;
    accu=0;
    numcount=0;
    if (d_status[nat]!=1){
        d_count[i]=d_count[i]+1;
        for (int j=0;j<numatm;j++){
             if (d_status[j]!=1){
             dx=d_x[i]-d_x[conf*numatm+j];
             dy=d_y[i]-d_y[conf*numatm+j];
             dz=d_z[i]-d_z[conf*numatm+j];

             dx=dx-d_xbox[conf]*(round(dx/d_xbox[conf]));
             dy=dy-d_ybox[conf]*(round(dy/d_ybox[conf]));
             dz=dz-d_zbox[conf]*(round(dz/d_zbox[conf]));

             r=sqrt(dx*dx+dy*dy+dz*dz);
             if (r<cut) {
                  
                   px=d_x[i+1]-d_x[i-1] ;     
                   py=d_y[i+1]-d_y[i-1] ;      
                   pz=d_z[i+1]-d_z[i-1] ;       
       
                   nx=d_x[conf*numatm+j+1]-d_x[conf*numatm+j-1] ;        
                   ny=d_y[conf*numatm+j+1]-d_y[conf*numatm+j-1] ;         
                   nz=d_z[conf*numatm+j+1]-d_z[conf*numatm+j-1] ;          
          
                   rp=sqrt(px*px+py*py+pz*pz);
                   rn=sqrt(nx*nx+ny*ny+nz*nz);
                   rpn= (nx*px+ny*py+nz*pz);
                   
                   vcos= rpn/(abs(rp)*abs(rn));
                   //cout<<i<<"\t"<<j<<"\t"<<vcos<<endl;
                   accu=accu+((3*vcos*vcos-1)/2.0);
                    numcount++;}
              } 
            }    
         if (numcount==0){
           d_ans[i]=0;
         }else{
         d_ans[i]=accu/numcount;    
         }
      }
    }
 }

int main()
{ 
 // int a,b,c;
  unsigned long long int numatm;
  int numfrm,countsum;
  int* h_count,*d_count;
  int* h_status,*d_status;
  float* h_xbox,*h_ybox,*h_zbox;
  float* d_xbox,*d_ybox,*d_zbox;
  float sum;  
  float* h_ans,*d_ans;
  float* h_x,*h_y,*h_z;
  float* d_x,*d_y,*d_z;
  char atmnm[2];
  string comment;
  unsigned long long int near2;
  int nthreads=128;
  int totalcount;
  float totalsum;
 
  
  ifstream infile,pbc,bond;
  ofstream outfile,p2;
  infile.open("XYZ.dat");
  bond.open("bond.dat");
  pbc.open("pbc.dat");
  p2.open("avrg.p2");
  outfile.open("local-order.dat");


bond>>numatm;
bond>>numfrm;
//numfrm=1;


//  cout<<numatm<<"\t"<<near2<<endl;
  unsigned int sizef= numfrm*numatm*sizeof(float);
  unsigned int sizef2= numfrm*sizeof(float);

  unsigned int sizei= numatm*sizeof(int);
  unsigned int sizei2= numfrm*numatm*sizeof(int);

  h_x= (float *) malloc(sizef);
  h_y= (float *) malloc(sizef);
  h_z= (float *) malloc(sizef);
  h_ans=  (float *) malloc(sizef);
  h_status= (int *) malloc(sizei);
  h_count= (int *) malloc(sizei2);
  h_xbox=  (float *) malloc(sizef2);
  h_ybox=  (float *) malloc(sizef2);
  h_zbox=  (float *) malloc(sizef2);

  cudaMalloc((void**)&d_x, sizef);
  cudaMalloc((void**)&d_y, sizef);
  cudaMalloc((void**)&d_z, sizef);
  cudaMalloc((void**)&d_ans, sizef);
  cudaMalloc((void**)&d_status, sizei);
  cudaMalloc((void**)&d_count, sizei2);
  cudaMalloc((void**)&d_xbox, sizef2);
  cudaMalloc((void**)&d_ybox, sizef2);
  cudaMalloc((void**)&d_zbox, sizef2);

//  bond>>h_status[0]>>a>>b;
  for (int i=0;i<numatm;i++) {
       bond>>h_status[i];
  }
/****copying status of carbon here as it need not to be copy again & again  ************/
  cudaMemcpy(d_status, h_status, sizei, cudaMemcpyHostToDevice);

/*****calculation over different frames **************************************/
  totalcount=0;
  totalsum=0;

  for (int k=0;k<numfrm;k++) {
    if(!(k%100)){
       cout<<"#"<<k<<endl;
    }
       pbc>>h_xbox[k]>>h_ybox[k]>>h_zbox[k];
      
      infile>>numatm;
      getline(infile,comment);
      getline(infile,comment);

      for (int i=0;i<numatm;i++) {
           infile>>atmnm>>h_x[k*numatm+i]>>h_y[k*numatm+i]>>h_z[k*numatm+i];
      }  
     }
  
      cudaMemcpy(d_x, h_x, sizef, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, h_y, sizef, cudaMemcpyHostToDevice);
      cudaMemcpy(d_z, h_z, sizef, cudaMemcpyHostToDevice);
      cudaMemcpy(d_xbox, h_xbox, sizef2, cudaMemcpyHostToDevice);
      cudaMemcpy(d_ybox, h_ybox, sizef2, cudaMemcpyHostToDevice);
      cudaMemcpy(d_zbox, h_zbox, sizef2, cudaMemcpyHostToDevice);

  
      near2=nthreads*(int(numatm*numfrm/nthreads)+1);
   
      p2_calc_gpu<<<near2/nthreads,nthreads >>>(d_x, d_y, d_z, d_ans, d_count, d_status, numatm,d_xbox,d_ybox,d_zbox, numfrm);
  
      cudaMemcpy(h_ans, d_ans, sizef, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_count, d_count,sizei2,  cudaMemcpyDeviceToHost);
      cout<<h_ans[2]<<" "<<h_ans[10]<<"\n";
      for (int k=0;k<numfrm;k++) {
      sum=0;
      countsum=0;
      for (int i=0;i<numatm;i++){
        if (h_status[i]==1){
      outfile<<i<<"\t"<<9999<<" "<<endl;
      }
        if (h_status[i]!=1){
      outfile<<i<<"\t"<<h_ans[k*numfrm+i]<<endl;
      sum=sum+h_ans[k*numfrm+i];
      countsum=countsum+h_count[k*numfrm+i];
      }
      }
      outfile<<"#"<<sum/countsum<<endl;
      p2<<k<<" "<<sum/countsum<<endl;
      totalsum=totalsum+sum;
      totalcount=totalcount+countsum;
}     
      outfile<<"##final value is : "<<(totalsum/totalcount)<<endl;
      p2.close();
      outfile.close();

/**************************************************************************************/

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_ans);
        cudaFree(d_status);
        cudaFree(d_count);
        cudaFree(d_xbox);
        cudaFree(d_ybox);
        cudaFree(d_zbox);

        free(h_x);
        free(h_y);
        free(h_z);
        free(h_ans);
        free(h_status);
        free(h_count);
        cudaFree(h_xbox);
        cudaFree(h_ybox);
        cudaFree(h_zbox);

  return 0;
}


