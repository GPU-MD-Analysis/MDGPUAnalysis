#include<iostream>
#include<cstdio>
#include<fstream>
#include<cmath>
#include<string>
#include<iomanip>
#include<assert.h>
#include"dcdread-s.h"
#include"cudaerr.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;


 __global__ void unwrap(double* d_x, double* d_y, double*
 d_z, int numatm, int nconf,double xbox, double ybox, double zbox);

  __global__ void msd_gpu
  ( double* d_x, double* d_y, double* d_z,float* d_acc, int numatm, int nconf);

int main(int argc , char* argv[] )
{ 
  int numatm,nconf,device,inconf;
  double xbox,ybox,zbox,tstep;
  double* h_x,*h_y,*h_z;
  double* d_x,*d_y,*d_z;
  float *h_acc,*d_acc;
  int nthreads;
  unsigned int near2;
  string file;

///////////////////////////////////////////////////////////////
  po::options_description desc("Options");
  desc.add_options()
 
  ("help,h","Display help")
  ("device,d",   po::value<int>(&device)->default_value(1),   "CUDA device to use (0 or 1)")
  ("nconf,m",    po::value<int>(&inconf)->default_value(1000),     "number ofconfigurations")
  ("nthreads,k", po::value<int>(&nthreads)->default_value(128), "number of threads")
  ("tstep,t",    po::value<double>(&tstep)->default_value(1), "time-interval in ps between two frames")
  ("filename,f", po::value<string>(&file)->default_value("traj.dcd"),"trajectory file");

////////////////////////////////////////////////////////////////////////
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   if ( vm.count("help") || vm.count("h") || argc == 1) {
     std::cout << "MSD Calculation Program"<<std::endl
     << desc << std::endl
     << "(defaults are in brackets)\n"
     << "Example: "
     <<argv[0]<<" -d 1 -m 5000 -k 128 -t 1 -f traj.dcd"
     <<std::endl;
     return 0;
   }
   po::notify(vm);
   cout<<"Using these options for "<<argv[0]<<endl
   <<" -d "<<device
   <<" -t "<<tstep
   <<" -k "<<nthreads
   <<" -m "<<inconf
   <<" -f "<<file
   <<endl;
   cout<<"\nHelp available using "<<argv[0]<<" -h\n\n";

////////////////////////////////////////////////////////////////////
  HANDLE_ERROR (cudaSetDevice(device));//pick the device to use
  
  std::ifstream infile;
  infile.open(file.c_str());
  if(!infile){
    cout<<"file "<<file.c_str()<<" not found\n";
    return 1;
   }
   assert(infile);
/////////////////////////////////////////////////////////
  dcdreadhead(&numatm,&nconf,infile);
  if (inconf>nconf) cout << "nconf is reset to "<< nconf <<endl;
  else
  cout<<"your dcd file has"<< numatm << "atoms with" << nconf << "frames"<<endl;
  {nconf=inconf;}
  cout<<"Calculation initiating for" << nconf << "frames"<<endl;
////////////////////////////////////////////////////////
  near2=nthreads*(int(numatm/nthreads)+1);
  unsigned long long int sized=numatm*nconf*sizeof(double);
  unsigned long long int sizef=nconf*sizeof(float);

///////////////////////////////////////////////////////////////
  HANDLE_ERROR (cudaHostAlloc((void**)&h_x,sized,cudaHostAllocDefault));
  HANDLE_ERROR (cudaHostAlloc((void**)&h_y,sized,cudaHostAllocDefault));
  HANDLE_ERROR (cudaHostAlloc((void**)&h_z,sized,cudaHostAllocDefault));
  HANDLE_ERROR (cudaHostAlloc((void**)&h_acc,sizef,cudaHostAllocDefault));
  memset(h_acc,0,sizef);

  HANDLE_ERROR (cudaMalloc((void**)&d_x,sized));
  HANDLE_ERROR (cudaMalloc((void**)&d_y,sized));
  HANDLE_ERROR (cudaMalloc((void**)&d_z,sized));
  HANDLE_ERROR (cudaMalloc((void**)&d_acc,sizef));
  HANDLE_ERROR (cudaMemset(d_acc,0,sizef));

////////////////reading of coordinates>>>>>>>>>>>>>
 
 double ax[numatm],ay[numatm],az[numatm];

  for (int i=0;i<nconf;i++) {
     dcdreadframe(ax,ay,az,infile,numatm,xbox,ybox,zbox);
     for (int j=0;j<numatm;j++){
       h_x[i*numatm+j]=ax[j];
       h_y[i*numatm+j]=ay[j];
       h_z[i*numatm+j]=az[j];
     }
  }
///////////////////////////////////////////////
  HANDLE_ERROR (cudaMemcpy(d_x,h_x,sized,cudaMemcpyHostToDevice) );
  HANDLE_ERROR (cudaMemcpy(d_y,h_y,sized,cudaMemcpyHostToDevice) );
  HANDLE_ERROR (cudaMemcpy(d_z,h_z,sized,cudaMemcpyHostToDevice) );

//////////////////////////////////////////////////////
  unwrap<<<near2/nthreads,nthreads>>>(d_x,d_y,d_z,numatm,nconf,xbox,ybox,zbox);
  cudaDeviceSynchronize();

  near2=nthreads*(int((numatm*(nconf-1))/nthreads)+1);

  msd_gpu<<<near2/nthreads,nthreads>>>(d_x,d_y,d_z,d_acc,numatm,nconf);
  cudaDeviceSynchronize();
///////////////////////////////////////////////////////////
  HANDLE_ERROR (cudaPeekAtLastError());
  HANDLE_ERROR (cudaThreadSynchronize());
  HANDLE_ERROR (cudaDeviceSynchronize());

//////////////////////////////////////////////////////////
  HANDLE_ERROR (cudaMemcpy(h_acc,d_acc,sizef,cudaMemcpyDeviceToHost));

////////////////////////////////////////////////////////////////
  ofstream outfile("Output.msd");        
          
  for (int i=0;i<(nconf-1);i++)
  outfile<<i*tstep<<" "
  <<((double)h_acc[i]/(double)(numatm*(nconf-i)))<<endl;

  HANDLE_ERROR ( cudaGetLastError() );
//////////////////////////////////////////////////////////////////////
  printf("\nby default timestep b/n conf is in ps\n");
  cout<<"Calculations was performed for "<<nconf<<" configurations\n"<<endl;
  cout<<"Using device  "<<device<<"\n"
  <<"Output is in `Output.msd'\n";

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_acc);
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  cudaFreeHost(h_z);
  cudaFreeHost(h_acc);
  return 0;

}

 __global__ void unwrap(double* d_x, double* d_y, double*
 d_z, int numatm, int nconf,double xbox, double ybox, double zbox) 
{ double a1,b1,c1;
  double px,qy,rz;
  int j=0;
  unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<numatm){
     j=0;
     a1=d_x[j*numatm+i];
     b1=d_y[j*numatm+i];
     c1=d_z[j*numatm+i];

     d_x[j*numatm+i]= d_x[j*numatm+i]*xbox;
     d_y[j*numatm+i]= d_y[j*numatm+i]*ybox;
     d_z[j*numatm+i]= d_z[j*numatm+i]*zbox;

     for (int j=1;j<nconf;j++){
         px=d_x[j*numatm+i]-a1;
         qy=d_y[j*numatm+i]-b1;
         rz=d_z[j*numatm+i]-c1;

         a1=d_x[j*numatm+i];
         b1=d_y[j*numatm+i];
         c1=d_z[j*numatm+i];

         px=px-round(px);
         qy=qy-round(qy);
         rz=rz-round(rz);

         px=px*xbox;
         qy=qy*ybox;
         rz=rz*zbox;

         d_x[(j)*numatm+i]= d_x[(j-1)*numatm+i]+px;
         d_y[(j)*numatm+i]= d_y[(j-1)*numatm+i]+qy;
         d_z[(j)*numatm+i]= d_z[(j-1)*numatm+i]+rz;
      }     
  }
}

__global__ void msd_gpu
( double* d_x, double* d_y, double* d_z, float* d_acc, int numatm, int nconf)
{ double dx,dy,dz;
  double acc;
  int dt;
  unsigned int i,j;

  i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<numatm*(nconf-1)) {
      dt=int(i/numatm);
      i=(i%numatm);
          acc=0;
         for(j=0;j<(nconf-dt);j++){
             dx=d_x[j*numatm+i]-d_x[(j+dt)*numatm+i];
             dy=d_y[j*numatm+i]-d_y[(j+dt)*numatm+i];
             dz=d_z[j*numatm+i]-d_z[(j+dt)*numatm+i];

            acc=acc+(double)(dx*dx+dy*dy+dz*dz);
          }
               atomicAdd(&d_acc[dt],acc);
  }
 }
