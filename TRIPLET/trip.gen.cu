/*---------------------------------------------------------------------
Compilation:
 f95 -c fortranread-analyze.f -o fort.o
 nvcc -c -arch=sm_35 trip.gen.cu  -o trip.gen.o
 nvcc -lboost_program_options -lgfortran trip.gen.o fort.o -o trip.x
 
 The Makefile is also available:
 $ make 
 will make the trip.x

 $ make clean 
 will remove the object files and the executable

Notes:
This assumes constant volume data. Modification to NPT is not difficult - need
to read and send xbox and cut each time.

This will write out two files g2.inp and g3.inp. They contain the binned data
for g2 and g3 respectively. These files will be read and processed by the
  FORTRAN subroutine.

trip.h has the function "fortreadframe" to read a single frame from the
trajectory file - modify it as per need - or define your own; if the trajectory
has extra header information which does NOT repeat each frame, a separate header
reading function needs to be written and called. e.g. for dcd files.

Some of the options are specific to the Kepler architecture. e.g.
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);


The code is quite a memory hog - It might not run on GTX 480 or below.
Typically, 4096 atoms use >1GB of memory
*/

#include <iostream> //normal i/o
#include <iomanip> // i/o manipulation
#include <fstream>  // file i/o
#include <string>   //File name is a string
#include <assert.h> //assert to catch undiagnosed problems
#include <boost/program_options.hpp> //for command line parameters
#include <time.h> //measure time 
#include "trip.h" //custom error handling and reading configurations
#include "dcdread.h"

namespace po = boost::program_options;

using namespace std;
__constant__ int d_numatm, d_nbin;
__constant__ double d_xbox, d_cut;

///////////FORTRAN Subroutine Prototype//////////////
extern "C" {
  void triplet_g3_(int *, int *, int *, double *, double *);
}

//Kernel prototypes are declared here. Defintions are at end of file.
//Kernel to calculate pairs
__global__ void pair_gpu(
const double* d_x, const double* d_y, const double* d_z, 
int* d_neigh, double *d_dx, double *d_dy, double *d_dz,int *d_g2
);

//Kernel to calculate triplets
__global__ void trip_gpu(
const double* d_dx, const double* d_dy, const double* d_dz, 
unsigned long long int* d_g3,
const int i, const int* d_neigh);

int main(int argc, char* argv[])
{
  clock_t tstart, tend, totstart,totend; //measure time
  tstart=clock();
  totstart=tstart;

  int numatm,device;
  double xbox,cut,sigma,dcdxbox;
  int istream, nbin,i,j,k;
  int* h_g2,*d_g2;
  unsigned long long int* h_g3,*d_g3;
  int *h_neigh,*d_neigh;
  int nstreams;
  double *h_x,*h_y,*h_z;
  double *d_x,*d_y,*d_z;
  double *d_dx,*d_dy,*d_dz;
  unsigned long long int near2;
  int nthreads,nconf,inconf;
  string file; 
/////////////////////////////////////////////////////////////////
///////Boost: command line options///////////////////////////////
  po::options_description desc("Options");
  desc.add_options()
  // Option 'numbin' and 'b' are equivalent.
  ("help,h","Display help")
  ("device,d",   po::value<int>(&device)->default_value(1),         "CUDA device to use (0 or 1)")
  ("nbin,b",     po::value<int>(&nbin)->default_value(300),         "number of bins")
  ("nconf,m",    po::value<int>(&inconf)->default_value(5),          "number of configurations")
  ("numatm,n",   po::value<int>(&numatm)->default_value(4096),      "number of atoms")
  ("xbox,x", po::value<double>(&xbox)->default_value(25.265l*2.0l),  "Cube box length")
  ("sigma", po::value<double>(&sigma)->default_value(2.3925l),  "Sigma")
  ("nthreads,t", po::value<int>(&nthreads)->default_value(128),     "number of threads")
  ("nstreams,s", po::value<int>(&nstreams)->default_value(14),
  "number of streams. On K20, 14 streams give the best performance")
  ("filename,f", po::value<string>(&file)->default_value("fort.91"),"trajectory file")
  ; //end add_options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if ( vm.count("help") || vm.count("h") || argc == 1) { 
    std::cout << "CUDA Triplet binning program" << std::endl 
    << desc << std::endl 
    << "(defaults are in brackets)\n"
    << "Example: "
    <<argv[0]<<" -d 1 -b 300 -m 5 -n 4096 -x 50.53 -t 128 -s 14 -f fort.91"
    <<std::endl; 
    return 0; 
  } 

  po::notify(vm);
  cout<<"Using these options for "<<argv[0]<<endl
  <<" -d "<<device
  <<" -b "<<nbin
  <<" -m "<<inconf
  <<" -n "<<numatm
  <<" -x "<<xbox
  <<" --sigma "<<sigma
  <<" -t "<<nthreads
  <<" -s "<<nstreams
  <<" -f "<<file
  <<endl;
  cout<<"\nHelp available using "<<argv[0]<<" -h\n\n";
////////////////////////////////////////////////////////  
  cudaSetDevice(device); //pick the device to use
  cut=xbox/2.0;// set the cutoff
////////////////////////////////////////////////////////
///////Check Device/////////////////////////////////////
  cudaDeviceProp prop;
  int whichDevice;
  HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
  HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
  if (!prop.deviceOverlap) {
    printf( "Device will not handle overlaps");
    return 0;
  }
  //we are using unsigned long long ints and doubles - all eight byte vars
  HANDLE_ERROR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
////////////////////////////////////////////////////////
//// Create streams ///////////////////////////////////
  cudaStream_t *streams=(cudaStream_t*) malloc(nstreams* sizeof(cudaStream_t));
  for(int i = 0; i < nstreams; i++)
     HANDLE_ERROR(cudaStreamCreate(&(streams[i])));
///////////////////////////////////////////////////////

  std::ifstream infile;
  infile.open(file.c_str());
  if(!infile){
    cout<<"file "<<file.c_str()<<" not found\n";
    return 1;
  }
  assert(infile);
  
  dcdreadhead(&numatm,&nconf,infile);
  if (inconf>nconf) cout << "nconf is reset to "<< nconf <<endl;
  else
  {nconf=inconf;}

  near2=nthreads*(int(numatm/nthreads)+1);

  unsigned long long int sizef=numatm*sizeof(double);
  unsigned long long int sizef2=numatm*(numatm/10)*sizeof(double);
  unsigned long long int sizei=numatm*sizeof(int);
  unsigned long long int sizebin=nbin*sizeof(unsigned long long int);
  unsigned long long int sizebin3=nbin*nbin*nbin*sizeof(unsigned long long int);

  HANDLE_ERROR(cudaHostAlloc((void **)&h_x, sizef, cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&h_y, sizef, cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&h_z, sizef, cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&h_neigh, sizei, cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&h_g2, sizebin, cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&h_g3, sizebin3, cudaHostAllocDefault));

  
  HANDLE_ERROR(cudaMalloc((void**)&d_x, sizef));
  HANDLE_ERROR(cudaMalloc((void**)&d_y, sizef));
  HANDLE_ERROR(cudaMalloc((void**)&d_z, sizef));
  HANDLE_ERROR(cudaMalloc((void**)&d_dx, sizef2));
  HANDLE_ERROR(cudaMalloc((void**)&d_dy, sizef2));
  HANDLE_ERROR(cudaMalloc((void**)&d_dz, sizef2));
  HANDLE_ERROR(cudaMalloc((void**)&d_neigh, sizei));
  HANDLE_ERROR(cudaMalloc((void**)&d_g2, sizebin));
  HANDLE_ERROR(cudaMalloc((void**)&d_g3, sizebin3));
 
//initialize host and device arrays to 0 
  memset(h_g2,0,sizebin);
  memset(h_g3,0,sizebin3);
  memset(h_neigh,0,sizei);
  HANDLE_ERROR(cudaMemset(d_neigh,0,sizei));
  HANDLE_ERROR(cudaMemset(d_g2,0,sizebin));
  HANDLE_ERROR(cudaMemset(d_g3,0,sizebin3));
 
  //copy over the various global constants
  HANDLE_ERROR(cudaMemcpyToSymbol(d_nbin,&nbin,sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(d_numatm,&numatm,sizeof(int)));
  //need to send this in the kernel if you want to do NPT
  HANDLE_ERROR(cudaMemcpyToSymbol(d_xbox,&xbox,sizeof(double)));
  HANDLE_ERROR(cudaMemcpyToSymbol(d_cut,&cut,sizeof(double)));


  cudaDeviceSynchronize();
  tend = clock();
  cout<<"\nInitialization took "<<(tend-tstart)/(double)CLOCKS_PER_SEC
  <<" seconds"<< endl <<"Starting processing of " 
  <<nconf<<" configurations...\n";
  tstart=clock();
  //Start the main configuration loop
  //read the header information here - if needed. 
  for (int iconf=0;iconf<nconf;iconf++) {
    //read frame
//    fortreadframe(h_x,h_y,h_z,infile,numatm);
      dcdreadframe(h_x,h_y,h_z,infile,numatm,dcdxbox);
      if (iconf==0) cout<<"DCD file has "<<numatm<<" atoms with a xbox length of "<<dcdxbox<<endl;
//transfer the coordinates
    HANDLE_ERROR(cudaMemcpy(d_x, h_x, sizef, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_y, h_y, sizef, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_z, h_z, sizef, cudaMemcpyHostToDevice));

    near2=nthreads*(int(numatm/nthreads)+1);
    pair_gpu<<<near2/nthreads,nthreads, 0,streams[0]>>>
    (d_x, d_y, d_z, d_neigh,
    d_dx, d_dy, d_dz,d_g2);
    HANDLE_ERROR(cudaDeviceSynchronize());

    //spawn independent streams for i's this way there is enough work
    // pipeline. dim3 or block-thread combination can also be used. 
    for(int i=0;i<numatm;i++){
      istream=i%nstreams;
      trip_gpu<<<near2/nthreads,nthreads,0,streams[istream]>>>
      (d_dx,d_dy,d_dz,d_g3,i,d_neigh);
    }

    HANDLE_ERROR(cudaDeviceSynchronize());

    if(((iconf+1)%10)==0) {
      cout<<iconf+1<<"...";
      cout.flush();
      if(((iconf+1)%1000)==0) {
        cout<<endl;
        cout.flush();
      }
    }
  }
  tend = clock();
  cout<<endl<<nconf
  <<" configurations took "<<(tend-tstart)/(double)CLOCKS_PER_SEC
  <<" seconds"<<endl;
  tstart=clock();

//  cudaMemcpy(h_neigh, d_neigh, sizei, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_g2, d_g2, sizebin, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_g3, d_g3, sizebin3, cudaMemcpyDeviceToHost);

  ofstream g3inp,g2inp;
  g2inp.open("g2.inp");
  g3inp.open("g3.inp");
  unsigned long long int g3;
  cout<<"\n# Writing out files to be read by fortran prog\n"
  <<" g2.inp and g3.inp ";

  g2inp<<nbin<<endl;
  g3inp<<nbin<<" "<<nconf<<endl;
  for(i=0;i<nbin;i++) {
    g2inp<<i+1<<" "<<h_g2[i]<<endl;
    for(j=0;j<nbin;j++) {
      for(k=0;k<nbin;k++) {
        g3=h_g3[i*nbin*nbin+j*nbin+k];
        if(g3!=0){
          //+1 for FORTRAN
          g3inp<<i+1<<" "<<j+1<<" "<<k+1<<" "
          <<setw(30)<<setprecision(20)<<g3<<endl;
        }
      }
    }
  }
  
  cout<<"... Written!\n";
  cout<<"\n\n\n#Freeing Device memory"<<endl;
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_neigh);
  cudaFree(d_dx);
  cudaFree(d_dy);
  cudaFree(d_dz);

  cout<<"#Freeing Host memory"<<endl;
  HANDLE_ERROR( cudaFreeHost ( h_x ) );
  HANDLE_ERROR( cudaFreeHost ( h_y ) );
  HANDLE_ERROR( cudaFreeHost ( h_z ) );
  HANDLE_ERROR( cudaFreeHost ( h_neigh ) );

  tend = clock();
  cout<<"Writing data and cleaning took "<<(tend-tstart)/(double)CLOCKS_PER_SEC
    <<" seconds"<< endl;

/////////////FORTRAN///////////////////////
  cout<<"\nCalling FORTRAN subroutine\n";
  cout.flush();
  tstart=clock();

  triplet_g3_(&nbin,&numatm,&nconf,&xbox,&sigma);
  
  tend = clock();
  cout<<"\nFORTRAN subroutine took "<<(tend-tstart)/(double)CLOCKS_PER_SEC
    <<" seconds"<< endl;
  cout.flush();
/////////////////////////////////////////
  totend=clock();
  cout<<"\nTotal time: "<<(totend-totstart)/(double)CLOCKS_PER_SEC
    <<" seconds"<< endl;
  cout.flush();
  return 0;
}

//d_dx/y/z[i] has the vector for the neighbours of i within box/4
//these will be used in the trip_gpu
__global__ void pair_gpu(
const double* d_x, const double* d_y, const double* d_z, 
int* d_neigh,double *d_dx, double *d_dy, double *d_dz,int * d_g2)
{
 double r,dx,dy,dz;
 int nnum=d_numatm/10;
 int ig2;
 double rdel=((double)(2.0*d_nbin))/d_xbox;
 double cut2=d_cut/2.0;
 unsigned long long int i = blockIdx.x * blockDim.x + threadIdx.x;
 if ( i < d_numatm ) {
    d_neigh[i]=0;
    for (int j=0;j<d_numatm;j++){
      if ( i!=j) {
        dx=d_x[i]-d_x[j];
        dy=d_y[i]-d_y[j];
        dz=d_z[i]-d_z[j];

        dx=dx-d_xbox*(round(dx/d_xbox));
        dy=dy-d_xbox*(round(dy/d_xbox));
        dz=dz-d_xbox*(round(dz/d_xbox));
        
        r=sqrt(dx*dx+dy*dy+dz*dz);
        if (r<d_cut) {
          ig2=trunc(r*rdel);
          atomicAdd(&d_g2[ig2],1) ;
          //store all neighbours within box/4. Indices are not relevant
          if (r<cut2) {
            d_dx[i*nnum+d_neigh[i]]=dx;
            d_dy[i*nnum+d_neigh[i]]=dy;
            d_dz[i*nnum+d_neigh[i]]=dz;
            d_neigh[i]++;
          }
        }
      }
    }
  }
}


//Triplet kernel - run for each i!
__global__ void trip_gpu( 
const double* d_dx, const double* d_dy, const double* d_dz, 
unsigned long long int* d_g3,
const int i,const int *d_neigh)
{
 int nnum=d_numatm/10;
  unsigned long long int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d_neigh[i]) {
    double r,dx,dy,dz;
    double rdel=((double)(2.0*d_nbin))/d_xbox; //inverse del
    int ijg3,jkg3,ikg3;

    //fetch rij 
    dx=d_dx[i*nnum+j];
    dy=d_dy[i*nnum+j];
    dz=d_dz[i*nnum+j];
    r=sqrt(dx*dx+dy*dy+dz*dz);
    ijg3=trunc(r*rdel);
    for( int k=0; k<d_neigh[i]; k++) {
      if(j!=k ) {
        //fetch rik
        dx=d_dx[i*nnum+k];
        dy=d_dy[i*nnum+k];
        dz=d_dz[i*nnum+k];
        r=sqrt(dx*dx+dy*dy+dz*dz);
        ikg3=trunc(r*rdel);

        //calc rjk
        dx=d_dx[i*nnum+j]-d_dx[i*nnum+k];
        dy=d_dy[i*nnum+j]-d_dy[i*nnum+k];
        dz=d_dz[i*nnum+j]-d_dz[i*nnum+k];
        r=sqrt(dx*dx+dy*dy+dz*dz);
        if(r<d_cut) {
          jkg3= trunc(r*rdel);
          atomicAdd(&d_g3[ijg3*d_nbin*d_nbin+ikg3*d_nbin+jkg3],1);
        }
      }
    }
  }
}
