using namespace std;

static void HandleError( cudaError_t err,
                         const char *file,
                        int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
    file, line );
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int fortreadframe(double *x, double *y, double *z, 
std::istream &infile, int numatm){
  int ind=0;
  assert(infile);
  for (int i=0;i<numatm;i++) {
    infile>>ind>>x[i]>>y[i]>>z[i];
  }
  return 0;
}
