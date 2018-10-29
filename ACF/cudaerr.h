//Error handling - specially usefull for not enough memory 

static void HandleError( cudaError_t err,
                         const char *file,
                        int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
    file, line );
    exit( EXIT_FAILURE );
  }
}
#define ERR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
/////////////////////////////////////////////////////////////////////

