#include <random>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define DataType float
#define bDim 16

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if ((col >= numBColumns) || (row >= numARows)) return;

  DataType temp=0;
  for(int i=0; i< numAColumns; i++){
    temp +=  A[row*numAColumns+i] * B[numBColumns*i+col];
  }
  C[row*numBColumns+col] = temp;
}

double Second() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBColumns = atoi(argv[3]);
  numBRows = numAColumns;
  numCRows = numARows;
  numCColumns = numBColumns;
  int m = numARows, k = numAColumns, n = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", m, k, k, n, m, n);
  
  //@@ Insert code below to allocate Host memory for input and output
  size_t vec_sizeA = m*k*sizeof(DataType);
  size_t vec_sizeB = k*n*sizeof(DataType);
  size_t vec_sizeC = m*n*sizeof(DataType);
  hostA = (DataType*) malloc(vec_sizeA);
  hostB = (DataType*) malloc(vec_sizeB);
  hostC = (DataType*) malloc(vec_sizeC);
  resultRef = (DataType*) malloc(vec_sizeC);
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  std::default_random_engine gen{};
  std::uniform_real_distribution<DataType> dist{0, 1.0};
  for(int i=0; i<m*k; i++) { 
    hostA[i] = dist(gen); 
  }
  for(int i=0; i<k*n; i++) {
    hostB[i] = dist(gen); 
  }
  
  for(int p=0; p < m; p++){
    for(int q=0; q < n; q++){
      DataType temp=0;
      for(int r=0; r < k; r++){
        temp += hostA[k*p+r] * hostB[r*n+q];
      }
      resultRef[p*n+q] = temp;
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, vec_sizeA);
  cudaMalloc(&deviceB, vec_sizeB);
  cudaMalloc(&deviceC, vec_sizeC);

  //@@ Insert code to below to Copy memory to the GPU here
  double iStart1 = Second();
  cudaMemcpy(deviceA, hostA, vec_sizeA,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, vec_sizeB,cudaMemcpyHostToDevice);
  double iElaps1 = Second() - iStart1;

  //@@ Initialize the grid and block dimensions here
  int Dgx, Dgy, Dbx, Dby;
  Dgy = (m + bDim - 1) / bDim;
  Dgx = (n + bDim - 1) / bDim;
  Dbx = bDim;
  Dby = bDim;

  //@@ Launch the GPU Kernel here
  double iStart2 = Second();
  gemm<<<dim3(Dgx, Dgy, 1), dim3(Dbx, Dby, 1)>>>(deviceA, deviceB, deviceC, m, k, k, n);
  cudaDeviceSynchronize();
  double iElaps2 = Second() - iStart2;

  //@@ Copy the GPU memory back to the CPU here
  double iStart3 = Second();
  cudaMemcpy(hostC, deviceC, vec_sizeC, cudaMemcpyDeviceToHost);
  double iElaps3 = Second() - iStart3; 

  //@@ Insert code below to compare the output with the reference
  DataType max_diff = 1e-2;
  int flag = 0;
  for (int i=0; i < m*n; i++) {
    if (abs(hostC[i]-resultRef[i]) > max_diff) {
      printf("Error!\n");
      printf("Host Calculated Value: %f\n", resultRef[i]);
      printf("Device Calculated Value: %f\n", hostC[i]);
      flag = 1;
    }
  }
  if(flag==0) printf("Equal!\n");

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  printf("Data copy from host to device time: %f\n", iElaps1);
  printf("Kernel launching time: %f\n", iElaps2);
  printf("Data copy from device to host time: %f\n", iElaps3);

  return 0;
}
