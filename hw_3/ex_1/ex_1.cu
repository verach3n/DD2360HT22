#include <random>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#define DataType double
#define TPB 64

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<len)
    out[i] = in1[i] + in2[i];
}

//@@ Insert code to implement timer start
//@@ Insert code to implement timer stop
double Second() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  
  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  size_t vec_size = inputLength*sizeof(DataType);
  hostInput1 = (DataType*) malloc(vec_size);
  hostInput2 = (DataType*) malloc(vec_size);
  hostOutput = (DataType*) malloc(vec_size);
  resultRef = (DataType*) malloc(vec_size);

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  std::default_random_engine gen{};
  std::uniform_real_distribution<DataType> dist{0, 1.0};
  for (int i=0; i<inputLength; i++) { 
    hostInput1[i] = dist(gen); 
    hostInput2[i] = dist(gen); 
    resultRef[i] =  hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, vec_size);
  cudaMalloc(&deviceInput2, vec_size);
  cudaMalloc(&deviceOutput, vec_size);

  //@@ Insert code to below to Copy memory to the GPU here
  double iStart1 = Second();
  cudaMemcpy(deviceInput1, hostInput1, vec_size,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, vec_size,cudaMemcpyHostToDevice);
  double iElaps1 = Second() - iStart1;

  //@@ Initialize the 1D grid and block dimensions here
  int Dg, Db;
  Dg = (inputLength + TPB - 1) / TPB;
  Db = TPB;

  //@@ Launch the GPU Kernel here
  double iStart2 = Second();
  vecAdd<<<Dg, Db>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double iElaps2 = Second() - iStart2;

  //@@ Copy the GPU memory back to the CPU here
  double iStart3 = Second();
  cudaMemcpy(hostOutput, deviceOutput, vec_size, cudaMemcpyDeviceToHost);
  double iElaps3 = Second() - iStart3;

  //@@ Insert code below to compare the output with the reference
  DataType max_diff = 1e-9;
  for (int i=0; i < inputLength; i++) {
    if (abs(hostOutput[i]-resultRef[i]) > max_diff) 
      printf("Error!\n");
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  printf("Data copy from host to device time: %f\n", iElaps1);
  printf("Kernel launching time: %f\n", iElaps2);
  printf("Data copy from device to host time: %f\n", iElaps3);
  return 0;
}
