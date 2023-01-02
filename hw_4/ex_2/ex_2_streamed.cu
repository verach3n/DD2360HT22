#include <random>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_profiler_api.h>

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

  int inputLength, nStreams, S_seg;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  
  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  nStreams = atoi(argv[2]);
  S_seg = inputLength / nStreams;
  printf("The input length is %d\n", inputLength);
  printf("The number of streams is %d\n", nStreams);

  //@@ Insert code below to allocate Host memory for input and output
  size_t vec_size = inputLength*sizeof(DataType);
  cudaHostAlloc((void **) &hostInput1, vec_size, cudaHostAllocDefault);
  cudaHostAlloc((void **) &hostInput2, vec_size, cudaHostAllocDefault);
  cudaHostAlloc((void **) &hostOutput, vec_size, cudaHostAllocDefault);
  resultRef = (DataType*)malloc(vec_size);

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

  //@@ Initialize the 1D grid and block dimensions here
  const int streamSize = S_seg;
  const int streamBytes = streamSize*sizeof(DataType);
  int Dg, Db;
  Dg = (streamSize + TPB - 1) / TPB;
  Db = TPB;

  //@@ Insert code to below to Copy memory and launch the kernel here
  cudaStream_t stream[nStreams];
  double Start = Second();
  cudaProfilerStart();
  for(int i=0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);
  for(int i=0; i < nStreams; ++i){
    int offset=i*streamSize;
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    vecAdd<<<Dg,Db,0,stream[i]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, S_seg);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
  }
  for(int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(stream[i]);
  }
  cudaDeviceSynchronize();
  cudaProfilerStop();
  double Elaps = Second() - Start;
  printf("Time2: %f\n", Elaps);

  //@@ Insert code below to compare the output with the reference
  DataType max_diff = 1e-9;
  for (int i=0; i < inputLength; i++) {
    if (abs(hostOutput[i]-resultRef[i]) > max_diff) 
      printf("Error!\n");
      //printf("cpu:%f, gpu:%f\n", resultRef[i], hostOutput[i]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFree(hostInput1);
  cudaFree(hostInput2);
  cudaFree(hostOutput);
  free(resultRef);

  //printf("Data copy from host to device time: %f\n", iElaps1);
  //printf("Kernel launching time: %f\n", iElaps2);
  //printf("Data copy from device to host time: %f\n", iElaps3);
  return 0;
}
