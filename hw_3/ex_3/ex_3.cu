#include <random>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_BINS 4096
#define TPB 64
#define MAX_SIZE 127
#define DataType unsigned int

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
//@@ Insert code below to compute histogram of input using shared memory and atomics

  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  const int tid = threadIdx.x;

  __shared__ unsigned int block_histogram[NUM_BINS];
  // Initialize the block histogram to 0 in each thread no.0
  if(tid == 0){
    for (int i = 0; i < num_bins; i++) {
      block_histogram[i] = 0;
    }
  }
  __syncthreads();

  if (id < num_elements) {
    // Calculate the bin index for the current data point
    // Use atomicAdd to add 1 to the bin count in the block histogram
    atomicAdd(&block_histogram[input[id]], 1);
  }
  __syncthreads();

  // Add the block histogram to the global histogram using atomicAdd
  if (tid == 0) {
    for(int i=0; i < num_bins; i++)
      atomicAdd(&bins[i], block_histogram[i]);
  }
  __syncthreads();
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(bins[id] > MAX_SIZE) bins[id] = MAX_SIZE;
}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  size_t vec_size1 = inputLength*sizeof(DataType);
  size_t vec_size2 = NUM_BINS*sizeof(DataType);
  hostInput = (DataType*) malloc(vec_size1);
  hostBins = (DataType*) malloc(vec_size2);
  resultRef = (DataType*) malloc(vec_size2);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  //std::default_random_engine gen{};
  //std::uniform_real_distribution<DataType> dist{0, (float)(NUM_BINS - 1)};
  for(int i=0; i<inputLength; i++) { 
    hostInput[i] = rand() % NUM_BINS; 
  }

  //@@ Insert code below to create reference result in CPU
  for(int i = 0; i < NUM_BINS; i++){
    resultRef[i] = 0;
  }
  for(int i = 0; i < inputLength; i++){
    resultRef[hostInput[i]]++;
  }
  for (int i = 0; i < NUM_BINS; i++){
    if (resultRef[i] > MAX_SIZE)
      resultRef[i] = MAX_SIZE;
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, vec_size1);
  cudaMalloc(&deviceBins, vec_size2);

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, vec_size1, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, vec_size2, cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
  
  //@@ Initialize the grid and block dimensions here
  int Dg1, Db1;
  Dg1 = (inputLength + TPB - 1) / TPB;
  Db1 = TPB;

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<Dg1, Db1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Initialize the second grid and block dimensions here
  int Dg2, Db2;
  Dg2 = ( NUM_BINS + TPB - 1) / TPB;
  Db2 = TPB;

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<Dg2, Db2>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, vec_size2, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  DataType max_diff = 0;
  for (int i=0; i < NUM_BINS; i++) {
    if (abs((int)(hostBins[i]-resultRef[i])) > max_diff) {
      printf("Error!\n");
      printf("host:%d, device:%d\n",resultRef[i],hostBins[i]);
    }
    else{
      printf("Equal!\n");
      printf("host:%d, device:%d\n",resultRef[i],hostBins[i]);
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  FILE *fp = fopen("ex_3.csv", "w+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
  for (int i=0; i < NUM_BINS; i++) {
    fprintf(fp, "%d\n",hostBins[i]);
  }
    fclose(fp);
    return 0;

  return 0;
}

