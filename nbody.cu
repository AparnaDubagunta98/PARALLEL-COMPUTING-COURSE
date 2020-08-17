// ----------------------------------------------------------------------------
// CUDA code to compute minimun distance between n points
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<limits>
#include<float.h>

#define MAX_POINTS 1048576
#define block_size 1024

// ----------------------------------------------------------------------------
// Kernel Function to compute distance between all pairs of points
// Input:
//	X: X[i] = x-coordinate of the ith point
//	Y: Y[i] = y-coordinate of the ith point
//	n: number of points
// Output:
//	D: D[0] = minimum distance
//

// variable to keep track of number of blocks for which minimum calculation is done on the device
__device__ unsigned int block_count = 0;
__global__ void minimum_distance(float * X, float * Y, volatile float * D, int n) {

// current thread operating = block_x * blickdim_x * threadid_x
unsigned int curr_i = blockIdx.x * blockDim.x + threadIdx.x;

float dx,dy, curr_dist;
//intializing minimum distance to a float max value
float minimum_dist = FLT_MAX;
// to check if last block is completed or not
bool last_blk;

// sharing a list of local minimum distances calculated by all threads in a block
__shared__ float block_local_mins[block_size];

// to make sure current thread lies withing the range of points in the grid
if(curr_i < n-1)
{
  //calculating distance for ith point pointed by curr_i to all other points
  for(int j = curr_i+1;j<n;j++)
  {
    dx = X[curr_i] - X[j]; //xi- xj
    dy = Y[curr_i] - Y[j]; //yi- yj

    curr_dist = sqrtf(dx*dx + dy*dy) ;// dij = √(xi − xj)^2+ (yi − yj)^2
    //update minimum distance computed by this thread
    if(curr_dist < minimum_dist)
    {
      minimum_dist = curr_dist;
    }

  }
  //updating block mins list with the thread's minimum
  block_local_mins[threadIdx.x] = minimum_dist;
  //wait for all threads in that block to update this list
  __syncthreads();

  // compute block local minimum with logarithmic reduction (binary tree based approach)
  int stop_index = (n%block_size);
  if(stop_index == 0){
    stop_index = block_size;
  }
  else{
    if(blockIdx.x != n/block_size)
    {
        stop_index = block_size;
    }
  }

  // traverse through the block adn update minimum
  //step of 2
  for(int i=1;i<stop_index;i *=2){
    if(threadIdx.x % (2*i) == 0 && (threadIdx.x +i) < stop_index-1){
       if(block_local_mins[threadIdx.x] > block_local_mins[threadIdx.x + i]){
         block_local_mins[threadIdx.x] = block_local_mins[threadIdx.x + i];
       }
      __syncthreads();
    }
  }

  //move the minimum computed from the first thread in every block to the correct spot in D
  if(threadIdx.x == 0){
    D[blockIdx.x] = block_local_mins[0];
    //also increment block_count
    int inc_val = atomicInc(&block_count,gridDim.x);
    last_blk = (inc_val == (gridDim.x -1));
  }

  //make last thread in list compute global minimum and put it in D[0]
  if(last_blk && threadIdx.x ==0){
    int num_blocks = n/block_size + (n % block_size != 0);
    for(int i=1;i<num_blocks;i++){
      if(D[0] > D[i]){
        D[0] = D[i];
      }
    }
  }
}
}
// ----------------------------------------------------------------------------
// Host function to compute minimum distance between points
// Input:
//	X: X[i] = x-coordinate of the ith point
//	Y: Y[i] = y-coordinate of the ith point
//	n: number of points
// Output:
//	D: minimum distance
//
float minimum_distance_host(float * X, float * Y, int n) {
    float dx, dy, Dij, min_distance, min_distance_i;
    int i, j;
    dx = X[1]-X[0];
    dy = Y[1]-Y[0];
    min_distance = sqrtf(dx*dx+dy*dy);
    for (i = 0; i < n-1; i++) {
	for (j = i+1; j < i+2; j++) {
	    dx = X[j]-X[i];
	    dy = Y[j]-Y[i];
	    min_distance_i = sqrtf(dx*dx+dy*dy);
	}
	for (j = i+1; j < n; j++) {
	    dx = X[j]-X[i];
	    dy = Y[j]-Y[i];
	    Dij = sqrtf(dx*dx+dy*dy);
	    if (min_distance_i > Dij) min_distance_i = Dij;
	}
	if (min_distance > min_distance_i) min_distance = min_distance_i;
    }
    return min_distance;
}
// ----------------------------------------------------------------------------
// Print device properties
void print_device_properties() {
    int i, deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    printf("------------------------------------------------------------\n");
    printf("Number of GPU devices found = %d\n", deviceCount);
    for ( i = 0; i < deviceCount; ++i ) {
	cudaGetDeviceProperties(&deviceProp, i);
	printf("[Device: %1d] Compute Capability %d.%d.\n", i, deviceProp.major, deviceProp.minor);
	printf(" ... multiprocessor count  = %d\n", deviceProp.multiProcessorCount);
	printf(" ... max threads per multiprocessor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf(" ... max threads per block = %d\n", deviceProp.maxThreadsPerBlock);
	printf(" ... max block dimension   = %d, %d, %d (along x, y, z)\n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf(" ... max grid size         = %d, %d, %d (along x, y, z)\n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf(" ... warp size             = %d\n", deviceProp.warpSize);
	printf(" ... clock rate            = %d MHz\n", deviceProp.clockRate/1000);
    }
    printf("------------------------------------------------------------\n");
}
// ----------------------------------------------------------------------------
// Main program - initializes points and computes minimum distance
// between the points
//
int main(int argc, char* argv[]) {

    // Host Data
    float * hVx;		// host x-coordinate array
    float * hVy;		// host y-coordinate array
    float * hmin_dist;		// minimum value on host

    // Device Data
    float * dVx;		// device x-coordinate array
    float * dVy;		// device x-coordinate array
    float * dmin_dist;		// minimum value on device

    // Device parameters
    int MAX_BLOCK_SIZE;		// Maximum number of threads allowed on the device
    int blocks;			// Number of blocks in grid
    int threads_per_block;	// Number of threads per block

    // Timing variables
    cudaEvent_t start, stop;		// GPU timing variables
    struct timespec cpu_start, cpu_stop; // CPU timing variables
    float time_array[10];

    // Other variables
    int i, size, num_points;
    float min_distance, sqrtn;
    int seed = 0;

    // Print device properties
    print_device_properties();

    // Get device information and set device to use
    int deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
	cudaSetDevice(0);
	cudaGetDeviceProperties(&deviceProp, 0);
	MAX_BLOCK_SIZE = deviceProp.maxThreadsPerBlock;
    } else {
	printf("Warning: No GPU device found ... results may be incorrect\n");
    }

    // Timing initializations
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Check input
    if (argc != 2) {
	printf("Use: %s <number of points>\n", argv[0]);
	exit(0);
    }
    if ((num_points = atoi(argv[argc-1])) < 2) {
	printf("Minimum number of points allowed: 2\n");
	exit(0);
    }
    if ((num_points = atoi(argv[argc-1])) > MAX_POINTS) {
	printf("Maximum number of points allowed: %d\n", MAX_POINTS);
	exit(0);
    }

    // Allocate host coordinate arrays
    size = num_points * sizeof(float);
    hVx = (float *) malloc(size);
    hVy = (float *) malloc(size);

    hmin_dist = (float *) malloc(size);

    // Initialize points
    srand48(seed);
    sqrtn = (float) sqrt(num_points);
    for (i = 0; i < num_points; i++) {
	hVx[i] = sqrtn * (float)drand48();
	hVy[i] = sqrtn * (float)drand48();
    }

    // Allocate device coordinate arrays
    cudaMalloc(&dVx, size);
    cudaMalloc(&dVy, size);
    cudaMalloc(&dmin_dist, size);

    // Copy coordinate arrays from host memory to device memory
    cudaEventRecord( start, 0 );

    cudaMemcpy(dVx, hVx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dVy, hVy, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(time_array[0]), start, stop);

    // Invoke kernel
    cudaEventRecord( start, 0 );

    // ------------------------------------------------------------
    int num_blocks = num_points/(block_size) + ((num_points % (block_size)) !=0);
    minimum_distance<<<num_blocks,block_size>>>(dVx,dVy,dmin_dist,num_points);
    // ------------------------------------------------------------

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(time_array[1]), start, stop);

    // Copy result from device memory to host memory
    cudaEventRecord( start, 0 );

    cudaMemcpy(hmin_dist, dmin_dist, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(time_array[2]), start, stop);

    // Compute minimum distance on host to check device computation
    clock_gettime(CLOCK_REALTIME, &cpu_start);

    min_distance = minimum_distance_host(hVx, hVy, num_points);

    clock_gettime(CLOCK_REALTIME, &cpu_stop);
    time_array[3] = 1000*((cpu_stop.tv_sec-cpu_start.tv_sec)
	    +0.000000001*(cpu_stop.tv_nsec-cpu_start.tv_nsec));

    // Print results
    printf("Number of Points    = %d\n", num_points);
    printf("GPU Host-to-device  = %f ms \n", time_array[0]);
    printf("GPU Device-to-host  = %f ms \n", time_array[2]);
    printf("GPU execution time  = %f ms \n", time_array[1]);
    printf("CPU execution time  = %f ms\n", time_array[3]);
    printf("Min. distance (GPU) = %e\n", hmin_dist);
    printf("Min. distance (CPU) = %e\n", min_distance);
    printf("Relative error      = %e\n", fabs(min_distance-hmin_dist[0])/min_distance);


    // Free device memory
    cudaFree(dVx);
    cudaFree(dVy);
    cudaFree(dmin_dist);

    // Free host memory
    free(hVx);
    free(hVy);
}
