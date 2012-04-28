#ifndef __REDUCE_MAX_H__
#define __REDUCE_MAX_H__


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////   Reduction   //////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned int blockSize>
__device__ void warpReduce(volatile real *sdata, unsigned int tid) {
	if (blockSize >=  64) sdata[tid] = fmax(sdata[tid], sdata[tid + 32]);
	if (blockSize >=  32) sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
	if (blockSize >=  16) sdata[tid] = fmax(sdata[tid], sdata[tid +  8]);
	if (blockSize >=   8) sdata[tid] = fmax(sdata[tid], sdata[tid +  4]);
	if (blockSize >=   4) sdata[tid] = fmax(sdata[tid], sdata[tid +  2]);
	if (blockSize >=   2) sdata[tid] = fmax(sdata[tid], sdata[tid +  1]);
}

// This reduction does not return the absolute maximum, but just the maximum. For our problem,
// we know the absolute values were taken in the local reduction stage, so we only need to look for the max.
template <unsigned int blockSize>
__global__ void reduceMax(real *globalWaves, int size)
{
	// size holds the exact number of wave speeds there are to be max reduced
	extern __shared__ real sharedWaveSpeeds[];	// This will be as big as the block size
	int tid = threadIdx.x;

	// each block handles the data it corresponds to and the data of same size right next to it
	int i = blockIdx.x*(blockSize*2) + tid;

	unsigned int gridSize = blockSize*2*gridDim.x;

	// test alternative: have a first phase to collect at least one layer in shared memory
	// then go block size by block size take the max of the elements

	while (i < size)	// !!!! WE MUST MAKE SURE THAT WHEN ALLOCATING MEMORY FOR THE GLOBAL WAVES SPEEDS
						// !!!! WE ALLCOATE MORE MEMORY THAN NECESSARY (BY A MARGINAL AMMOUNT) AND WE
						// !!!! INITIALIZE THE MEMORY TO BE FILLED WITH ZEROS, AS i + blockSize COULD
						// !!!! READ OUTSIDE THE BOUNDS OF GLOBAL WAVE SPEEDS AND PRODUCE ERRONEOUS RESULTS
	{
		// all of these accesses will be cached,
		// in fact the whole reduced waves would fit in shared memory/L1 for medium sized problems
		// L2 has 768KB of memory, for a 1024*1024 grid we have about 16K elements to reduce
		// We do not need to take fabs as it is done in the local reduction stage
		sharedWaveSpeeds[tid] = fmax(globalWaves[i], globalWaves[i+blockSize]);	//fmax(fabs(globalWaves[i]), fabs(globalWaves[i+blockSize]));
		i += gridSize;
	}
	__syncthreads();

	// ASSUMING BLOCK SIZES CAN BE ONLY 512, 256, 128, 64, OR 32, ... 1
	// At this stage we are sure that whatever is in the shared memory was already taken absolutely
	// that is why we do not need to invoke the fabs function again.
	if (blockSize >= 512)
	{ 
		if (tid < 256)
			sharedWaveSpeeds[tid] = fmax(sharedWaveSpeeds[tid], sharedWaveSpeeds[tid + 256]);
		__syncthreads();
	}
	if (blockSize >= 256)
	{ 
		if (tid < 128)
			sharedWaveSpeeds[tid] = fmax(sharedWaveSpeeds[tid], sharedWaveSpeeds[tid + 128]);
		__syncthreads();
	}
	if (blockSize >= 128)
	{ 
		if (tid < 64)
			sharedWaveSpeeds[tid] = fmax(sharedWaveSpeeds[tid], sharedWaveSpeeds[tid + 64]);
		__syncthreads();
	}
	if (tid < 32)
		warpReduce<blockSize>(sharedWaveSpeeds, tid);
	if (tid == 0 && blockIdx.x == 0)
		globalWaves[0] = sharedWaveSpeeds[0];
}

#endif