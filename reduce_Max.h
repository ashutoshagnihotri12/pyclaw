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

// Single block reduce Max. The assumption is that a single block reduces over the data.
// Technically gridDim = (1,0,0), blockDim = (blockSize,0,0)
// if there are more elements than threads, reduce elements into shared until all elements are checked
// else make sure the available data is put in shared and extra shared space is made 0
// Then reduce over all elements in shared memory
// return the first element in shared to the global input's first place
template <unsigned int blockSize>
__global__ void reduceMax_simplified(real *globalWaves, int size)
{
	// size holds the exact number of wave speeds there are to be max reduced
	extern __shared__ real sharedWaveSpeeds[];	// This will be as big as the block size
	int tid = threadIdx.x;

	// test alternative: have a first phase to collect at least one layer in shared memory
	// then go block size by block size take the max of the elements
	if ( size > blockSize )
	{
		sharedWaveSpeeds[tid] = globalWaves[tid];

		int i = tid;

		while (i + blockSize < size)	// !!!! WE MUST MAKE SURE THAT WHEN ALLOCATING MEMORY FOR THE GLOBAL WAVES SPEEDS
										// !!!! WE ALLCOATE MORE MEMORY THAN NECESSARY (BY A MARGINAL AMMOUNT) AND WE
										// !!!! INITIALIZE THE MEMORY TO BE FILLED WITH ZEROS, AS i + blockSize COULD
										// !!!! READ OUTSIDE THE BOUNDS OF GLOBAL WAVE SPEEDS AND PRODUCE ERRONEOUS RESULTS
		{
			// all of these accesses will be cached,
			// in fact the whole reduced waves would fit in shared memory/L1 for medium sized problems
			// L2 has 768KB of memory, for a 1024*1024 grid we have about 16K elements to reduce
			// We do not need to take fabs as it is done in the local reduction stage
			sharedWaveSpeeds[tid] = fmax(sharedWaveSpeeds[tid], globalWaves[i+blockSize]);
			i += blockSize;
		}
	}
	else
	{
		if (tid < size)
			sharedWaveSpeeds[tid] = globalWaves[tid];
		else
			sharedWaveSpeeds[tid] = (real)0.0f;
	}
	__syncthreads();

	// ASSUMING BLOCK SIZES CAN BE ONLY 1024, 512, 256, 128, 64, OR 32, ... 1
	// At this stage we are sure that whatever is in the shared memory was already taken absolutely
	// that is why we do not need to invoke the fabs function again.
	if (blockSize >= 1024)
	{ 
		if (tid < 512)
			sharedWaveSpeeds[tid] = fmax(sharedWaveSpeeds[tid], sharedWaveSpeeds[tid + 512]);
		__syncthreads();
	}
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
	if (tid == 0)
		globalWaves[blockIdx.x] = sharedWaveSpeeds[0];
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
	int i = blockIdx.x*blockSize + tid;

	unsigned int gridSize = blockSize*gridDim.x;

	// test alternative: have a first phase to collect at least one layer in shared memory
	// then go block size by block size take the max of the elements
	if ( size > blockSize )
	{
		if ( i < size )
			sharedWaveSpeeds[tid] = globalWaves[i];
		else
			sharedWaveSpeeds[tid] = 0.0f;

		while (i+blockSize < size)	// !!!! WE MUST MAKE SURE THAT WHEN ALLOCATING MEMORY FOR THE GLOBAL WAVES SPEEDS
									// !!!! WE ALLCOATE MORE MEMORY THAN NECESSARY (BY A MARGINAL AMMOUNT) AND WE
									// !!!! INITIALIZE THE MEMORY TO BE FILLED WITH ZEROS, AS i + blockSize COULD
									// !!!! READ OUTSIDE THE BOUNDS OF GLOBAL WAVE SPEEDS AND PRODUCE ERRONEOUS RESULTS
		{
			// all of these accesses will be cached,
			// in fact the whole reduced waves would fit in shared memory/L1 for medium sized problems
			// L2 has 768KB of memory, for a 1024*1024 grid we have about 16K elements to reduce
			// We do not need to take fabs as it is done in the local reduction stage
			sharedWaveSpeeds[tid] = fmax(sharedWaveSpeeds[tid], globalWaves[i+blockSize]);
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
		if (tid == 0)
			globalWaves[blockIdx.x] = sharedWaveSpeeds[0];
	}
	else
	{
		if (tid < size)
			sharedWaveSpeeds[tid] = globalWaves[i];
		else
			sharedWaveSpeeds[tid] = (real)0.0f;

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
		if (tid == 0)
			globalWaves[blockIdx.x] = sharedWaveSpeeds[0];
	}
}

// double reduction, stupid...
template <unsigned int blockSize>
__global__ void reduceMax(real *globalWavesX, real*globalWavesY, int sizeX, int sizeY)
{
	// size holds the exact number of wave speeds there are to be max reduced
	extern __shared__ real sharedWaveSpeeds[];	// This will be as big as the block size
	int tid = threadIdx.x;

	// each block handles the data it corresponds to and the data of same size right next to it
	int i = blockIdx.x*(blockSize*2) + tid;

	unsigned int gridSize = blockSize*2*gridDim.x;

	// test alternative: have a first phase to collect at least one layer in shared memory
	// then go block size by block size take the max of the elements
	if ( sizeX > blockSize )
	{
		while (i+blockSize < sizeX)	// !!!! WE MUST MAKE SURE THAT WHEN ALLOCATING MEMORY FOR THE GLOBAL WAVES SPEEDS
									// !!!! WE ALLCOATE MORE MEMORY THAN NECESSARY (BY A MARGINAL AMMOUNT) AND WE
									// !!!! INITIALIZE THE MEMORY TO BE FILLED WITH ZEROS, AS i + blockSize COULD
									// !!!! READ OUTSIDE THE BOUNDS OF GLOBAL WAVE SPEEDS AND PRODUCE ERRONEOUS RESULTS
		{
			// all of these accesses will be cached,
			// in fact the whole reduced waves would fit in shared memory/L1 for medium sized problems
			// L2 has 768KB of memory, for a 1024*1024 grid we have about 16K elements to reduce
			// We do not need to take fabs as it is done in the local reduction stage
			sharedWaveSpeeds[tid] = fmax(globalWavesX[i], globalWavesX[i+blockSize]);	//fmax(fabs(globalWaves[i]), fabs(globalWaves[i+blockSize]));
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
			globalWavesX[0] = sharedWaveSpeeds[0];
	}
	else
	{
		if (tid < sizeX)
			sharedWaveSpeeds[tid] = globalWavesX[i];
		else
			sharedWaveSpeeds[tid] = (real)0.0f;

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
			globalWavesX[0] = sharedWaveSpeeds[0];
	}




	i = blockIdx.x*(blockSize*2) + tid;
	if ( sizeY > blockSize )
	{
		while (i+blockSize < sizeY)	// !!!! WE MUST MAKE SURE THAT WHEN ALLOCATING MEMORY FOR THE GLOBAL WAVES SPEEDS
									// !!!! WE ALLCOATE MORE MEMORY THAN NECESSARY (BY A MARGINAL AMMOUNT) AND WE
									// !!!! INITIALIZE THE MEMORY TO BE FILLED WITH ZEROS, AS i + blockSize COULD
									// !!!! READ OUTSIDE THE BOUNDS OF GLOBAL WAVE SPEEDS AND PRODUCE ERRONEOUS RESULTS
		{
			// all of these accesses will be cached,
			// in fact the whole reduced waves would fit in shared memory/L1 for medium sized problems
			// L2 has 768KB of memory, for a 1024*1024 grid we have about 16K elements to reduce
			// We do not need to take fabs as it is done in the local reduction stage
			sharedWaveSpeeds[tid] = fmax(globalWavesY[i], globalWavesY[i+blockSize]);	//fmax(fabs(globalWaves[i]), fabs(globalWaves[i+blockSize]));
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
			globalWavesY[0] = sharedWaveSpeeds[0];
	}
	else
	{
		if (tid < sizeY)
			sharedWaveSpeeds[tid] = globalWavesY[i];
		else
			sharedWaveSpeeds[tid] = (real)0.0f;

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
			globalWavesY[0] = sharedWaveSpeeds[0];
	}
}

#endif