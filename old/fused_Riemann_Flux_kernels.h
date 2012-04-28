#ifndef __FUSED_RIEMANN_FLUX_KERNELS_H__
#define __FUSED_RIEMANN_FLUX_KERNELS_H__

//#include "common.h"
//#include <cmath>

//#define max(a,b) (((a) > (b)) ? (a):(b))
//#define min(a,b) (((a) < (b)) ? (a):(b))


// This will handle data in shared memory,
// the concern at this stage is not coalescing and alignment (not as it would be in global)
// but bank conflicts, different schemes are likely to yield different performances
// Note however that the Riemann solver depends on the distribution of this data,
// and assumes to have wave1[state1, state2, state3] wave2[state1, state2, state3]
// If this is to remain we must keep the fastest changing components and fiddle
// only with the slowest. In the case of the waves, the wave number and states must
// remain as they are, and in the case of the wave speeds the wave number must remain.
// Alternatively we must provide a function to the user to set his/her own waves
// such that it would be compatible to the way the framework lays down the memory
// A first implementation will be a straight
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////      Waves    //////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ int getIndex_sharedWaves(int row, int col, int waveNum, int state, int numStates, int numWaves, int blockWidth)
{
	return (row*numStates*numWaves*blockWidth + col*numStates*numWaves + waveNum*numStates + state);
}
inline __device__ real &getSharedWave(real* sharedWaves, int row, int col, int waveNum, int state, int numStates, int numWaves, int blockWidth)
{
	return sharedWaves[getIndex_sharedWaves(row, col, waveNum, state, numStates, numWaves, blockWidth)];
}
inline __device__ void setSharedWave(real* sharedWaves, int row, int col, int waveNum, int state, int numStates, int numWaves, int blockWidth, real newVal)
{
	sharedWaves[getIndex_sharedWaves(row, col, waveNum, state, numStates, numWaves, blockWidth)] = newVal;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////  Wave Speeds  //////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ int getIndex_waveSpeed(int row, int col, int waveNum, int numWaves, int blockWidth)
{
	return (row*numWaves*blockWidth + col*numWaves + waveNum);
}
inline __device__ real &getWaveSpeed(real* waveSpeeds, int row, int col, int waveNum, int numWaves, int blockWidth)
{
	return waveSpeeds[getIndex_waveSpeed(row, col, waveNum, numWaves, blockWidth)];
}
inline __device__ void setWaveSpeed(real* waveSpeeds, int row, int col, int waveNum, int numWaves, int blockWidth, real newSpeed)
{
	waveSpeeds[getIndex_waveSpeed(row, col, waveNum, numWaves, blockWidth)] = newSpeed;
}

template<unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
	if (blockSize >=  64) sdata[tid] = fmax(fabs(sdata[tid]), fabs(sdata[tid + 32]));
	if (blockSize >=  32) sdata[tid] = fmax(fabs(sdata[tid]), fabs(sdata[tid + 16]));
	if (blockSize >=  16) sdata[tid] = fmax(fabs(sdata[tid]), fabs(sdata[tid +  8]));
	if (blockSize >=   8) sdata[tid] = fmax(fabs(sdata[tid]), fabs(sdata[tid +  4]));
	if (blockSize >=   4) sdata[tid] = fmax(fabs(sdata[tid]), fabs(sdata[tid +  2]));
	if (blockSize >=   2) sdata[tid] = fmax(fabs(sdata[tid]), fabs(sdata[tid +  1]));
}

template<int numStates, class Limiter>
__device__ real limiting (Limiter phi, real* main_wave, real* aux_wave)
{
	real main_wave_norm_square = main_wave[0]*main_wave[0];
	real aux_wave_dot_main_wave = aux_wave[0]*main_wave[0];
	#pragma unroll
	for (int i = 1; i < numStates; i++)
	{
		main_wave_norm_square += main_wave[i]*main_wave[i];
		aux_wave_dot_main_wave += aux_wave[i]*main_wave[i];
	}
		////real theta = aux_wave_dot_main_wave/main_wave_norm_square;
		////real minimum = min(2.0, 2.0*theta);
		////return 0.5;
		////return max(0, min((1+theta)/2.0, minimum));
		//
		//real theta = aux_wave_dot_main_wave/main_wave_norm_square;
		//real absTheta = abs(theta);
		//return (theta + absTheta) / (1 + absTheta);
	if (main_wave_norm_square < (real) 0.00000001)				// flying constant, make sure to see what to do with it
		return (real)0;
	return phi(aux_wave_dot_main_wave/main_wave_norm_square);
}

extern __shared__ real shared_elem[];
template <const int numStates, int numWaves, int blockSize, class Riemann ,class Limiter>
__global__ void fused_Riemann_limiter_horizontal_update_kernel(pdeParam param, real dt, Riemann Riemann_solver_h, Limiter phi)
{
	int col = threadIdx.x + blockIdx.x*blockDim.x - 3*blockIdx.x;//- 3*blockDim.x;	// 3 if we want to use limiters
	int row = threadIdx.y + blockIdx.y*blockDim.y;

	real apdq[numStates];
	real amdq[numStates];

	real* waves			= &shared_elem[0];
	real* waveSpeeds	= &shared_elem[(blockDim.x*blockDim.y)*numWaves*numStates];	

	// Riemann solver
	if (row < param.cellsY && col < param.cellsX-1)	// if there are 512 cells in X 511 threads are in use
	{
		Riemann_solver_h(&param.getElement_qNew(row, col, 0),		// input comes from global memory
							&param.getElement_qNew(row, col+1, 0),	//
							numStates,								//
							&param.getElement_coeff(row, col, 0),	//
							&param.getElement_coeff(row, col+1, 0),	//
			/*shared*/		&getSharedWave(waves, threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x),	// &waves[threadIdx.x*numWaves*numStates + threadIdx.y*numWaves*numStates*blockDim.x]
			/*shared*/		&getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, 0, numWaves, blockDim.x));			// &waveSpeeds[threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x]
	}
	__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed

	if (row < param.cellsY && col < param.cellsX-1 && (threadIdx.x > 0 && threadIdx.x < blockDim.x-1))
	{
		// Update Fluctuation
		// grab the necessary waves and add, the most general case would be to grab the wave and speed check the speed sign and choose where to add
		// this might give rise to race conditions if addition is not handled properly. In acoustics' case we know the sign of the waves beforehand
		// but this is not the case in general, and therefore a sign check must be made and perhaps atomics used to add to qNew, or
		// (best for performance) uglyly held in many variables residing in thread registers which accumulate the necessary sums to be applied
		// to qNew. Maybe with the use of the occupancy calculator we can determine the maximum number of such variables possible.
		// I sleep now. I continue now.
		#pragma unroll
		for (int k = 0; k < numStates; k++)		// I do not know if there are better ways to do this... (this = initialize the fluctuations to 0)
		{
			amdq[k] = 0;
			apdq[k] = 0;
		}
		#pragma unroll
		for (int w = 0; w < numWaves; w++)
		{
			real waveSpeed =  getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, w, numWaves, blockDim.x);
			if (waveSpeed < 0)
			{
				real limiting_factor = limiting<numStates>(phi,
												&getSharedWave(waves, threadIdx.y, threadIdx.x, w, 0, numStates, numWaves, blockDim.x),
												&getSharedWave(waves, threadIdx.y, threadIdx.x+1, w, 0, numStates, numWaves, blockDim.x));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x);
					amdq[k] +=	waveSpeed * wave_state
							+
								-0.5*waveSpeed*(1+(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
				}
			}
			else
			{
				real limiting_factor = limiting<numStates>(phi,
												&getSharedWave(waves, threadIdx.y, threadIdx.x, w, 0, numStates, numWaves, blockDim.x),
												&getSharedWave(waves, threadIdx.y, threadIdx.x-1, w, 0, numStates, numWaves, blockDim.x));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x);
					apdq[k] +=	waveSpeed * wave_state
							-
								0.5*waveSpeed*(1-(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
				}
			}
		}
	}
	__syncthreads();	// make sure all threads finish computing their fluctuations
						// at this stage the threads consumed the generated waves,
						// and have no further use for them.
						// an alternative here is to have this sync at the end of every wave (pth) processing
						// after which the pth component of fluctuation is stored in shared
	
	if (row < param.cellsY && col < param.cellsX-1 && (threadIdx.x > 0 && threadIdx.x < blockDim.x-1))
	{
		// store the right going fluctuation onto shared to let other\neighboring thread grab thread them.

		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			setSharedWave(waves, threadIdx.y, threadIdx.x, 0, k, numStates, numWaves, blockDim.x, apdq[k]);
		}

		// Once storing the right fluctuation is done we need to make sure all threads have stored theirs
		// so that they are sure that the place they are going to to grab the right fluctuation of their left neighbor
		// indeed holds the right fluctuation of their left neighbor and not the wave generated at the Riemann stage
		// This is only really necessary at the stage of threads at the extremeties of warps, because threads in warps
		// actually do the job at the same time, and a discrepancy may only occur if the left threads of a warp is
		// trying to pick the right fluctuation of the thread at the rightmost side of the left neighboring warp,
		// and that only if the right warp went before the left warp.
		// To be more clear let's use some ascii art:
		//      warp i        warp i+1				all in same block
		//  [I I ... I I_31]  [I_0 I ... I I] 
		//
		//  say warp i+1 went first and without waiting for warp i (at a __syncthreads() ) tried to proceed
		//  with the stage of grabbing left neighbor's right fluctuation, as warp i has not executed yet,
		//  the memory location where the [warp i I_31]'s right fluctuation should be does not have the 
		//  correct fluctuation yet, but instead has the generated waves at the Riemann stage.
		//  That's why a block has to synchronize its own threads to make the computation correct.
		// __syncthreads() being lightweight should not cost the program to much.
		// Disabeling __syncthreads() is possible just to see how much it affects.
	}
	__syncthreads(); // read the above comment

	if (row < param.cellsY && col < param.cellsX-1)
	{
		// Final Update, in One Shot
		// !!! The condition here decides which thread is to update the cells
		// !!! The conditions are different for simple Riemann solver,
		// !!! than for the Riemann solver coupled with a limiter
		if ( threadIdx.x > 1 && threadIdx.x < blockDim.x-1 )
		{
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				param.setElement_q(row, col, k, param.getElement_qNew(row, col, k) - dt/param.dx * (amdq[k] + getSharedWave(waves, threadIdx.y, threadIdx.x-1, 0, k, numStates, numWaves, blockDim.x)));
			}
		}
	}

	// Loacl Reduce over Wave Speeds
	// Stage 1
	// Bringing down the number of elements to compare to block size
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	#pragma unroll
	for (int i = 1; i < numWaves; i++)
	{
		// blockDim.x * blockDim.y is usally a multiple of 32, so there should be no bank conflicts.
		waveSpeeds[tid] = fmax(fabs(waveSpeeds[tid]), fabs(waveSpeeds[tid + i*blockSize]));
	}
	__syncthreads();

	// Stage 2
	// Reducing over block size elements
	// use knowledge about your own block size:
	// I am using blocks of size 192 for the horizontal direction
	if (blockSize >= 192)
	{
		if (tid < 96)
			waveSpeeds[tid] = fmax(fabs(waveSpeeds[tid]), fabs(waveSpeeds[tid + 96]));
		__syncthreads();
	}
	if (blockSize >= 96)
	{
		if (tid < 64)
			waveSpeeds[tid] = fmax(fabs(waveSpeeds[tid]), fabs(waveSpeeds[tid + 64]));
		__syncthreads();
	}
	if (tid < 32)
		warpReduce(waveSpeeds, tid);
	if (tid == 0)
		setWaveSpeed(waveSpeeds, /**/ waveSpeeds[0]);
}

extern __shared__ real shared_elem[];
template <const int numStates, int numWaves, int blockSize, class Riemann ,class Limiter>
__global__ void fused_Riemann_limiter_vertical_update_kernel(pdeParam param, real dt, Riemann Riemann_solver_v, Limiter phi)
{
	int col = threadIdx.x + blockIdx.x*blockDim.x;
	int row = threadIdx.y + blockIdx.y*blockDim.y - 3*blockIdx.y;//- 3*blockDim.y;	// 3 if we want to use limiters

	real apdq[numStates];
	real amdq[numStates];

	real* waves			= &shared_elem[0];
	real* waveSpeeds	= &shared_elem[(blockDim.x*blockDim.y)*numWaves*numStates];	

	// Riemann solver
	if (row < param.cellsY-1 && col < param.cellsX)	// if there are 512 cells in Y 511 threads are in use
	{
		Riemann_solver_v(&param.getElement_qNew(row, col, 0),		// input comes from global memory
							&param.getElement_qNew(row+1, col, 0),	//
							numStates,								//
							&param.getElement_coeff(row, col, 0),	//
							&param.getElement_coeff(row+1, col, 0),	//
			/*shared*/		&getSharedWave(waves, threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x),	// &waves[threadIdx.x*numWaves*numStates + threadIdx.y*numWaves*numStates*blockDim.x]
			/*shared*/		&getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, 0, numWaves, blockDim.x));			// &waveSpeeds[threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x]
	}

	__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed

	if(row < param.cellsY-1 && col < param.cellsX && (threadIdx.y > 0 && threadIdx.y < blockDim.y-1))
	{
		// Update Fluctuation
		// grab the necessary waves and add, the most general case would be to grab the wave and speed check the speed sign and choose where to add
		// this might give rise to race conditions if addition is not handled properly. In acoustics' case we know the sign of the waves beforehand
		// but this is not the case in general, and therefore a sign check must be made and perhaps atomics used to add to qNew, or
		// (best for performance) uglyly held in many variables residing in thread registers which accumulate the necessary sums to be applied
		// to qNew. Maybe with the use of the occupancy calculator we can determine the maximum number of such variables possible.
		// I sleep now. I continue now.
		#pragma unroll
		for (int k = 0; k < numStates; k++)		// I do not know if there are better ways to do this... (this = initialize the fluctuations to 0)
		{
			amdq[k] = 0;
			apdq[k] = 0;
		}
		#pragma unroll
		for (int w = 0; w < numWaves; w++)
		{
			real waveSpeed =  getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, w, numWaves, blockDim.x);
			if (waveSpeed < 0)
			{
				real limiting_factor = limiting<numStates>(phi,
												&getSharedWave(waves, threadIdx.y, threadIdx.x, w, 0, numStates, numWaves, blockDim.x),
												&getSharedWave(waves, threadIdx.y+1, threadIdx.x, w, 0, numStates, numWaves, blockDim.x));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x);
					amdq[k] +=	waveSpeed * wave_state
							+
								-0.5*waveSpeed*(1+(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
				}
			}
			else
			{
				real limiting_factor = limiting<numStates>(phi,
												&getSharedWave(waves, threadIdx.y, threadIdx.x, w, 0, numStates, numWaves, blockDim.x),
												&getSharedWave(waves, threadIdx.y-1, threadIdx.x, w, 0, numStates, numWaves, blockDim.x));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x);
					apdq[k] +=	waveSpeed * wave_state
							-
								0.5*waveSpeed*(1-(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
				}
			}
		}
	}
	__syncthreads();
	
	if(row < param.cellsY-1 && col < param.cellsX && (threadIdx.y > 0 && threadIdx.y < blockDim.y-1))
	{
		// store down going fluctuation is shared, to pass them to the bottom neighbor threads
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			setSharedWave(waves, threadIdx.y, threadIdx.x, 0, k, numStates, numWaves, blockDim.x, apdq[k]);
		}
	}
	__syncthreads();

	if(row < param.cellsY-1 && col < param.cellsX)
	{
		// Final Update, in One Shot
		// !!! The condition here decides which thread is to update the cells
		// !!! The conditions are different for simple Riemann solver,
		// !!! than for the Riemann solver coupled with a limiter
		if ( threadIdx.y > 1 && threadIdx.y < blockDim.y-1 )
			#pragma unroll
			for (int k = 0; k < numStates; k++)
			{
				param.setElement_q(row, col, k, param.getElement_q(row, col, k) - dt/param.dy * (amdq[k]+getSharedWave(waves, threadIdx.y-1, threadIdx.x, 0, k, numStates, numWaves, blockDim.x)));
			}
	}
}

extern __shared__ real shared_elem[];
template <const int numStates, int numWaves, class Riemann ,class Limiter>
__global__ void fused_Riemann_limiter_horizontal_update_kernel2(pdeParam param, real dt, Riemann Riemann_solver_h, Limiter phi)
{
	// Blocks are responsible to handle the update of certain cells
	// Some thread will just do the Riemann solution and finish their jobs
	// providing for other threads the waves and speeds they need to update
	// the states.
	int col = threadIdx.x + blockIdx.x*blockDim.x - 3*blockIdx.x;// 1 if we want simple Riemann, 3 if we want to use limiters
	int row = threadIdx.y + blockIdx.y*blockDim.y;

	real* waves			= &shared_elem[0];
	real* waveSpeeds	= &shared_elem[(blockDim.x*blockDim.y)*numWaves*numStates];	// I do not know if we need to store the alpha, or how the computation for the limiters will be done, for now I keep things simple for the Riemann solvers, the number 256 is the number of threads per block

	// Riemann solver
	if (row < param.cellsY && col < param.cellsX-1)	// if there are 512 cells in X 511 threads are in use
	{
		Riemann_solver_h(&param.getElement_qNew(row, col, 0),		// input comes from global memory
							&param.getElement_qNew(row, col+1, 0),	//
							numStates,								//
							&param.getElement_coeff(row, col, 0),	//
							&param.getElement_coeff(row, col+1, 0),	//
			/*shared*/		&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
			/*shared*/		&waveSpeeds[getIndex_waveSpeed(threadIdx.y, threadIdx.x, 0, numWaves, blockDim.x)]);
	}

	__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed

	if ( (threadIdx.x > 1 && threadIdx.x < blockDim.x-1) && row < param.cellsY && col < param.cellsX-1 )
	{
		// Update Incoming Fluctuations
		// Each active thread will look for the incoming waves to its cell, and for the update term

		// I propose we do a local reduction on the wave speeds and return just the absolute maximum
		// saving both time and space. This can be done once all updates are done.
		//param.setElement_waveSpeed(row, col, w, waveSpeed);

		real amdq_apdq[numStates];
		//#pragma unroll numStates
		for (int k = 0; k < numStates; k++)		// I do not know if there are better ways to do this... (this = initialize the fluctuations to 0)
		{
			amdq_apdq[k] = 0;
		}

		// We seperated left and right looking interfaces to prehaps avaoid bank conflicts when reading the wave speeds

		//#pragma unroll numWaves
		for (int w = 0; w < numWaves; w++)	// in his book, Randy Leveque uses p to denote the pth wave and not w
		{
			/////////
			///////// THREAD IS LOOKING TO THE WAVES FROM THE INTERFACES TO ITS LEFT
			/////////
			real waveComingFromLeft_waveSpeed = getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x-1, w, numWaves, blockDim.x);

			if (waveComingFromLeft_waveSpeed > 0)		// warp divergence will happen at rarefation points and shock formation points
			{
				real limiting_factor = limiting(phi,
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x-1, 0, 0, numStates, numWaves, blockDim.x)],	// main wave
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x-2, 0, 0, numStates, numWaves, blockDim.x)],	// aux wave
												numStates);
				//#pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] +=		waveComingFromLeft_waveSpeed * waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x-1, w, k, numStates, numWaves, blockDim.x)]
									-
										0.5*waveComingFromLeft_waveSpeed*(1-(dt/param.dx)*waveComingFromLeft_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x-1, w, k, numStates, numWaves, blockDim.x)];
				}
			}
			else	// wave is negative there is no actual wave coming to the cell, but there is some correction
			{
				real limiting_factor = limiting(phi,
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x-1, 0, 0, numStates, numWaves, blockDim.x)],	// main wave
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],		// aux wave
												numStates);
				//#pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] +=	- -0.5*waveComingFromLeft_waveSpeed*(1+(dt/param.dx)*waveComingFromLeft_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x-1, w, k, numStates, numWaves, blockDim.x)];
				}
			}
		}
		for (int w = 0; w < numWaves; w++)	// in his book, Randy Leveque uses p to denote the pth wave and not w
		{
			/////////
			///////// THREAD IS LOOKING TO THE WAVES FROM THE INTERFACES TO ITS RIGHT
			/////////
			real waveComingFromRight_waveSpeed = getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, w, numWaves, blockDim.x);

			if (waveComingFromRight_waveSpeed < 0)		// warp divergence will happen at rarefation points and shock formation points
			{
				real limiting_factor = limiting(phi,
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],		// main wave
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x+1, 0, 0, numStates, numWaves, blockDim.x)],	// aux wave
												numStates);
				//#pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] +=		waveComingFromRight_waveSpeed * waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x)]
									+
										-0.5*waveComingFromRight_waveSpeed*(1+(dt/param.dx)*waveComingFromRight_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x)];
				}
			}
			else	// wave is positive there is no actual wave coming to the cell, but there is some correction
			{
				real limiting_factor = limiting(phi,
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
												&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x-1, 0, 0, numStates, numWaves, blockDim.x)],
												numStates);
				//pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] +=	0.5*waveComingFromRight_waveSpeed*(1-(dt/param.dx)*waveComingFromRight_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x)];
				}
			}
		}

		// Final Update
		// !!! The condition here decides which thread is to update the cells
		// !!! The conditions are different for simple Riemann solver,
		// !!! than for the Riemann solver coupled with a limiter
		//#pragma unroll numStates
		for (int k = 0; k < numStates; k++)
		{
			param.setElement_q(row, col, k, param.getElement_qNew(row, col, k) - dt/param.dx * amdq_apdq[k]);
		}
	}
}

extern __shared__ real shared_elem[];
template <const int numStates, int numWaves, class Riemann ,class Limiter>
__global__ void fused_Riemann_limiter_vertical_update_kernel2(pdeParam param, real dt, Riemann Riemann_solver_v, Limiter phi)		// NOT WORKING WELL WITH LIMITERS???
{
	// Blocks are responsible to handle the update of certain cells
	// Some thread will just do the Riemann solution and finish their jobs
	// providing for other threads the waves and speeds they need to update
	// the states.
	int col = threadIdx.x + blockIdx.x*blockDim.x;
	int row = threadIdx.y + blockIdx.y*blockDim.y - 3*blockIdx.y;	// 1 if we want to use simple Riemann, 3 if we want to use limiters

	real* waves			= &shared_elem[0];
	real* waveSpeeds	= &shared_elem[(blockDim.x*blockDim.y)*numWaves*numStates];	// I do not know if we need to store the alpha, or how the computation for the limiters will be done, for now I keep things simple for the Riemann solvers, the number 256 is the number of threads per block

	// Riemann solver
	if (row < param.cellsY-1 && col < param.cellsX)	// if there are 512 cells in Y 511 threads are in use
	{
		Riemann_solver_v(&param.getElement_qNew(row, col, 0),		// input comes from global memory
							&param.getElement_qNew(row+1, col, 0),	//
							numStates,								//
							&param.getElement_coeff(row, col, 0),	//
							&param.getElement_coeff(row+1, col, 0),	//
			/*shared*/		&waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
			/*shared*/		&waveSpeeds[getIndex_waveSpeed(threadIdx.y, threadIdx.x, 0, numWaves, blockDim.x)]);
	}

	__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed

	if ( (threadIdx.y > 1 && threadIdx.y < blockDim.y-1) && row < param.cellsY-1 && col < param.cellsX )
	{
		// Update Incoming Fluctuations
		// Each active thread will look for the incoming waves to its cell, and for the update term

		real amdq_apdq[numStates];
		//#pragma unroll numStates
		for (int k = 0; k < numStates; k++)		// I do not know if there are better ways to do this... (this = initialize the fluctuations to 0)
		{
			amdq_apdq[k] = 0;
		}
		//#pragma unroll numWaves
		for (int w = 0; w < numWaves; w++)
		{
			/////////
			///////// THREAD IS LOOKING TO THE WAVES FROM THE INTERFACES TO ITS UP
			/////////
			real waveComingFromUp_waveSpeed = getWaveSpeed(waveSpeeds, threadIdx.y-1, threadIdx.x, w, numWaves, blockDim.x);//waveSpeeds[getIndex_waveSpeed(threadIdx.y-1, threadIdx.x, w, numWaves, blockDim.x)];

			if (waveComingFromUp_waveSpeed > 0)		// a warp divergence at Rarefaction and shock formation points
			{
				real limiting_factor = 1.0;/*limiting (phi,
												 &waves[getIndex_sharedWaves(threadIdx.y-1, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],	// main wave
												 &waves[getIndex_sharedWaves(threadIdx.y-2, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],	// aux wave
												 numStates);*/
				//#pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] +=		waveComingFromUp_waveSpeed * waves[getIndex_sharedWaves(threadIdx.y-1, threadIdx.x, w, k, numStates, numWaves, blockDim.x)];
									-
										0.5*waveComingFromUp_waveSpeed*(1-(dt/param.dy)*waveComingFromUp_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y-1, threadIdx.x, w, k, numStates, numWaves, blockDim.x)];
				}
			}
			else	// no incoming wave, but still a correction term
			{
				real limiting_factor = 1.0;/*limiting (phi,
												 &waves[getIndex_sharedWaves(threadIdx.y-1, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
												 &waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
												 numStates);*/
				//#pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] += - -0.5*waveComingFromUp_waveSpeed*(1+(dt/param.dy)*waveComingFromUp_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y-1, threadIdx.x, w, k, numStates, numWaves, blockDim.x)];
				}
			}
		}
		for (int w = 0; w < numWaves; w++)
		{
			/////////
			///////// THREAD IS LOOKING TO THE WAVES FROM THE INTERFACES TO ITS DOWN
			/////////
			real waveComingFromDown_waveSpeed = getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, w, numWaves, blockDim.x);//waveSpeeds[getIndex_waveSpeed(threadIdx.y, threadIdx.x, w, numWaves, blockDim.x)];

			if (waveComingFromDown_waveSpeed < 0)
			{
				real limiting_factor = 1.0;/*limiting (phi,
												 &waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
												 &waves[getIndex_sharedWaves(threadIdx.y+1, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
												 numStates);*/
				//#pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] +=		waveComingFromDown_waveSpeed * waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x)]
									+
										-0.5*waveComingFromDown_waveSpeed*(1+(dt/param.dy)*waveComingFromDown_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x)];
				}
			}
			else	// no incoming wave but still a correction term
			{
				real limiting_factor = 1.0;/*limiting (phi,
												 &waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
												 &waves[getIndex_sharedWaves(threadIdx.y-1, threadIdx.x, 0, 0, numStates, numWaves, blockDim.x)],
												 numStates);*/
				//#pragma unroll numStates
				for (int k = 0; k < numStates; k++)
				{
					amdq_apdq[k] += 0.5*waveComingFromDown_waveSpeed*(1-(dt/param.dy)*waveComingFromDown_waveSpeed)*limiting_factor*waves[getIndex_sharedWaves(threadIdx.y, threadIdx.x, w, k, numStates, numWaves, blockDim.x)];
				}
			}
		}

		// Final Update
		// !!! The condition here decides which thread is to update the cells
		// !!! The conditions are different for simple Riemann solver,
		// !!! than for the Riemann solver coupled with a limiter
		//#pragma unroll numStates
		for (int k = 0; k < numStates; k++)
		{
			param.setElement_q(row, col, k, param.getElement_q(row, col, k) - dt/param.dy * amdq_apdq[k]);
		}
	}
}

//#define dimGrid_h (5,2048,1)		// defintions for 2048 by 2048 mesh
//#define dimBlock_h (512,1,1)
//#define dimGrid_v (256,33,1)
//#define dimBlock_v (8,64,1)
//#define SharedMemorySize 16384

template <class Riemann_h, class Riemann_v, class Limiter>
/*extern "C"*/void limited_Riemann_Update(pdeParam &param,						// Problem parameters
										real dt,								//
										Riemann_h Riemann_pointwise_solver_h,	//
										Riemann_v Riemann_pointwise_solver_v,	//
										Limiter limiter_phi					//
										)
{
	// ! WE HAVE TO BE VERY CAREFUL WITH THE SIZES OF THESE BLOCKS AS WE ARE GOING TO USE
	// SHARED MEMORY AND WE HAVE TO KEEP IN MIND THE AMOUNT OF DATA THAT WILL FIT THERE
	
	// might use later on to be compatible automatically with the real data type (float vs double)
	// includes the size of a block x the number of waves x (the number of state + 1=for speed) x size of real
	// each horizontal line in a block will have to (re)compute the Riemann problem at its right end,
	// so if a horizontal slice has n interfaces the block has to compute n+1 interfaces the extra one being at
	// at the right end, an extra job for the rightmost thread. This is necessary to avoid a sync between neighbors,
	// which can be only done (as far as I know) through a global sync, requiring a kernel launch.
	// The recomputation can be minimized with a flat approach, however I do not know how it would effect the overall performance.
	// Possibly one could use the constant memory to make such barrier conditions fast and only for neighboring blocks...
	// I have not explored that possibilty any further.


	//const int numStates = 3;/*/param.numStates;/**/
	//const int numWaves  = 2;/*/param.numWaves;/**/
	{
		const unsigned int blockDim_X = 96;
		const unsigned int blockDim_Y = 2;

		size_t SharedMemorySize = (blockDim_X*blockDim_Y)*(param.numWaves)*(param.numStates+1)*sizeof(real);

		unsigned int gridDim_X = (param.cellsX + (blockDim_X-3-1)) / (blockDim_X-3);	// -1 for simple -3 for limiters
		unsigned int gridDim_Y = (param.cellsY + (blockDim_Y-1)) / blockDim_Y;

		dim3 dimGrid_h(gridDim_X, gridDim_Y);
		dim3 dimBlock_h(blockDim_X, blockDim_Y);
		fused_Riemann_limiter_horizontal_update_kernel<3, 2, blockDim_X*blockDim_Y, Riemann_h, Limiter><<<dimGrid_h, dimBlock_h, SharedMemorySize>>>(param, dt, Riemann_pointwise_solver_h, limiter_phi);
	}
	{
		const unsigned int blockDim_X = 16;		// fine tune the best block size
		const unsigned int blockDim_Y = 24;

		size_t SharedMemorySize = (blockDim_X*blockDim_Y)*(param.numWaves)*(param.numStates+1)*sizeof(real);

		unsigned int gridDim_X = (param.cellsX + (blockDim_X-1)) / blockDim_X;
		unsigned int gridDim_Y = (param.cellsY + (blockDim_Y-3-1)) / (blockDim_Y-3);	// -1 for simlpe -3 for limiters

		dim3 dimGrid_v(gridDim_X, gridDim_Y);
		dim3 dimBlock_v(blockDim_X, blockDim_Y);
		fused_Riemann_limiter_vertical_update_kernel<3, 2, blockDim_X*blockDim_Y, Riemann_v, Limiter><<<dimGrid_v, dimBlock_v, SharedMemorySize>>>(param, dt, Riemann_pointwise_solver_v, limiter_phi);
	}

	// Swap q and qNew before stepping
	// At this stage qNew is old and q has the latest state that is
	// because q was updated based on qNew, which right before step
	// held the latest update.
	real* temp = param.qNew;
	param.qNew = param.q;
	param.q = temp;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////// to resolve external function which is not supported
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// As a general rule a Riemann solver must have the following arguments:
// 2 cells, one left one right (equivalently up and down)
// 2 sets of coefficients one for the left cell the other for the right
// a vector amdq
// a vector apdq
// a set of waves
//
// These data will come from both the global and shared memory, this will be decided
// according to the needs of the limiter. It is certain that amdq and apdq and the waves
// will be stored in shared memory, however we might need the speeds of the waves and 
// perhaps the coefficients and even the data itself. The shared memory on 2.x architectures
// is common with the hardware of the L1 cache, we could potentially leave some of
// the required data there without control, and control the essentials with from the shared memory
//
// For the acoustics example, one would need:
// blockSize * ( q + (amdq apdq) + coeff + waves + speeds ) * sizeof(real) ... computed per cell/thread
// =
// 16*16 * (3 + (3 3) + 2 + 2*3 + 1)* (4 or 8) = 18Kbytes or 36Kbytes => 16/48 or 8/48 warps
//
// Eliminating the need to store at least the coefficients and data q, we can go up to full 48/48
// warp usage with 16*8 blocks. Remains to see what is needed for the limiters.
//
// A slightly independent note: The pointers that the Riemann solver will have can point to either global or
// shared memory. The exact location with offset will be resolved before the kernel calls the device function
// (the Riemann solver). The writer of the Riemann device function (the user) must be careful how he/she
// fills the data in amdq, apdq, wave speeds and the waves themselves. A certain convention must be maintained
// with the layout of the main data q, mainly [state1 state2 state3] [state1 state2 state3] ...

struct acoustics_horizontal
{
	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
								/*real* amdq, real* apdq,*/ real* wave, real* waveSpeeds)						// output
	{
		real rho_l = u_left[0];
		real bulk_l = u_left[1];

		real rho_r = u_right[0];
		real bulk_r = u_right[1];

		real c_l = sqrt(bulk_l/rho_l);	// sound speed
		real z_l = c_l*rho_l;			// impedance

		real c_r = sqrt(bulk_r/rho_r);
		real z_r = c_r*rho_r;

		waveSpeeds[0] = -c_l;
		waveSpeeds[1] = c_r;

		real alpha1 = ( q_left[0] - q_right[0] + z_r*(q_right[1] - q_left[1])) / (z_l+z_r);	// ( -(pr-pl) + zr(vr-vl) )/ (zl+zr)
		real alpha2 = ( q_right[0] - q_left[0] + z_l*(q_right[1] - q_left[1])) / (z_l+z_r);	// (  (pr-pl) + zl(vr-vl) )/ (zl+zr)

		wave[0 + 0*numStates] = -alpha1*z_l;
		wave[1 + 0*numStates] = alpha1;
		wave[2 + 0*numStates] = 0;

		wave[0 + 1*numStates] = alpha2*z_r;
		wave[1 + 1*numStates] = alpha2;
		wave[2 + 1*numStates] = 0;


		//amdq[0] = -c_l * wave[0 + 0*numStates];
		//amdq[1] = -c_l * wave[1 + 0*numStates];
		//amdq[2] =  0;							// 0   * wave[2 + 0*numStates];

		//apdq[0] = c_r * wave[0 + 1*numStates];
		//apdq[1] = c_r * wave[1 + 1*numStates];
		//apdq[2] = 0;							// 0   * wave[2 + 1*numStates];
	}
};
struct acoustics_vertical
{
	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
								/*real* amdq, real* apdq,*/ real* wave, real* waveSpeeds)						// output
	{
		real rho_l = u_left[0];
		real bulk_l = u_left[1];

		real rho_r = u_right[0];
		real bulk_r = u_right[1];

		real c_l = sqrt(bulk_l/rho_l);	// sound speed
		real z_l = c_l*rho_l;			// impedance

		real c_r = sqrt(bulk_r/rho_r);
		real z_r = c_r*rho_r;

		waveSpeeds[0] = -c_l;
		waveSpeeds[1] = c_r;

		real alpha1 = ( q_left[0] - q_right[0] + z_r*(q_right[2] - q_left[2])) / (z_l+z_r);	// ( -(pr-pl) + zr(vr-vl) )/ (zl+zr)
		real alpha2 = ( q_right[0] - q_left[0] + z_l*(q_right[2] - q_left[2])) / (z_l+z_r);	// (  (pr-pl) + zl(vr-vl) )/ (zl+zr)

		wave[0 + 0*numStates] = -alpha1*z_l;
		wave[1 + 0*numStates] = 0;
		wave[2 + 0*numStates] = alpha1;

		wave[0 + 1*numStates] = alpha2*z_r;
		wave[1 + 1*numStates] = 0;
		wave[2 + 1*numStates] = alpha2;

		//amdq[0] = -c_l * wave[0 + 0*numStates];
		//amdq[1] =  0;							// 0   * wave[1 + 0*numStates];
		//amdq[2] = -c_l * wave[2 + 0*numStates];

		//apdq[0] = c_r * wave[0 + 1*numStates];
		//apdq[1] = 0;							// 0   * wave[1 + 1*numStates];
		//apdq[2] = c_r * wave[2 + 1*numStates];
	}
};

struct limiter_none
{
	__device__ real operator() (real theta)
	{
		return (real)0.0;
	}
};
struct limiter_LaxWendroff
{
	__device__ real operator() (real theta)
	{
		return (real)1.0;
	}
};
struct limiter_MC
{
	__device__ real operator() (real theta)
	{
		real minimum = fmin((real)2.0, (real)2.0*theta);
		return fmax(0, fmin(((real)1.0+theta)/(real)2.0, minimum));
	}
};
struct limiter_superbee
{
	__device__ real operator() (real theta)
	{
		real maximum = fmax((real)0.0, fmin((real)1.0,(real)2.0*theta));
		return fmax(maximum, fmin((real)2.0,theta));
	}
};
struct limiter_VanLeer
{
	__device__ real operator() (real theta)
	{
		real absTheta = fabs(theta);
		return (theta + absTheta) / (1 + absTheta);
	}
};


#endif

//struct acoustics_horizontal_fake
//{
//	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
//								/*real* amdq, real* apdq,*/ real* wave, real* waveSpeeds)						// output
//	{
//		//real rho_l = 1.0f;//u_left[0];
//		//real bulk_l = 4.0f;//u_left[1];
//
//		//real rho_r = 1.0f;//u_right[0];
//		//real bulk_r = 4.0f;//u_right[1];
//
//		//real c_l = sqrt(bulk_l/rho_l);	// sound speed
//		//real z_l = c_l*rho_l;			// impedance
//
//		//real c_r = sqrt(bulk_r/rho_r);
//		//real z_r = c_r*rho_r;
//
//		real c = 2.0;
//		real z = 2.0;
//
//		waveSpeeds[0] = -c;
//		waveSpeeds[1] = c;
//
//		real alpha1 = ( q_left[0] - q_right[0] + z*(q_right[1] - q_left[1])) / (2*z);	// ( -(pr-pl) + zr(vr-vl) )/ (zl+zr)
//		real alpha2 = ( q_right[0] - q_left[0] + z*(q_right[1] - q_left[1])) / (2*z);	// (  (pr-pl) + zl(vr-vl) )/ (zl+zr)
//
//		wave[0 + 0*numStates] = -alpha1*z;
//		wave[1 + 0*numStates] = alpha1;
//		wave[2 + 0*numStates] = 0;
//
//		wave[0 + 1*numStates] = alpha2*z;
//		wave[1 + 1*numStates] = alpha2;
//		wave[2 + 1*numStates] = 0;
//
//		//amdq[0] = -c_l * wave[0 + 0*numStates];
//		//amdq[1] = -c_l * wave[1 + 0*numStates];
//		//amdq[2] =  0;							// 0   * wave[2 + 0*numStates];
//
//		//apdq[0] = c_r * wave[0 + 1*numStates];
//		//apdq[1] = c_r * wave[1 + 1*numStates];
//		//apdq[2] = 0;							// 0   * wave[2 + 1*numStates];
//	}
//};
//
//extern __shared__ real shared_elem[];
//template <const int numStates, int numWaves, class Riemann /*,class Limiter*/>
//__global__ void testKernel(pdeParam param, real dt, Riemann Riemann_solver_h)
//{
//	int col = threadIdx.x + blockIdx.x*blockDim.x - blockDim.x;//- 3*blockDim.x;	// 3 if we want to use limiters
//	int row = threadIdx.y + blockIdx.y*blockDim.y;
//
//	real apdq[numStates];
//	real amdq[numStates];
//
//	real* waves			= &shared_elem[0];
//	real* waveSpeeds	= &shared_elem[(32*8)*numWaves*numStates];	// I do not know if we need to store the alpha, or how the computation for the limiters will be done, for now I keep things simple for the Riemann solvers, the number 256 is the number of threads per block
//
//	// is param in the registers? how fast is it to get the param info? our pdeParams are very large, we might need a new parameter which holds only the necessary
//
//	// Riemann solver
//	if (row < param.cellsY && col < param.cellsX-1)	// if there are 512 cells in X 511 threads are in use
//	{
//		Riemann_solver_h(&param.getElement_qNew(row, col, 0),		// input comes from global memory
//							&param.getElement_qNew(row, col+1, 0),	//
//							numStates,								//
//							&param.getElement_coeff(row, col, 0),	//
//							&param.getElement_coeff(row, col+1, 0),	//
//			/*shared*/		&waves[threadIdx.x*numWaves*numStates + threadIdx.y*numWaves*numStates*blockDim.x],
//			/*shared*/		&waveSpeeds[threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x]);
//
//		__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed
//
//		// Update Fluctuation
//		// grab the necessary waves and add, the most general case would be to grab the wave and speed check the speed sign and choose where to add
//		// this might give rise to race conditions if addition is not handled properly. In acoustics' case we know the sign of the waves beforehand
//		// but this is not the case in general, and therefore a sign check must be made and perhaps atomics used to add to qNew, or
//		// (best for performance) uglyly held in many variables residing in thread registers which accumulate the necessary sums to be applied
//		// to qNew. Maybe with the use of the occupancy calculator we can determine the maximum number of such variables possible.
//		// I sleep now. I continue now.
//		//#pragma unroll numStates
//		for (int k = 0; k < numStates; k++)		// I do not know if there are better ways to do this... (this = initialize the fluctuations to 0)
//		{
//			amdq[k] = 0;
//			apdq[k] = 0;
//		}
//		//#pragma unroll numWaves
//		for (int w = 0; w < numWaves; w++)
//		{
//			real waveSpeed = waveSpeeds[w + threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x];
//			param.setElement_waveSpeed(row, col, w, waveSpeed);
//			if (waveSpeed < 0)
//			{
//				//#pragma unroll numStates
//				for (int k = 0; k < numStates; k++)
//				{
//					amdq[k] += waveSpeed * waves[k + w*numStates + threadIdx.x*numStates*numWaves + threadIdx.y*numStates*numWaves*blockDim.x];
//				}
//			}
//			else
//			{
//				//#pragma unroll numStates
//				for (int k = 0; k < numStates; k++)
//				{
//					apdq[k] += waveSpeed * waves[k + w*numStates + threadIdx.x*numStates*numWaves + threadIdx.y*numStates*numWaves*blockDim.x];
//				}
//			}
//		}
//
//		// Ignores all of the above
//		for (int k = 0; k < numStates; k++)
//		{
//			param.setElement_q(row, col, k, (real)(threadIdx.x+blockIdx.x*blockDim.x + (threadIdx.y+blockIdx.y*blockDim.y)*param.cellsX)/(real)(param.cellsX*param.cellsY));
//		}
//	}
//}


//extern __shared__ real shared_elem[];
//template <const int numStates, int numWaves, class Riemann ,class Limiter>
//__global__ void fused_Riemann_limiter_horizontal_update_kernel(pdeParam param, real dt, Riemann Riemann_solver_h, Limiter phi)
//{
//	int col = threadIdx.x + blockIdx.x*blockDim.x - blockIdx.x;//- 3*blockDim.x;	// 3 if we want to use limiters
//	int row = threadIdx.y + blockIdx.y*blockDim.y;
//
//	real apdq[numStates];
//	real amdq[numStates];
//
//	real* waves			= &shared_elem[0];
//	real* waveSpeeds	= &shared_elem[blockDim.x*blockDim.y*numWaves*numStates];	// I do not know if we need to store the alpha, or how the computation for the limiters will be done, for now I keep things simple for the Riemann solvers, the number 256 is the number of threads per block
//
//	// is param in the registers? how fast is it to get the param info? our pdeParams are very large, we might need a new parameter which holds only the necessary
//
//	// Riemann solver
//	if (row < param.cellsY && col < param.cellsX-1)	// if there are 512 cells in X 511 threads are in use
//	{
//		Riemann_solver_h(&param.getElement_qNew(row, col, 0),		// input comes from global memory
//							&param.getElement_qNew(row, col+1, 0),	//
//							numStates,								//
//							&param.getElement_coeff(row, col, 0),	//
//							&param.getElement_coeff(row, col+1, 0),	//
//			/*shared*/		&waves[threadIdx.x*numWaves*numStates + threadIdx.y*numWaves*numStates*blockDim.x],
//			/*shared*/		&waveSpeeds[threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x]);
//	}
//
//	__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed
//		
//	if (row < param.cellsY && col < param.cellsX-1 && (threadIdx.x > 0 && threadIdx.x < blockDim.x))
//	{
//		// Update Fluctuation
//		// grab the necessary waves and add, the most general case would be to grab the wave and speed check the speed sign and choose where to add
//		// this might give rise to race conditions if addition is not handled properly. In acoustics' case we know the sign of the waves beforehand
//		// but this is not the case in general, and therefore a sign check must be made and perhaps atomics used to add to qNew, or
//		// (best for performance) uglyly held in many variables residing in thread registers which accumulate the necessary sums to be applied
//		// to qNew. Maybe with the use of the occupancy calculator we can determine the maximum number of such variables possible.
//		// I sleep now. I continue now.
//		//#pragma unroll numStates
//		for (int k = 0; k < numStates; k++)		// I do not know if there are better ways to do this... (this = initialize the fluctuations to 0)
//		{
//			amdq[k] = 0;
//			apdq[k] = 0;
//		}
//		//#pragma unroll numWaves
//		for (int w = 0; w < numWaves; w++)
//		{
//			real waveSpeed = waveSpeeds[w + threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x];
//
//			if (waveSpeed < 0)
//			{
//				//#pragma unroll numStates
//				for (int k = 0; k < numStates; k++)
//				{
//					amdq[k] += waveSpeed * waves[k + w*numStates + threadIdx.x*numStates*numWaves + threadIdx.y*numStates*numWaves*blockDim.x];
//				}
//			}
//			else
//			{
//				//#pragma unroll numStates
//				for (int k = 0; k < numStates; k++)
//				{
//					apdq[k] += waveSpeed * waves[k + w*numStates + threadIdx.x*numStates*numWaves + threadIdx.y*numStates*numWaves*blockDim.x];
//				}
//			}
//		}
//
//		// No Limiter
//		//
//		// only concern here is: will the waves alone be enough to provide input to the limiter function phi?
//		// + I think we might need to call the limiter twice.
//
//		// Final Update
//		// left going waves first
//		// !!! The condition here decides which thread is to update the cells
//		// !!! The conditions are different for simple Riemann solver,
//		// !!! than for the Riemann solver coupled with a limiter
//		if ( threadIdx.x != 0 )
//		{
//			//#pragma unroll numStates
//			for (int k = 0; k < numStates; k++)
//			{
//				param.setElement_q(row, col, k, param.getElement_qNew(row, col, k) - dt/param.dx * amdq[k]);
//			}
//		}
//
//		// Sync between threads in the same block so that the right going waves apply correctly
//		__syncthreads();
//
//		// right going waves second
//		// !!! The condition here decides which thread is to update the cells
//		// !!! The conditions are different for simple Riemann solver,
//		// !!! than for the Riemann solver coupled with a limiter
//		if ( threadIdx.x != blockDim.x-1 )
//		{
//			//#pragma unroll numStates
//			for (int k = 0; k < numStates; k++)
//			{
//				param.setElement_q(row, col+1, k, param.getElement_q(row, col+1, k) - dt/param.dx * apdq[k]);
//			}
//		}
//	}
//}
//
//extern __shared__ real shared_elem[];
//template <const int numStates, int numWaves, class Riemann ,class Limiter>
//__global__ void fused_Riemann_limiter_vertical_update_kernel(pdeParam param, real dt, Riemann Riemann_solver_v, Limiter phi)
//{
//	int col = threadIdx.x + blockIdx.x*blockDim.x;
//	int row = threadIdx.y + blockIdx.y*blockDim.y - blockIdx.y;//- 3*blockDim.y;	// 3 if we want to use limiters
//
//	real apdq[numStates];
//	real amdq[numStates];
//
//	real* waves			= &shared_elem[0];
//	real* waveSpeeds	= &shared_elem[blockDim.x*blockDim.y*numWaves*numStates];	// I do not know if we need to store the alpha, or how the computation for the limiters will be done, for now I keep things simple for the Riemann solvers, the number 256 is the number of threads per block
//
//	// is param in the registers? how fast is it to get the param info? our pdeParams are very large, we might need a new parameter which holds only the necessary
//
//	// Riemann solver
//	if (row < param.cellsY-1 && col < param.cellsX)	// if there are 512 cells in X 511 threads are in use
//	{
//		Riemann_solver_v(&param.getElement_qNew(row, col, 0),		// input comes from global memory
//							&param.getElement_qNew(row+1, col, 0),	//
//							numStates,								//
//							&param.getElement_coeff(row, col, 0),	//
//							&param.getElement_coeff(row+1, col, 0),	//
//			/*shared*/		&waves[threadIdx.x*numWaves*numStates + threadIdx.y*numWaves*numStates*blockDim.x],
//			/*shared*/		&waveSpeeds[threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x]);
//	}
//
//		__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed
//		
//	if (row < param.cellsY-1 && col < param.cellsX)
//	{
//		// Update Fluctuation
//		// grab the necessary waves and add, the most general case would be to grab the wave and speed check the speed sign and choose where to add
//		// this might give rise to race conditions if addition is not handled properly. In acoustics' case we know the sign of the waves beforehand
//		// but this is not the case in general, and therefore a sign check must be made and perhaps atomics used to add to qNew, or
//		// (best for performance) uglyly held in many variables residing in thread registers which accumulate the necessary sums to be applied
//		// to qNew. Maybe with the use of the occupancy calculator we can determine the maximum number of such variables possible.
//		// I sleep now. I continue now.
//		//#pragma unroll numStates
//		for (int k = 0; k < numStates; k++)		// I do not know if there are better ways to do this... (this = initialize the fluctuations to 0)
//		{
//			amdq[k] = 0;
//			apdq[k] = 0;
//		}
//		//#pragma unroll numWaves
//		for (int w = 0; w < numWaves; w++)
//		{
//			real waveSpeed = waveSpeeds[w + threadIdx.x*numWaves + threadIdx.y*numWaves*blockDim.x];
//			param.setElement_waveSpeed(row, col, w, waveSpeed);
//			if (waveSpeed < 0)
//			{
//				//#pragma unroll numStates
//				for (int k = 0; k < numStates; k++)
//				{
//					amdq[k] += waveSpeed * waves[k + w*numStates + threadIdx.x*numStates*numWaves + threadIdx.y*numStates*numWaves*blockDim.x];
//				}
//			}
//			else
//			{
//				//#pragma unroll numStates
//				for (int k = 0; k < numStates; k++)
//				{
//					apdq[k] += waveSpeed * waves[k + w*numStates + threadIdx.x*numStates*numWaves + threadIdx.y*numStates*numWaves*blockDim.x];
//				}
//			}
//		}
//
//		// No Limiter
//		//
//		// only concern here is: will the waves alone be enough to provide input to the limiter function phi?
//		// + I think we might need to call the limiter twice.
//
//		// Final Update
//		// up going waves first
//		// !!! The condition here decides which thread is to update the cells
//		// !!! The conditions are different for simple Riemann solver,
//		// !!! than for the Riemann solver coupled with a limiter
//		if ( threadIdx.y != 0 )
//			//#pragma unroll numStates
//			for (int k = 0; k < numStates; k++)
//			{
//				param.setElement_q(row, col, k, param.getElement_q(row, col, k) - dt/param.dy * amdq[k]);
//			}
//
//		// Sync between threads in the same block so that the right going waves apply correctly
//		__syncthreads();
//
//		// right going waves second
//		// !!! The condition here decides which thread is to update the cells
//		// !!! The conditions are different for simple Riemann solver,
//		// !!! than for the Riemann solver coupled with a limiter
//		if ( threadIdx.y != blockDim.y-1 )
//		{
//			//#pragma unroll numStates
//			for (int k = 0; k < numStates; k++)
//			{
//				param.setElement_q(row+1, col, k, param.getElement_q(row+1, col, k) - dt/param.dy * apdq[k]);
//			}
//		}
//	}
//}
