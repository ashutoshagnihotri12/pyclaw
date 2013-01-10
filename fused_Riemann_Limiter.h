#ifndef __FUSED_RIEMANN_LIMITER_H__
#define __FUSED_RIEMANN_LIMITER_H__

#include "fused_Riemann_Limiter_headers.h"
#include "reduce_Max.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////   Fused Solvers   //////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ real shared_elem[];
template <const int numStates, const int numWaves, const int numCoeff, const unsigned int blockSize, class Riemann ,class Limiter>
__global__ void fused_Riemann_limiter_horizontal_update_kernel(pdeParam param, Riemann Riemann_solver_h, Limiter phi)
{
	int col = threadIdx.x + blockIdx.x*HORIZONTAL_BLOCKSIZEX - 3*blockIdx.x;//- 3*blockDim.x;	// 3 if we want to use limiters
	int row = threadIdx.y + blockIdx.y*HORIZONTAL_BLOCKSIZEY;
	
	real dt = *param.dt;

	real apdq[numStates];
	real amdq[numStates];

	real* waves			= &shared_elem[0];
	real* waveSpeeds	= &shared_elem[blockSize*numWaves*numStates];

	real leftCell[numStates];	real leftCoeff[numCoeff];
	real rightCell[numStates];  real rightCoeff[numCoeff];

	bool grid_valid = row < param.cellsY && col < param.cellsX-1;
	bool grid_block_valid =  grid_valid && threadIdx.x < HORIZONTAL_BLOCKSIZEX-1;

	if (blockIdx.x > gridDim.x-3 || blockIdx.y > gridDim.y-2)
	{
		#pragma unroll
		for (int i = 0; i < numWaves; i++)
			setWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, i, numWaves, HORIZONTAL_BLOCKSIZEX, (real)0.0f);
	}

	// Riemann solver
	if (grid_valid)	// if there are 512 cells in X 511 threads are in use
	{
		#pragma unroll
		for (int i = 0; i < numStates; i++)
		{
			leftCell[i] = param.getElement_qNew(row,col,i);
			rightCell[i] = param.getElement_qNew(row,col+1,i);
		}
		#pragma unroll
		for (int i = 0; i < numCoeff; i++)
		{
			leftCoeff[i] = param.getElement_coeff(row,col,i);
			rightCoeff[i] = param.getElement_coeff(row,col+1,i);
		}
		Riemann_solver_h(leftCell,			// input comes from global memory
							rightCell,		//
							numStates,		//
							leftCoeff,		//
							rightCoeff,		//
			/*shared*/		&getSharedWave(waves, threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, HORIZONTAL_BLOCKSIZEX),
			/*shared*/		&getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, 0, numWaves, HORIZONTAL_BLOCKSIZEX));
	}
	__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed

	if (grid_block_valid && threadIdx.x > 0)
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
			real waveSpeed =  getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, w, numWaves, HORIZONTAL_BLOCKSIZEX);
			real* mainWave = &getSharedWave(waves, threadIdx.y, threadIdx.x, w, 0, numStates, numWaves, HORIZONTAL_BLOCKSIZEX);

			if (waveSpeed < (real)0.0f)
			{
				real limiting_factor = limiting<numStates>(phi,
												mainWave,
												&getSharedWave(waves, threadIdx.y, threadIdx.x+1, w, 0, numStates, numWaves, HORIZONTAL_BLOCKSIZEX));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, HORIZONTAL_BLOCKSIZEX);
					amdq[k] +=	waveSpeed * wave_state
							+
								-(real)0.5f*waveSpeed*(1+(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
				}
			}
			else
			{
				real limiting_factor = limiting<numStates>(phi,
												mainWave,
												&getSharedWave(waves, threadIdx.y, threadIdx.x-1, w, 0, numStates, numWaves, HORIZONTAL_BLOCKSIZEX));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, HORIZONTAL_BLOCKSIZEX);
					apdq[k] +=	waveSpeed * wave_state
							-
								(real)0.5f*waveSpeed*(1-(dt/param.dx)*waveSpeed)*limiting_factor*wave_state;
				}
			}

		}
	}
	__syncthreads();	// unmovable syncthreads
	// Local Reduce over Wave Speeds
	// Stage 1
	// Bringing down the number of elements to compare to block size
	int tid = threadIdx.x + threadIdx.y*HORIZONTAL_BLOCKSIZEX;
	waveSpeeds[tid] = fabs(waveSpeeds[tid]);//fmax(waveSpeeds[tid], -waveSpeeds[tid]);
	// alternative: use fabs to do one block away and start loop from second block
	// caveat: may have just a single wave, so looking ahead a block will take us out of bounds
	// waveSpeeds[tid] = fmax(fabs(waveSpeeds[tid]), fabs(waveSpeeds[tid + blockSize]));
	// waveSpeeds[tid + blockSize] could possible not exist.
	#pragma unroll
	for (int i = 1; i < numWaves; i++)
	{
		// blockDim.x * blockDim.y is usally a multiple of 32, so there should be no bank conflicts.
		waveSpeeds[tid] = fmax(waveSpeeds[tid], fabs(waveSpeeds[tid + i*blockSize]));
	}

	/////////////////////////////////////////////////////////////////////// 
	// At this stage the threads consumed the generated waves, and have no further use for them.
	if (grid_block_valid && threadIdx.x > 0)
	{
		// store the right going fluctuation onto shared to let other\neighboring thread grab thread them.

		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			setSharedWave(waves, threadIdx.y, threadIdx.x, 0, k, numStates, numWaves, HORIZONTAL_BLOCKSIZEX, apdq[k]);
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
	__syncthreads();	// unmovable syncthreads

	// Stage 2
	// Reducing over block size elements
	// use knowledge about your own block size:
	// I am using blocks of size 192 for the horizontal direction
	// At this stage there is no need to use fabs again, as all speeds were taken absolutely
	if (BLOCKSIZEX >= 64 )
	if (blockSize >= BLOCKSIZEX)
	{
		if (tid < BLOCKSIZEX/2)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEX/2]);
		__syncthreads();
	}
	if (BLOCKSIZEX/2 >= 64 )
	if (blockSize >= BLOCKSIZEX/2)
	{
		if (tid < BLOCKSIZEX/4)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEX/4]);
		__syncthreads();
	}
	if (BLOCKSIZEX/4 >= 64 )
	if (blockSize >= BLOCKSIZEX/4)
	{
		if (tid < BLOCKSIZEX/8)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEX/8]);
		__syncthreads();
	}
	if (BLOCKSIZEX/8 >= 64 )
	if (blockSize >= BLOCKSIZEX/8)
	{
		if (tid < BLOCKSIZEX/16)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEX/16]);
		__syncthreads();
	}
	if (tid < 32)
		warpReduce<blockSize>(waveSpeeds, tid);
	

	/////////////////////////////////////////////////////////////////////// Incrusted Update stage
	if (grid_block_valid && threadIdx.x > 1)							///
	{
		// Final Update, in One Shot
		// !!! The condition here decides which thread is to update the cells
		// !!! The conditions are different for simple Riemann solver,
		// !!! than for the Riemann solver coupled with a limiter
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			param.setElement_q(row, col, k, param.getElement_qNew(row, col, k) - dt/param.dx * (amdq[k] + getSharedWave(waves, threadIdx.y, threadIdx.x-1, 0, k, numStates, numWaves, HORIZONTAL_BLOCKSIZEX)));
		}
	}

	if (tid == 0)
		param.waveSpeedsX[blockIdx.x + blockIdx.y*gridDim.x] =  waveSpeeds[0];
}

extern __shared__ real shared_elem[];
template <const int numStates, const int numWaves, const int numCoeff, const unsigned int blockSize, class Riemann ,class Limiter>
__global__ void fused_Riemann_limiter_vertical_update_kernel(pdeParam param, Riemann Riemann_solver_v, Limiter phi)
{
	int col = threadIdx.x + blockIdx.x*VERTICAL_BLOCKSIZEX;
	int row = threadIdx.y + blockIdx.y*VERTICAL_BLOCKSIZEY - 3*blockIdx.y;//- 3*blockDim.y;	// 3 if we want to use limiters

	real dt = *param.dt;

	real apdq[numStates];
	real amdq[numStates];

	real* waves			= &shared_elem[0];
	real* waveSpeeds	= &shared_elem[blockSize*numWaves*numStates];	
	
	real upCell[numStates];		real upCoeff[numCoeff];
	real downCell[numStates];	real downCoeff[numCoeff];

	bool grid_valid = row < param.cellsY-1 && col < param.cellsX;
	bool grid_block_valid =  grid_valid && threadIdx.y < VERTICAL_BLOCKSIZEY-1;

	if (blockIdx.x > gridDim.x-2 || blockIdx.y > gridDim.y-3)
	{
		#pragma unroll
		for (int i = 0; i < numWaves; i++)
			setWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, i, numWaves, VERTICAL_BLOCKSIZEX, (real)0.0f);
	}

	// Riemann solver
	if (grid_valid)	// if there are 512 cells in Y 511 threads are in use
	{
		#pragma unroll
		for (int i = 0; i < numStates; i++)
		{
			upCell[i] = param.getElement_qNew(row+1,col,i);
			downCell[i] = param.getElement_qNew(row,col,i);
		}
		#pragma unroll
		for (int i = 0; i < numCoeff; i++)
		{
			upCoeff[i] = param.getElement_coeff(row+1,col,i);
			downCoeff[i] = param.getElement_coeff(row,col,i);
		}

		Riemann_solver_v(	downCell,		// input comes from global memory
							upCell,			//
							numStates,		//
							downCoeff,		//
							upCoeff,		//
			/*shared*/		&getSharedWave(waves, threadIdx.y, threadIdx.x, 0, 0, numStates, numWaves, VERTICAL_BLOCKSIZEX),
			/*shared*/		&getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, 0, numWaves, VERTICAL_BLOCKSIZEX));
	}

	__syncthreads();	// a busy barrier if it exits, would be more suitable here, we only need to make sure that the waves were computed

	if(grid_block_valid && threadIdx.y > 0)
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
			real waveSpeed =  getWaveSpeed(waveSpeeds, threadIdx.y, threadIdx.x, w, numWaves, VERTICAL_BLOCKSIZEX);
			real* mainWave = &getSharedWave(waves, threadIdx.y, threadIdx.x, w, 0, numStates, numWaves, VERTICAL_BLOCKSIZEX);

			if (waveSpeed < (real)0.0f)
			{
				real limiting_factor = limiting<numStates>(phi,
												mainWave,
												&getSharedWave(waves, threadIdx.y+1, threadIdx.x, w, 0, numStates, numWaves, VERTICAL_BLOCKSIZEX));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, VERTICAL_BLOCKSIZEX);
					amdq[k] +=	waveSpeed * wave_state
							+
								-(real)0.5f*waveSpeed*(1+(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
				}
			}
			else
			{
				real limiting_factor = limiting<numStates>(phi,
												mainWave,
												&getSharedWave(waves, threadIdx.y-1, threadIdx.x, w, 0, numStates, numWaves, VERTICAL_BLOCKSIZEX));
				#pragma unroll
				for (int k = 0; k < numStates; k++)
				{
					real wave_state = getSharedWave(waves, threadIdx.y, threadIdx.x, w, k, numStates, numWaves, VERTICAL_BLOCKSIZEX);
					apdq[k] +=	waveSpeed * wave_state
							-
								(real)0.5f*waveSpeed*(1-(dt/param.dy)*waveSpeed)*limiting_factor*wave_state;
				}
			}
		}
	}
	__syncthreads();	// unmovable syncthreads
	
	// Local Reduce over Wave Speeds
	// Stage 1
	// Bringing down the number of elements to compare to block size
	int tid = threadIdx.x + threadIdx.y*VERTICAL_BLOCKSIZEX;
	// See horizontal version for comments
	waveSpeeds[tid] = fabs(waveSpeeds[tid]);
	#pragma unroll
	for (int i = 1; i < numWaves; i++)
	{
		// blockDim.x * blockDim.y is usally a multiple of 32, so there should be no bank conflicts.
		waveSpeeds[tid] = fmax(waveSpeeds[tid], fabs(waveSpeeds[tid + i*blockSize]));
	}

	///////////////////////////////////////////////////////////////////////
	if(grid_block_valid && threadIdx.y > 0)								///	Stage 1 local reduce->copy flux data to shared->Stage 2
	{
		// store down going fluctuation is shared, to pass them to the bottom neighbor threads
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			setSharedWave(waves, threadIdx.y, threadIdx.x, 0, k, numStates, numWaves, VERTICAL_BLOCKSIZEX, apdq[k]);
		}
	}
	///////////////////////////////////////////////////////////////////////

	__syncthreads();	// unmovable syncthread
	
	if (BLOCKSIZEY >= 64)
	if (blockSize >= BLOCKSIZEY)
	{
		if (tid < BLOCKSIZEY/2)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEY/2]);
		__syncthreads();
	}
	if (BLOCKSIZEY/2 >= 64)
	if (blockSize >= BLOCKSIZEY/2)
	{
		if (tid < BLOCKSIZEY/4)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEY/4]);
		__syncthreads();
	}
	if (BLOCKSIZEY/4 >= 64)
	if (blockSize >= BLOCKSIZEY/4)
	{
		if (tid < BLOCKSIZEY/8)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEY/8]);
		__syncthreads();
	}
	if (BLOCKSIZEY/8 >= 64)
	if (blockSize >= BLOCKSIZEY/8)
	{
		if (tid < BLOCKSIZEY/16)
			waveSpeeds[tid] = fmax(waveSpeeds[tid], waveSpeeds[tid + BLOCKSIZEY/16]);
		__syncthreads();
	}
	if (tid < 32)
		warpReduce<blockSize>(waveSpeeds, tid);

	///////////////////////////////////////////////////////////////////////	incrusted update stage
	if(grid_block_valid && threadIdx.y > 1)								///
	{
		// Final Update, in One Shot
		// !!! The condition here decides which thread is to update the cells
		// !!! The conditions are different for simple Riemann solver,
		// !!! than for the Riemann solver coupled with a limiter
		#pragma unroll
		for (int k = 0; k < numStates; k++)
		{
			param.setElement_q(row, col, k, param.getElement_q(row, col, k) - dt/param.dy * (amdq[k]+getSharedWave(waves, threadIdx.y-1, threadIdx.x, 0, k, numStates, numWaves, VERTICAL_BLOCKSIZEX)));
		}
	}																	///
	///////////////////////////////////////////////////////////////////////

	if (tid == 0)
		param.waveSpeedsY[blockIdx.x + blockIdx.y*gridDim.x] =  waveSpeeds[0];
}

extern __shared__ real shared_elem[];
__global__ void timeStepAdjust(pdeParam param, real desired_CFL, int size1, int size2)
{
	real* finalist_speeds = &shared_elem[0];		// volatile?

	finalist_speeds[threadIdx.x+32] = 0.0f;

	if (threadIdx.x < size1)
		finalist_speeds[threadIdx.x] = param.waveSpeedsX[threadIdx.x];
	else
		finalist_speeds[threadIdx.x] = 0.0f;

	if (threadIdx.x + 32 < size1)
		finalist_speeds[threadIdx.x] = fmax(finalist_speeds[threadIdx.x], param.waveSpeedsX[threadIdx.x+32]);
	
	warpReduce<32>(finalist_speeds, threadIdx.x);
	real speedx = finalist_speeds[0];
	
	if (threadIdx.x < size2)
		finalist_speeds[threadIdx.x] = param.waveSpeedsY[threadIdx.x];
	else
		finalist_speeds[threadIdx.x] = 0.0f;

	if (threadIdx.x + 32 < size2)
		finalist_speeds[threadIdx.x] = fmax(finalist_speeds[threadIdx.x], param.waveSpeedsY[threadIdx.x+32]);


	warpReduce<32>(finalist_speeds, threadIdx.x);
	real speedy = finalist_speeds[0];

	if (threadIdx.x == 0)
		param.waveSpeedsX[0] = desired_CFL/(speedx/param.dx + speedy/param.dy);
}

__global__ void timeStepAdjust_simple(pdeParam param)
{
	// check CFL violation and determine next time step dt
	real u = param.waveSpeedsX[0];
	real v = param.waveSpeedsY[0];
	real dx = param.dx;
	real dy = param.dy;
	real dt = *param.dt;
	real dt_used = *param.dt_used;
	if (dt*(u/dx + v/dy) > 1.0f)	// if CFL condition violated
	{
		// simulation did not advance, no time is to be incremented to the global simulation time
		dt_used = 0.0f;
		// compute a new dt with a stricter assumption
		dt = param.CFL_lower_bound/(u/dx + v/dy);

		// condition failed do not swap buffers, effectively reverts the time step
		*param.revert = false;
	}
	else	// else if no violation was recorded
	{
		// remember the time step used to increment the global simulation time
		dt_used = dt;
		// compute the the next dt to be used
		dt = param.desired_CFL/(u/dx + v/dy);

		// allow buffer swap
		*param.revert = true;
	}
	*param.dt = dt;
	*param.dt_used = dt_used;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////   Wrapper Function   ////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Riemann_h, class Riemann_v, class Limiter>
/*extern "C"*/void limited_Riemann_Update(pdeParam &param,						// Problem parameters
										Riemann_h Riemann_pointwise_solver_h,	//
										Riemann_v Riemann_pointwise_solver_v,	//
										Limiter limiter_phi						//
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
	
	int horizontal_gridSize, vertical_gridSize, depth_gridSize;
	{
		const unsigned int blockDim_X = HORIZONTAL_BLOCKSIZEX;
		const unsigned int blockDim_Y = HORIZONTAL_BLOCKSIZEY;

		size_t SharedMemorySize = (blockDim_X*blockDim_Y)*(param.numWaves)*(param.numStates+1)*sizeof(real);

		unsigned int gridDim_X = (param.cellsX-1 + (blockDim_X-3-1)) / (blockDim_X-3);	// -1 for simple -3 for limiters
		unsigned int gridDim_Y = (param.cellsY + (blockDim_Y-1)) / blockDim_Y ;

		dim3 dimGrid_h(gridDim_X, gridDim_Y);
		dim3 dimBlock_h(blockDim_X, blockDim_Y);
		fused_Riemann_limiter_horizontal_update_kernel<NUMSTATES, NUMWAVES, NUMCOEFF, blockDim_X*blockDim_Y, Riemann_h, Limiter><<<dimGrid_h, dimBlock_h, SharedMemorySize>>>(param, Riemann_pointwise_solver_h, limiter_phi);
		
		horizontal_gridSize = gridDim_X * gridDim_Y;
	}
	{
		const unsigned int blockDim_X = VERTICAL_BLOCKSIZEX;		// fine tune the best block size
		const unsigned int blockDim_Y = VERTICAL_BLOCKSIZEY;

		size_t SharedMemorySize = (blockDim_X*blockDim_Y)*(param.numWaves)*(param.numStates+1)*sizeof(real);

		unsigned int gridDim_X = (param.cellsX + (blockDim_X-1)) / blockDim_X;
		unsigned int gridDim_Y = (param.cellsY-1 + (blockDim_Y-3-1)) / (blockDim_Y-3);	// -1 for simlpe -3 for limiters

		dim3 dimGrid_v(gridDim_X, gridDim_Y);
		dim3 dimBlock_v(blockDim_X, blockDim_Y);
		fused_Riemann_limiter_vertical_update_kernel<NUMSTATES, NUMWAVES, NUMCOEFF, blockDim_X*blockDim_Y, Riemann_v, Limiter><<<dimGrid_v, dimBlock_v, SharedMemorySize>>>(param, Riemann_pointwise_solver_v, limiter_phi);
		
		vertical_gridSize = gridDim_X * gridDim_Y;
	}
	{
		const unsigned int blockDim_X = 512;		// fine tune the best block size
		
		size_t SharedMemorySize = (blockDim_X)*sizeof(real);
		unsigned int gridDim_X1, gridDim_X2;
		
		gridDim_X1 = 1;

		dim3 dimGrid1(gridDim_X1);
		dim3 dimBlock1(blockDim_X);
		
		reduceMax_simplified<blockDim_X><<< dimGrid1, dimBlock1, SharedMemorySize>>>(param.waveSpeedsX, horizontal_gridSize);
		
		gridDim_X2 = 1;

		dim3 dimGrid2(gridDim_X2);
		dim3 dimBlock2(blockDim_X);
		
		reduceMax_simplified<blockDim_X><<< dimGrid2, dimBlock2, SharedMemorySize>>>(param.waveSpeedsY, vertical_gridSize);
	
		timeStepAdjust_simple<<<1,1>>>(param);

		bool revert;
		cudaMemcpy(&revert, param.revert, sizeof(bool), cudaMemcpyDeviceToHost);
		if (revert)
		{
			// Swap q and qNew before stepping again
			// At this stage qNew became old and q has the latest state that is
			// because q was updated based on qNew, which right before 'step'
			// held the latest update.
			real* temp = param.qNew;
			param.qNew = param.q;
			param.q = temp;
		}
	}
}

#endif