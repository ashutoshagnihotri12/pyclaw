#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <string>

typedef double real;

struct pdeParam
{
	int cellsX;				// number of cells on the horizontal, including ghost cells (unpadded)
	int cellsY;				// number of cells on the vertical, including ghost cells
	const int ghostCells;	// ghost cells for the boundaries
	const int numStates;	// number of states of the problem, number of scalars per cell
	const int numWaves;		// number of nonzero waves generated by the Riemann solver (usually as many as there are equations)
	const int numCoeff;		// number of different types medium properties

	bool entropy_fix;		// entropy fix flag
	bool second_order;		// second order accuracy computation flag

	real width;				// physical domain's width
	real height;			// physical domain's height;
	real dx;				// physical domain's spacial discritization size on the horizontal
	real dy;				// physical domain's spacial discritization size on the vertical
	const real startX;		// physical domain's starting x in cartesian coordinates
	const real endX;		// physical domain's ending x in cartesian coordinates
	const real startY;		// physical domain's starting y in cartesian coordinates
	const real endY;		// physical domain's ending y in cartesian coordinates

	const real startTime;	// simulation start time
	const real endTime;		// simulation end time

	real desired_CFL;		// a CFL number which determines the next time step length (dt)
	real CFL_lower_bound;	// lower bound on the CFL to be used

	bool snapshots;				// flag to allow or disallow solution snapshots
	real snapshotTimeInterval;	// time interval between snapshots
	
	real* dt;					// time step to be used in the computation
	real* dt_used;				// time step used in the ccompleted computation (without reverting), used to increment global simulation time
	bool* revert;				// boolean flag to determine whether the time step has to be reverted, this is because we cannot swap the buffer
								// on the GPU itself, and it has to be done on CPU, and this is the flag that let's the CPU know what to do.
								// Alternatively, leave all the work of determining CFL violation and time step calculation to the CPU.

	real* coefficients;			// physical medium coefficients					//	GPU residents
	real* q;					// data, cells' states							//
	real* qNew;					// intermediate update holder
	real* waveSpeedsX;			// speed of horizontal waves					//
	real* waveSpeedsY;			// speed of vertical waves						//

	pdeParam()
		: cellsX(256), cellsY(256), ghostCells(2), numStates(1), numWaves(1), numCoeff(1), startX(0), endX(1), startY(0), endY(1), startTime(0.0f), endTime(0.0f)
	{
		width = endX-startX;
		height = endY-startY;
		dx = width/cellsX;
		dy = height/cellsY;

		entropy_fix = false;
		second_order = true;

		desired_CFL = 0.90f;
		CFL_lower_bound = 0.45f;

		real dt_cpu = 10.0f;
		real dt_used_cpu = 0.0f;
		
		cudaError_t alloc_real1 = cudaMalloc((void**)&dt, sizeof(real));
		cudaError_t alloc_real2 = cudaMalloc((void**)&dt_used, sizeof(real));
		cudaError_t alloc_bool1 = cudaMalloc((void**)&revert, sizeof(bool));
		
		cudaMemcpy(dt, &dt_cpu, sizeof(real), cudaMemcpyHostToDevice);
		cudaMemcpy(dt_used, &dt_used_cpu, sizeof(real), cudaMemcpyHostToDevice);

		snapshots = false;

		int cells = cellsX*cellsY;
		int horizontal_blocks = ((cellsX-1 + HORIZONTAL_BLOCKSIZEX-3-1)/(HORIZONTAL_BLOCKSIZEX-3)) * ((cellsY + HORIZONTAL_BLOCKSIZEY-1)/(HORIZONTAL_BLOCKSIZEY));
		int vertical_blocks = ((cellsX + VERTICAL_BLOCKSIZEX-1)/(VERTICAL_BLOCKSIZEX)) * ((cellsY-1 + VERTICAL_BLOCKSIZEY-3-1)/(VERTICAL_BLOCKSIZEY-3));

		cudaError_t alloc1 = cudaMalloc((void**)&coefficients, cells*numCoeff*sizeof(real));
		cudaError_t alloc2 = cudaMalloc((void**)&q, cells*numStates*sizeof(real));
		cudaError_t alloc3 = cudaMalloc((void**)&qNew, cells*numStates*sizeof(real));
		cudaError_t alloc4 = cudaMalloc((void**)&waveSpeedsX, horizontal_blocks*sizeof(real));
		cudaError_t alloc5 = cudaMalloc((void**)&waveSpeedsY, vertical_blocks*sizeof(real));

		real* zerosX = (real*)calloc(horizontal_blocks, sizeof(real));
		real* zerosY = (real*)calloc(vertical_blocks, sizeof(real));
		cudaMemcpy(waveSpeedsX, zerosX, horizontal_blocks*sizeof(real), cudaMemcpyHostToDevice);
		cudaMemcpy(waveSpeedsY, zerosY, vertical_blocks*sizeof(real), cudaMemcpyHostToDevice);
		free(zerosX);
		free(zerosY);
	};
	pdeParam(int cellsX, int cellsY, int ghostCells, int numStates, int numWaves, int numCoeff,
		real startX, real endX, real startY, real endY, real startTime, real endTime)
		:
	cellsX(cellsX), cellsY(cellsY), ghostCells(ghostCells), numStates(numStates), numWaves(numWaves), numCoeff(numCoeff),
		startX(startX), endX(endX), startY(startY), endY(endY), startTime(startTime), endTime(endTime)
	{
		width = endX-startX;
		height = endY-startY;
		dx = width/cellsX;
		dy = height/cellsY;

		entropy_fix = false;
		second_order = true;
		
		desired_CFL = 0.90f;
		CFL_lower_bound = 0.9f;

		real dt_cpu = 10.0f;
		real dt_used_cpu = 0.0f;
		
		cudaError_t alloc_real1 = cudaMalloc((void**)&dt, sizeof(real));
		cudaError_t alloc_real2 = cudaMalloc((void**)&dt_used, sizeof(real));
		cudaError_t alloc_bool1 = cudaMalloc((void**)&revert, sizeof(bool));
		
		cudaMemcpy(dt, &dt_cpu, sizeof(real), cudaMemcpyHostToDevice);
		cudaMemcpy(dt_used, &dt_used_cpu, sizeof(real), cudaMemcpyHostToDevice);

		snapshots = false;

		int cells = cellsX*cellsY;
		int horizontal_blocks = ((CELLSX-1 + HORIZONTAL_BLOCKSIZEX-3-1)/(HORIZONTAL_BLOCKSIZEX-3)) * ((CELLSY + HORIZONTAL_BLOCKSIZEY-1)/(HORIZONTAL_BLOCKSIZEY));
		int vertical_blocks = ((CELLSX + VERTICAL_BLOCKSIZEX-1)/(VERTICAL_BLOCKSIZEX)) * ((CELLSY-1 + VERTICAL_BLOCKSIZEY-3-1)/(VERTICAL_BLOCKSIZEY-3));

		cudaError_t alloc1 = cudaMalloc((void**)&coefficients, cells*numCoeff*sizeof(real));
		cudaError_t alloc2 = cudaMalloc((void**)&q, cells*numStates*sizeof(real));
		cudaError_t alloc3 = cudaMalloc((void**)&qNew, cells*numStates*sizeof(real));
		cudaError_t alloc4 = cudaMalloc((void**)&waveSpeedsX, horizontal_blocks*sizeof(real));
		cudaError_t alloc5 = cudaMalloc((void**)&waveSpeedsY, vertical_blocks*sizeof(real));

		// Slightly fragile approach:
		// We have to assign for every "active block" a single real
		// This number has to be exact, but forcing an initial zero, we can
		// allocate an approximate memory size, which works, but feels fragile
		// This way we take into account all blocks, including some that might not be active
		// Note: by active we mean that a block is not at the very edge and participate in
		// the computation by at least a wave ;

		real* zerosX = (real*)calloc(horizontal_blocks, sizeof(real));
		real* zerosY = (real*)calloc(vertical_blocks, sizeof(real));
		cudaMemcpy(waveSpeedsX, zerosX, horizontal_blocks*sizeof(real), cudaMemcpyHostToDevice);
		cudaMemcpy(waveSpeedsY, zerosY, vertical_blocks*sizeof(real), cudaMemcpyHostToDevice);
		free(zerosX);
		free(zerosY);
	};
	void setOrderOfAccuracy(int order)
	{
		if(order == 2)
			second_order = true;
		else
			second_order = false;
	}
	void setDesiredCFL(real desired_CFL)
	{
		this->desired_CFL = desired_CFL;
	}
	void setLowerBoundCFL(real lower_bound_CFL)
	{
		CFL_lower_bound = lower_bound_CFL;
	}

	void clean()
	{
		cudaFree(waveSpeedsY);
		cudaFree(waveSpeedsX);
		cudaFree(q);
		cudaFree(qNew);
		cudaFree(coefficients);
		cudaFree(revert);
		cudaFree(dt_used);
		cudaFree(dt);
	};

	void setSnapshotRate(real interval)
	{
		snapshots = true;
		snapshotTimeInterval = interval;
	};

	// Taking a solution snapshot saves the solution onto disk
	// there is a matlab script file that reads and displays the snapshot
	void takeSnapshot(int iteration, char* fileName="pde data")
	{
		// copy data from GPU to CPU
		// create CPU buffers then copy
		int cells = cellsX*cellsY;
		real* cpu_q			   = (real*)malloc(cells*numStates*sizeof(real));
		real* cpu_coefficients = (real*)malloc(cells*numCoeff*sizeof(real));

		cudaError_t cpy_err1 = cudaMemcpy(cpu_q, q, cells*numStates*sizeof(real), cudaMemcpyDeviceToHost);
		cudaError_t cpy_err2 = cudaMemcpy(cpu_coefficients, coefficients, cells*numCoeff*sizeof(real), cudaMemcpyDeviceToHost);

		// copy data from CPU to disk
		real xRange = endX-startX;
		real yRange = endY-startY;

		int size = cellsX*cellsY;

		float* chunk = (float*)malloc(cellsY*sizeof(float));
		
		FILE* pdeData;
		char filename[256];
		sprintf(filename, "%s%i.dat", fileName, iteration);

		pdeData = fopen(filename, "wb");

		fprintf(pdeData, "%i,%i,%f,%f,%f,%f,%i.\n", cellsX, cellsY, startX, endX, startY, endY, numStates);

		for (int state = 0; state < numStates; state++)
		{
			for (int row = 0; row < cellsX; row++)
			{
				memcpy(chunk, &cpu_q[state*size + row*cellsY], cellsY*sizeof(float));
				for(int col = 0; col < cellsY; col++)
				{
					fprintf(pdeData, "%f ", chunk[col]);
				}
				fprintf(pdeData, "\n");
			}
			fprintf(pdeData, "\n");
		}
		fclose(pdeData);
		free(chunk);
	};

	// GETTER AND SETTER FUNCTIONS
	// Q
	// Dictates how the memory layout for q will be
	inline __device__ __host__ int getIndex_q(int row, int column, int state)
	{
		// Usual C/C++ row major order
		//return (row*cellsX*numStates + column*numStates + state);
		// state is the slowest moving dimension now, then row, then column
		return (state*cellsX*cellsY + row*cellsX + column);
	}
	inline __device__ real &getElement_q(int row, int column, int state)
	{
		return q[getIndex_q(row, column, state)];
	}
	inline __host__ real &getElement_q_cpu(real* cpu_q, int row, int column, int state)
	{
		return cpu_q[getIndex_q(row, column, state)];
	}
	inline __device__ void setElement_q(int row, int column, int state, real setValue)
	{
		q[getIndex_q(row, column, state)] = setValue;
	}
	inline __host__ void setElement_q_cpu(real* cpu_q, int row, int column, int state, real setValue)
	{
		cpu_q[getIndex_q(row, column, state)] = setValue;
	}

	// QNEW
	// Dictates how the memory layout for qNew will be
	inline __device__ int getIndex_qNew(int row, int column, int state)
	{
		// Usual C/C++ row major order
		//return (row*cellsX*numStates + column*numStates + state);
		// state is the slowest moving dimension now, then row, then column
		return (state*cellsX*cellsY + row*cellsX + column);
	}
	inline __device__ real &getElement_qNew(int row, int column, int state)
	{
		return qNew[getIndex_qNew(row, column, state)];
	}
	inline __device__ void setElement_qNew(int row, int column, int state, real setValue)
	{
		qNew[getIndex_qNew(row, column, state)] = setValue;
	}

	// COEFFICIENTS
	inline __device__ __host__ int getIndex_coeff(int row, int column, int coeff)
	{
		// Usual C/C++ row major order
		//return (row*cellsX*numCoeff + column*numCoeff + coeff);
		// coeff is the slowest moving dimension now, then row, then column
		return (coeff*cellsX*cellsY + row*cellsX + column);
	}
	inline __device__ real &getElement_coeff(int row, int column, int coeff)
	{
		return coefficients[getIndex_coeff(row, column, coeff)];
	}
	inline __host__ real &getElement_coeff_cpu(real* cpu_coeff, int row, int column, int coeff)
	{
		return cpu_coeff[getIndex_coeff(row, column, coeff)];
	}
	inline __device__ void setElement_coeff(int row, int column, int coeff, real setValue)
	{
		coefficients[getIndex_coeff(row, column, coeff)] = setValue;
	}
	inline __host__ void setElement_coeff_cpu(real* cpu_coeff, int row, int column, int coeff, real setValue)
	{
		cpu_coeff[getIndex_coeff(row, column, coeff)] = setValue;
	}

	// WAVESPEEDS
	// Dictates how the memory layout for waveSpeeds will be, we might be better off using all first waves first then second waves then third...
	// instead of having the first wave then second then third... of the first cell, then those of the second cell...
	// that is: return (waveNum*cellsX*cellsY + row*cellsX + col);
	inline __device__ int getIndex_waveSpeed(int row, int column, int waveNum)
	{
		// Usual C/C++ row major order
		return (row*cellsX*numWaves + column*numWaves + waveNum);
	}
	inline __device__ real &getElement_waveSpeedX(int row, int column, int waveNum)
	{
		return waveSpeedsX[getIndex_waveSpeed(row, column, waveNum)];
	}
	inline __device__ void setElement_waveSpeedX(int row, int column, int waveNum, real waveSpeed)
	{
		waveSpeedsX[getIndex_waveSpeed(row, column, waveNum)] = waveSpeed;
	}

	inline __device__ real &getElement_waveSpeedY(int row, int column, int waveNum)
	{
		return waveSpeedsY[getIndex_waveSpeed(row, column, waveNum)];
	}
	inline __device__ void setElement_waveSpeedY(int row, int column, int waveNum, real waveSpeed)
	{
		waveSpeedsY[getIndex_waveSpeed(row, column, waveNum)] = waveSpeed;
	}
};

#endif