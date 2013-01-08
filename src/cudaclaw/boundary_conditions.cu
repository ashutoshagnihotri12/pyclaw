#include "boundary_condtions.cuh"

//////////////////////////////////////// ABSORBING CONDITIONS //////////////////////////////////////////////////////////

// The idea behind these boundry condition kernels is to have a set of threads work on the edges
// For example for the left edge,  threads will be created equal to the amount of rows there are
// each will handle the boundary cells on its assigned row.
// ! The number of rows (or columns) includes the boundaries on both sides
__global__ void BC_left_Absorbing_Kernel(pdeParam param)
{
	int row = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsY;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;

	if ( row < range )													// the number of working threads must be equal to the number of existing rows
		for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
				param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_end, k));
}
__global__ void BC_right_Absorbing_Kernel(pdeParam param)
{
	int row = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsY;
	int boundary_start = param.cellsX - param.ghostCells;
	int boundary_end = param.cellsX;

	if ( row < range )													// the number of working threads must be equal to the number of existing rows
		for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
				param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_start-1, k));
}
__global__ void BC_up_Absorbing_Kernel(pdeParam param)
{
	int col = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsX;
	int boundary_start = param.cellsY - param.ghostCells;
	int boundary_end = param.cellsY;

	if ( col < range )													// the number of working threads must be equal to the number of existing rows
		for (int row = boundary_start; row < boundary_end; row++)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
				param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_start-1, col, k));
}
__global__ void BC_down_Absorbing_Kernel(pdeParam param)
{
	int col = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsX;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;

	if ( col < range )													// the number of working threads must be equal to the number of existing rows
		for (int row = boundary_start; row < boundary_end; row++)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
				param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_end, col, k));
}


//////////////////////////////////////// REFLECTIVE CONDITIONS //////////////////////////////////////////////////////////

__global__ void BC_left_Reflective_Kernel(pdeParam param)
{
	int row = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsY;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;
	int boundary_length = boundary_end - boundary_start;

	if ( row < range )													// the number of working threads must be equal to the number of existing rows
		for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
				if (k == 0)
					param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_end + boundary_length - 1 - col, k));
				else
					param.setElement_qNew(row, col, k,-param.getElement_qNew(row, boundary_end + boundary_length - 1 - col, k));
}
__global__ void BC_right_Reflective_Kernel(pdeParam param)
{
	int row = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsY;
	int boundary_start = param.cellsX - param.ghostCells;
	int boundary_end = param.cellsX;

	if ( row < range )													// the number of working threads must be equal to the number of existing rows
		for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
				if ( k == 0)
					param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_start -(col - boundary_start), k));
				else
					param.setElement_qNew(row, col, k,-param.getElement_qNew(row, boundary_start -(col - boundary_start), k));
}
__global__ void BC_up_Reflective_Kernel(pdeParam param)
{
	int col = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsX;
	int boundary_start = param.cellsY - param.ghostCells;
	int boundary_end = param.cellsY;
	int reverse_counter = 1;		// replaced the working but undreadable construct (row-boundary_start+1)

	if ( col < range )																		// the number of working threads must be equal to the number of existing rows
		for (int row = boundary_start; row < boundary_end; row++, reverse_counter++)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)										// could be modified to be handled by threads in the z directions
				if (k == 0)
					param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_start - reverse_counter, col, k));
				else
					param.setElement_qNew(row, col, k,-param.getElement_qNew(boundary_start - reverse_counter, col, k));
}
__global__ void BC_down_Reflective_Kernel(pdeParam param)
{
	int col = threadIdx.x + blockDim.x*blockIdx.x;

	int range = param.cellsX;
	int boundary_start = 0;
	int boundary_end = param.ghostCells;
	int reverse_counter = param.ghostCells - 1;// replaced working but hard to read construct: int boundary_length = boundary_end - boundary_start;

	if ( col < range )																		// the number of working threads must be equal to the number of existing rows
		for (int row = boundary_start; row < boundary_end; row++, reverse_counter--)		// could be modified to be handled by threads in the y directions
			for ( int k = 0; k < param.numStates; k++)										// could be modified to be handled by threads in the z directions
				if (k == 0)
					param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_end + reverse_counter, col, k));
				else
					param.setElement_qNew(row, col, k,-param.getElement_qNew(boundary_end + reverse_counter, col, k));
}

// might want to pass the struct as is and not a pointer,
// which might be too big, so we might have to make a 
// new parameter type for boundary conditions.
// the main method takes the pdeParam, and makes a new
// paramter for boundaries, which is smaller and pass
// that to the boundary condtion functions
extern "C" void setBoundaryConditions(pdeParam &param,
										BOUNDARY_CONDITIONS cond_l,
										BOUNDARY_CONDITIONS cond_r,
										BOUNDARY_CONDITIONS cond_u,
										BOUNDARY_CONDITIONS cond_d)
{
	unsigned int blockDimensionX = 256; 

	int maxSide = (param.cellsX >= param.cellsY)? param.cellsX:param.cellsY;
	unsigned int gridDimensionX = (maxSide+blockDimensionX-1)/blockDimensionX;

	dim3 dimGrid(gridDimensionX);
	dim3 dimBlock(blockDimensionX);

	switch (cond_l)
	{
		case BC_ABSORBING:
			BC_left_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		case BC_REFLECTIVE:
			BC_left_Reflective_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		default:
			BC_left_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
	}
	switch (cond_r)
	{
		case BC_ABSORBING:
			BC_right_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		case BC_REFLECTIVE:
			BC_right_Reflective_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		default:
			BC_right_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
	}
	switch (cond_u)
	{
		case BC_ABSORBING:
			BC_up_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		case BC_REFLECTIVE:
			BC_up_Reflective_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		default:
			BC_up_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
	}
	switch (cond_d)
	{
		case BC_ABSORBING:
			BC_down_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		case BC_REFLECTIVE:
			BC_down_Reflective_Kernel<<<dimGrid, dimBlock>>>(param);
			break;
		default:
			BC_down_Absorbing_Kernel<<<dimGrid, dimBlock>>>(param);
	}
}

//template<class BC>
//__global__ void boundary_kernel(pdeParam param, BC condition)
//{
//	condition(pdeParam);
//}
//
//template<class BCS>
//void setBoundaryConditions(	pdeParam& param,
//										BCS bc)
//{
//	unsigned int blockDimensionX = 256; 
//
//	int maxSide = (param.cellsX >= param.cellsY)? param.cellsX:param.cellsY;
//	unsigned int gridDimensionX = (maxSide+blockDimensionX-1)/blockDimensionX;
//
//	dim3 dimGrid(gridDimensionX);
//	dim3 dimBlock(blockDimensionX);
//
//	boundary_kernel<<<dimGrid, dimBlock>>>(param, bc.condition_left);
//	boundary_kernel<<<dimGrid, dimBlock>>>(param, bc.condition_right);
//	boundary_kernel<<<dimGrid, dimBlock>>>(param, bc.condition_up);
//	boundary_kernel<<<dimGrid, dimBlock>>>(param, bc.condition_down);
//}