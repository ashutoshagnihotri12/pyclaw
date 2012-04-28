#ifndef __BOUNDARY_CONDITIONS_CUH__
#define __BOUNDARY_CONDITIONS_CUH__

#include "common.h"

//template<class L, class R, class U, class D>
//class boundaryConditions
//{
//public:
//	L condition_left;
//	R condition_right;
//	U condition_up;
//	D condition_down;
//};

enum BOUNDARY_CONDITIONS
{
	BC_ABSORBING = 1,
	BC_REFLECTIVE = 2,
	BC_PERIODIC = 3,
};

__global__ void BC_left_Absorbing_Kernel(pdeParam param);
__global__ void BC_right_Absorbing_Kernel(pdeParam param);
__global__ void BC_up_Absorbing_Kernel(pdeParam param);
__global__ void BC_down_Absorbing_Kernel(pdeParam param);

__global__ void BC_left_Reflective_Kernel(pdeParam param);
__global__ void BC_right_Reflective_Kernel(pdeParam param);
__global__ void BC_up_Reflective_Kernel(pdeParam param);
__global__ void BC_down_Reflective_Kernel(pdeParam param);

extern "C" void setBoundaryConditions(pdeParam &param,
										BOUNDARY_CONDITIONS cond_l,
										BOUNDARY_CONDITIONS cond_r,
										BOUNDARY_CONDITIONS cond_u,
										BOUNDARY_CONDITIONS cond_d);

//template<class BCS>
//void setBoundaryConditions(	pdeParam& param,
//											BCS bc);
//
////////////////////////////////////////// ABSORBING CONDITIONS //////////////////////////////////////////////////////////
//
//// The idea behind these boundry condition kernels is to have a set of threads work on the edges
//// For example for the left edge,  threads will be created equal to the amount of rows there are
//// each will handle the boundary cells on its assigned row.
//// ! The number of rows (or columns) includes the boundaries on both sides
//
//struct BC_left_Absorbing
//{
//	__device__ void operator()(pdeParam param)
//	{
//		int row = threadIdx.x + blockDim.x*blockIdx.x;
//
//		int range = param.cellsY;
//		int boundary_start = 0;
//		int boundary_end = param.ghostCells;
//
//		if ( row < range )													// the number of working threads must be equal to the number of existing rows
//			for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
//				for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
//					param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_end, k));
//	}
//};
//struct BC_right_Absorbing
//{
//	__device__ void operator()(pdeParam param)
//	{
//		int row = threadIdx.x + blockDim.x*blockIdx.x;
//
//		int range = param.cellsY;
//		int boundary_start = param.cellsX - param.ghostCells;
//		int boundary_end = param.cellsX;
//
//		if ( row < range )													// the number of working threads must be equal to the number of existing rows
//			for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
//				for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
//					param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_start-1, k));
//	}
//};
//struct BC_up_Absorbing
//{
//	__device__ void operator()(pdeParam param)
//	{
//		int col = threadIdx.x + blockDim.x*blockIdx.x;
//
//		int range = param.cellsX;
//		int boundary_start = param.cellsY - param.ghostCells;
//		int boundary_end = param.cellsY;
//
//		if ( col < range )													// the number of working threads must be equal to the number of existing rows
//			for (int row = boundary_start; row < boundary_end; row++)		// could be modified to be handled by threads in the y directions
//				for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
//					param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_start-1, col, k));
//	}
//};
//struct BC_down_Absorbing
//{
//	__device__ void operator()(pdeParam param)
//	{
//		int col = threadIdx.x + blockDim.x*blockIdx.x;
//
//		int range = param.cellsX;
//		int boundary_start = 0;
//		int boundary_end = param.ghostCells;
//
//		if ( col < range )													// the number of working threads must be equal to the number of existing rows
//			for (int row = boundary_start; row < boundary_end; row++)		// could be modified to be handled by threads in the y directions
//				for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
//					param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_end, col, k));
//	}
//};
//
////////////////////////////////////////// REFLECTIVE CONDITIONS //////////////////////////////////////////////////////////
//
//__global__ void BC_left_Reflective_Kernel(pdeParam param)
//{
//	int row = threadIdx.x + blockDim.x*blockIdx.x;
//
//	int range = param.cellsY;
//	int boundary_start = 0;
//	int boundary_end = param.ghostCells;
//	int boundary_length = boundary_end - boundary_start;
//
//	if ( row < range )													// the number of working threads must be equal to the number of existing rows
//		for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
//			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
//				if (k == 0)
//					param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_end + boundary_length - 1 - col, k));
//				else
//					param.setElement_qNew(row, col, k,-param.getElement_qNew(row, boundary_end + boundary_length - 1 - col, k));
//}
//__global__ void BC_right_Reflective_Kernel(pdeParam param)
//{
//	int row = threadIdx.x + blockDim.x*blockIdx.x;
//
//	int range = param.cellsY;
//	int boundary_start = param.cellsX - param.ghostCells;
//	int boundary_end = param.cellsX;
//
//	if ( row < range )													// the number of working threads must be equal to the number of existing rows
//		for (int col = boundary_start; col < boundary_end; col++)		// could be modified to be handled by threads in the y directions
//			for ( int k = 0; k < param.numStates; k++)					// could be modified to be handled by threads in the z directions
//				if ( k == 0)
//					param.setElement_qNew(row, col, k, param.getElement_qNew(row, boundary_start -(col - boundary_start), k));
//				else
//					param.setElement_qNew(row, col, k,-param.getElement_qNew(row, boundary_start -(col - boundary_start), k));
//}
//__global__ void BC_up_Reflective_Kernel(pdeParam param)
//{
//	int col = threadIdx.x + blockDim.x*blockIdx.x;
//
//	int range = param.cellsX;
//	int boundary_start = param.cellsY - param.ghostCells;
//	int boundary_end = param.cellsY;
//	int reverse_counter = 1;		// replaced the working but undreadable construct (row-boundary_start+1)
//
//	if ( col < range )																		// the number of working threads must be equal to the number of existing rows
//		for (int row = boundary_start; row < boundary_end; row++, reverse_counter++)		// could be modified to be handled by threads in the y directions
//			for ( int k = 0; k < param.numStates; k++)										// could be modified to be handled by threads in the z directions
//				if (k == 0)
//					param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_start - reverse_counter, col, k));
//				else
//					param.setElement_qNew(row, col, k,-param.getElement_qNew(boundary_start - reverse_counter, col, k));
//}
//__global__ void BC_down_Reflective_Kernel(pdeParam param)
//{
//	int col = threadIdx.x + blockDim.x*blockIdx.x;
//
//	int range = param.cellsX;
//	int boundary_start = 0;
//	int boundary_end = param.ghostCells;
//	int reverse_counter = param.ghostCells - 1;// replaced working but hard to read construct: int boundary_length = boundary_end - boundary_start;
//
//	if ( col < range )																		// the number of working threads must be equal to the number of existing rows
//		for (int row = boundary_start; row < boundary_end; row++, reverse_counter--)		// could be modified to be handled by threads in the y directions
//			for ( int k = 0; k < param.numStates; k++)										// could be modified to be handled by threads in the z directions
//				if (k == 0)
//					param.setElement_qNew(row, col, k, param.getElement_qNew(boundary_end + reverse_counter, col, k));
//				else
//					param.setElement_qNew(row, col, k,-param.getElement_qNew(boundary_end + reverse_counter, col, k));
//}

#endif