#ifndef __BOUNDARY_CONDITIONS_H__ 
#define __BOUNDARY_CONDITIONS_H__

#include "boundary_conditions_header.h"

template<class BC>
__global__ void boundary_kernel(pdeParam param, BC condition)
{
	condition(param);
}

template<class BCS>
void setBoundaryConditions(pdeParam& param, BCS bcs)
{
	unsigned int blockDimensionX = 256; 

	int maxSide = (param.cellsX >= param.cellsY)? param.cellsX:param.cellsY;
	unsigned int gridDimensionX = (maxSide+blockDimensionX-1)/blockDimensionX;

	dim3 dimGrid(gridDimensionX);
	dim3 dimBlock(blockDimensionX);

	boundary_kernel<<<dimGrid, dimBlock>>>(param, bcs.condition_left);
	boundary_kernel<<<dimGrid, dimBlock>>>(param, bcs.condition_right);
	boundary_kernel<<<dimGrid, dimBlock>>>(param, bcs.condition_up);
	boundary_kernel<<<dimGrid, dimBlock>>>(param, bcs.condition_down);
}

#endif