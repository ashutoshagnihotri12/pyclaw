////#include "Visualizer2D.h" 
////
////// Some getter functions similiar to those in pdeParam struct
////inline __device__ int getIndex_q(int cellsX, int cellsY, int numStates, int row, int column, int state)
////{
////	// Usual C/C++ row major order
////	return (row*cellsX*numStates + column*numStates + state);
////}
////inline __device__ real &getElement_q(real* q, int cellsX, int cellsY, int numStates, int row, int column, int state)
////{
////	return q[getIndex_q(cellsX, cellsY, numStates, row, column, state)];
////}
////// PBO getters and setters and memory access pattern
////inline __device__ int getIndex_PBO(int dispResolutionX, int dispResolutionY, int row, int column)
////{
////	// Usual C/C++ row major order
////	return (row*dispResolutionX + column);
////}
////inline __device__ GLfloat &getElement_PBO(GLfloat* PBO, int dispResolutionX, int dispResolutionY, int row, int column)
////{
////	return PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column)];
////}
////inline __device__ void setDisplayPBO(GLfloat* PBO, int dispResolutionX, int dispResolutionY, int row, int column, GLfloat pixelValue)
////{
////	PBO[getIndex_PBO(dispResolutionX, dispResolutionY, row, column)] = pixelValue;
////}
////
/////*
////float findAbsMaxPerFrame(Float* data, int width, int height, int paddedWidth, int numBoundCellsX, int numBoundCellsY)
////{
////	// we have a choice to include or not the boundary cells in search for the absolute max, of the values
////	// The default will be kept as including the boundary cells.
////	float maximum = -FLT_MAX;
////
////	for( int j = 0; j < height-1; j++)
////	{
////		for( int i = 0; i < width; i++)
////		{
////			if (absoluteValue(data[i + j*paddedWidth]) > maximum)
////			{
////				maximum = absoluteValue(data[i + j*paddedWidth]);
////			}
////		}
////	}
////	printf("\nmax of this frame is: %f.\n\n", maximum);
////	return maximum;
////}*/
/////*
////float setRangePerFrame(Float * data_gpu, int width, int paddedWidth, int height, int numBoundCellsX, int numBoundCellsY)
////{
////	Float * data_cpu = (Float*)malloc(paddedWidth*height*sizeof(Float));
////
////	cudaMemcpy(data_cpu, data_gpu, paddedWidth*height*sizeof(Float), cudaMemcpyDeviceToHost);	// This is inefficient but it is done once.
////
////	float halfRange = findAbsMaxPerFrame(data_cpu, width, height, paddedWidth, numBoundCellsX, numBoundCellsY);
////
////	free(data_cpu);
////
////	if (halfRange == 0)
////		return 0.5;//1.0f;
////	else
////		return halfRange;
////}*/
////
////// This kernel would only work if the resolution is smaller than the computation data
////// Will fix that next.
////// Will need to take care of maybe modes of display, smoothing and whatnot when things arent of matching sizes
////__global__ void copyDisplay_Kernel(GLfloat* PBO, int dispResolutionX, int dispResolutionY,
////									real* q, int cellsX, int cellsY, int numStates, int ghostCells,
////									int state_display, bool boundary_display)
////{
////	int col = blockIdx.y*blockDim.y + threadIdx.y;
////	int row = blockIdx.x*blockDim.x + threadIdx.x;
////
////	if ( col < cellsX && row < cellsY )
////	{
////		if (!boundary_display)
////		{
////			if (col < ghostCells || col >= cellsX - ghostCells)
////				setDisplayPBO(PBO, dispResolutionX, dispResolutionY, row, col, 0.0f);
////			else if (row < ghostCells || row >= cellsY - ghostCells)
////				setDisplayPBO(PBO, dispResolutionX, dispResolutionY, row, col, 0.0f);
////			else
////				setDisplayPBO(PBO, dispResolutionX, dispResolutionY, row, col, getElement_q(q, cellsX, cellsY, numStates, row, col, state_display));
////		}
////		else
////			setDisplayPBO(PBO, dispResolutionX, dispResolutionY, row, col, getElement_q(q, cellsX, cellsY, numStates, row, col, state_display));
////	}
////}
////extern "C" void copyDisplayData(GLfloat* PBO, int dispResolutionX, int dispResolutionY,
////								real* q, int cellsX, int cellsY, int numStates, int ghostCells,
////								int state_display, bool boundary_display)
////{
////	// this function would be more interesting when GPU does not support non power of two textures //?
////
////	// kernel to copy the data from vis.param.qNew to vis.PBO_DISP_data_d
////	unsigned int blockDimensionX = 16;
////	unsigned int blockDimensionY = 16;
////
////	unsigned int gridDimensionX = (dispResolutionX+blockDimensionX-1)/blockDimensionX;
////	unsigned int gridDimensionY = (dispResolutionY+blockDimensionY-1)/blockDimensionY;
////
////	dim3 dimGrid(gridDimensionX, gridDimensionY);
////	dim3 dimBlock(blockDimensionX, blockDimensionY);
////
////	//range = setRangePerFrame(data, width, paddedWidth, height, numBoundCellsX, numBoundCellsY); //?
////
////	copyDisplay_Kernel<<<dimGrid, dimBlock>>>(PBO, dispResolutionX, dispResolutionY,
////												q, cellsX, cellsY, numStates, ghostCells,
////												state_display, boundary_display);
////}
/////*
////__global__ void pulse_Kernel(int X, int Y,
////							 Float * data, int padded_width, int width, int height, int numBoundCellsX, int numBoundCellsY, int state = 0)
////{
////	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
////	int tidy = threadIdx.y + blockIdx.y*blockDim.y;
////
////	// this is quite innefficient in terms of warps being scheduled and divergence,
////	// but seeing that it happens only by user input it can be overlooked
////	if ( (tidx - X)*(tidx - X) + (tidy - Y)*(tidy - Y) < (float)min(width,height)/4.0)
////	{
////		data[state*(padded_width*height) + tidy*(padded_width) + tidx] = 
////			data[state*(padded_width*height) + tidy*(padded_width) + tidx]
////			+ 10.0f*exp( -1.0f*((float)(tidx - X)*(tidx - X)/width + (float)(tidy - Y)*(tidy - Y))/height );
////	}
////}
////*/
/////*
////extern "C" void doImpulse(GLdouble impulseX, GLdouble impulseY, GLdouble impulseZ,
////						  float centerX, float centerY, float centerZ, float canvasRatio,
////						  Float * data, int paddedWidth, int width, int height, int numBoundCellsX, int numBoundCellsY,
////						  int state)
////{
////	// check if impulse coordinate is on canvas, checking only the Z should be enough
////	if ( absoluteValue(impulseZ - centerZ) < 0.001 )
////	{
////		// do whatever ratio thing that has to be done, and do some kind of impulse.
////		float xRatio = (float)(impulseX - centerX) + 0.5f; // the width is fixed at 1, centered around 0, from -0.5 to 0.5
////		float yRatio = ((float)(impulseY - centerY) + 0.5f/canvasRatio)*canvasRatio;
////
////		int approxX = xRatio*width;
////		int approxY = yRatio*height;
////
////		//// reading values at point click
////		//Float clickedValue = 0.0;
////		//cudaMemcpy(&clickedValue, (data+ state*paddedWidth*height + approxY*paddedWidth + approxX), sizeof(Float), cudaMemcpyDeviceToHost);
////		//float clickedValue_float = clickedValue;
////		//
////		//printf("\n************************** Value of clicked point is: %f\n", clickedValue_float);
////
////		if ( xRatio < 0.9 && yRatio < 0.9 && xRatio > 0.1 && yRatio > 0.1 )
////		{
////			// kernel to copy the data from driver->params.gpuQ or driver->params.gpuQNew to PBO_DISP_data_d
////			unsigned int blockDimensionX = 16;
////			unsigned int blockDimensionY = 16;
////
////			unsigned int gridDimensionX = (width+blockDimensionX-1)/blockDimensionX;
////			unsigned int gridDimensionY = (height+blockDimensionY-1)/blockDimensionY;
////
////			dim3 dimGrid(gridDimensionX, gridDimensionY);
////			dim3 dimBlock(blockDimensionX, blockDimensionY);
////
////			pulse_Kernel<<<dimGrid, dimBlock>>>(approxX, approxY, data, paddedWidth, width, height, numBoundCellsX, numBoundCellsY, state);
////		}
////	}
////}
////
////*/