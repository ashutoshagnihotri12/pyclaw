#include "cudaClaw_main.h"

#include <iostream>

#include "test.h"

// Defined falgs to switch between GPUs for debug or run
#define GPU_RELEASE 0
#define GPU_DEBUG 1

template<class Vis>
Vis* GlutInterface<Vis>::visualizer = NULL;

// This object holds the solver and limiter, and must be generated
typedef Visualizer2D<acoustics_horizontal, acoustics_vertical, limiter_MC, 
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_reflective>> acoustics_vis;

int main(int argc, char** argv)
{
	setupCUDA();

	// Boundary setup
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_reflective> conditions;

	BC_left_reflective left;
	BC_right_reflective right;
	BC_up_reflective up;
	BC_down_reflective down;

	conditions.condition_left = left;
	conditions.condition_right = right;
	conditions.condition_up = up;
	conditions.condition_down = down;

	// Solver setup
	acoustics_horizontal acoustic_h;
	acoustics_vertical   acoustic_v;

	// Limiter setup
	limiter_MC phi;

	GlutInterface<acoustics_vis>::InitGlut(argc, argv, 512, 512, CELLSX, CELLSY);	// the last 2 arguments are the resolution of the texture, may be changed inside the method

	pdeParam problemParam = setupAcoustics(circle_q, uniform_coefficients);

	GlutInterface<acoustics_vis>::visualizer->setParam(problemParam);
	GlutInterface<acoustics_vis>::visualizer->setBoundaryConditions(conditions);
	GlutInterface<acoustics_vis>::visualizer->setSolvers(acoustic_h, acoustic_v);
	GlutInterface<acoustics_vis>::visualizer->setLimiter(phi);
	GlutInterface<acoustics_vis>::visualizer->initializeDisplay();
	GlutInterface<acoustics_vis>::visualizer->launchDisplay();

	// We never reach this step as the game loop is launched before it
	gracefulExit(problemParam);
}

void mid_Gaussian_q(pdeParam &param)
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);

	real center_r = param.cellsX/2;
	real center_c = param.cellsY/2;
	real x;
	real y;

	for (int row = 0; row < param.cellsY; row++)
		for (int col = 0; col < param.cellsX; col++)
			for (int state = 0; state < param.numStates; state++)
			{
				if (state == 0)
				{
					x = param.width*(row-center_r)/(real)param.cellsY;
					y = param.height*(col-center_c)/(real)param.cellsX;
					param.setElement_q_cpu(cpu_q, row, col, state, exp( -((x*x)+(y*y))/0.1 )  );
				}
				else
				{
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0 );
				}
			}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}
void circle_q(pdeParam &param)
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);
	
	real pi = 3.14159;
	real w = 0.2;	// multiplier that determines the span/largeness of the bell curve

	real center_r = param.cellsX/2;
	real center_c = param.cellsY/2;
	real x;
	real y;
	real r;

	for (int row = 0; row < param.cellsY; row++)
		for (int col = 0; col < param.cellsX; col++)
			for (int state = 0; state < param.numStates; state++)
			{
				if (state == 0)
				{
					x = param.width*(row-center_r)/(real)param.cellsY;
					y = param.height*(col-center_c)/(real)param.cellsX;
					r = sqrt(x*x + y*y);
					if ( abs(r-0.25) <= w )
						param.setElement_q_cpu(cpu_q, row, col, state, (1 + cos( pi*(r-0.25)/w))/4.0f);//exp( -((r-0.25)*(r-0.25))/0.02 )  );
					else
						param.setElement_q_cpu(cpu_q, row, col, state, 0.0);
				}
				else
				{
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0);
				}
			}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}
void uniform_coefficients(pdeParam &param, real* u)
{
	size_t size = param.cellsX*param.cellsY*param.numCoeff*sizeof(real);
	real* cpu_coeff = (real*)malloc(size);

	for (int row = 0; row < param.cellsY; row++)
	{
		for (int col = 0; col < param.cellsX; col++)
		{
			for (int coeff = 0; coeff < param.numCoeff; coeff++)
			{
				param.setElement_coeff_cpu(cpu_coeff, row, col, coeff, u[coeff]);
			}
		}
	}

	cudaError_t memCpyCheck = cudaMemcpy(param.coefficients, cpu_coeff, size, cudaMemcpyHostToDevice);
	free(cpu_coeff);
}
pdeParam setupAcoustics(void (*init_q)(pdeParam &), void (*init_coeff)(pdeParam &,real*) )
{
	pdeParam param(
		CELLSX,	//cellsX
		CELLSY,	//cellsY
		2,		//ghostCells
		3,		//numStates
		2,		//numWaves
		2,		//numCoeff
		0,		//startX
		1,		//endX
		0,		//startY
		1);		//endY

	real u[2] = {1.0, 4.0};
	init_q(param);
	init_coeff(param, u);

	return param;
}
void setupCUDA()
{
	int device = GPU_DEBUG;	//1 for debug 0 for run, chooses the gpu

	// cudaSetDevice and cudaGLSetGLDevice do not make contexts
	// if both choose the same device, the cuda runtime functions
	// will not work properly, so only one of the setter functions
	// must be called, and so cudaGLSetGLDevice chooses the first CUDA device
	cudaError_t errorDevice = cudaSetDevice(device);
	cudaError_t errorGLdevice = cudaGLSetGLDevice(device);

	cudaDeviceProp device_property;
	cudaGetDeviceProperties(&device_property, device);

	// Some error when choosing cache configuration, could be with the order of the call, 
	if (device_property.major < 2)
		// cache-shared memory configuring not possible, no cache
		printf("Cache configuration not possible\n");
	else
	{
		//cudaError_t errorCachePref1 = cudaFuncSetCacheConfig("fused_Riemann_limiter_horizontal_update_kernel", cudaFuncCachePreferShared);
		//cudaError_t errorCachePref2 = cudaFuncSetCacheConfig("fused_Riemann_limiter_vertical_update_kernel", cudaFuncCachePreferShared);
		//printf("Cache configuration done, config1: %i, config2: %i\n",errorCachePref1,errorCachePref2);
	}

}
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute)
{
	// Credit to Nvidia GPU computing SDK, deviceQuery project.
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if( CUDA_SUCCESS != error)
	{
		printf("cuSafeCallNoSync() Driver API error = %04d\n", error);
        exit(-1);
    }
}
void gracefulExit(pdeParam &param)
{
	delete(GlutInterface<acoustics_vis>::visualizer);
	cudaThreadExit();
	exit(0);
}