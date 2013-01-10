#define VIS
#include "cudaClaw_main.h"

#include <iostream>

// Defined falgs to switch between GPUs for debug or run
#define GPU_RELEASE 0
#define GPU_DEBUG 1


// Problem Objects
template<class Vis>
Vis* GlutInterface<Vis>::visualizer = NULL;

// This object holds the solver and limiter, and must be generated
typedef Visualizer2D<acoustics_horizontal, acoustics_vertical, limiter_MC, 
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_reflective> > acoustics_vis;
	//boundaryConditions<BC_left_absorbing, BC_right_absorbing, BC_up_absorbing, BC_down_absorbing> > acoustics_vis;

typedef Visualizer2D<shallow_water_horizontal, shallow_water_vertical, limiter_MC, 
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_absorbing> > shallowWater_vis;

int main(int argc, char** argv)
{
	setupCUDA();

	// Boundary setup
	boundaryConditions<BC_left_reflective, BC_right_reflective, BC_up_reflective, BC_down_absorbing> reflective_conditions;

	BC_left_reflective left;
	BC_right_reflective right;
	BC_up_reflective up;
	BC_down_absorbing down;

	//boundaryConditions<BC_left_absorbing, BC_right_absorbing, BC_up_absorbing, BC_down_absorbing> absorbing_conditions;
	//BC_left_absorbing left;
	//BC_right_absorbing right;
	//BC_up_absorbing up;
	//BC_down_absorbing down;

	reflective_conditions.condition_left = left;
	reflective_conditions.condition_right = right;
	reflective_conditions.condition_up = up;
	reflective_conditions.condition_down = down;

	// Solver setup
	// acoustics
	acoustics_horizontal acoustic_h;
	acoustics_vertical   acoustic_v;
	// shallow water
	shallow_water_horizontal shallow_water_h;
	shallow_water_vertical   shallow_water_v;

	// Limiter setup
	limiter_MC phi;
	limiter_superbee phi1;

	//GlutInterface<acoustics_vis>::InitGlut(argc, argv, 512, 512, CELLSX, CELLSY);	// the last 2 arguments are the resolution of the texture, may be changed inside the method
	GlutInterface<shallowWater_vis>::InitGlut(argc, argv, 512, 512, CELLSX, CELLSY);	// the last 2 arguments are the resolution of the texture, may be changed inside the method

	real ratio = (real)CELLSY/(real)CELLSX;

	real simulation_start = 0.0f;
	real simulation_end = 1.0f;

	real snapshotRate = 0;

	pdeParam problemParam = setupShallowWater(-1, 1, -1, ratio, simulation_start, simulation_end, snapshotRate, radial_plateau);

	GlutInterface<shallowWater_vis>::visualizer->setParam(problemParam);
	GlutInterface<shallowWater_vis>::visualizer->setBoundaryConditions(reflective_conditions);
	GlutInterface<shallowWater_vis>::visualizer->setSolvers(shallow_water_h, shallow_water_v);
	GlutInterface<shallowWater_vis>::visualizer->setLimiter(phi);
	GlutInterface<shallowWater_vis>::visualizer->initializeDisplay();
	GlutInterface<shallowWater_vis>::visualizer->launchDisplay();

	//pdeParam problemParam = setupAcoustics(0,1,0,ratio, /*off_circle_q/*/centered_circle_q/**/, uniform_coefficients);

	//GlutInterface<acoustics_vis>::visualizer->setParam(problemParam);
	//GlutInterface<acoustics_vis>::visualizer->setBoundaryConditions(reflective_conditions);
	//GlutInterface<acoustics_vis>::visualizer->setSolvers(acoustic_h, acoustic_v);
	//GlutInterface<acoustics_vis>::visualizer->setLimiter(phi);
	//GlutInterface<acoustics_vis>::visualizer->initializeDisplay();
	//GlutInterface<acoustics_vis>::visualizer->launchDisplay();
	
	// We never reach this step as the game loop is launched before it
	gracefulExit();
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
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
							 int device)
{
	// Credit to Nvidia GPU computing SDK, deviceQuery project.
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if( CUDA_SUCCESS != error)
	{
		printf("cuSafeCallNoSync() Driver API error = %04d\n", error);
        exit(-1);
    }
}
void gracefulExit()
{
	delete(GlutInterface<acoustics_vis>::visualizer);
	cudaThreadExit();
	exit(0);
}