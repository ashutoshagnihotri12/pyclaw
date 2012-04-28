#ifndef __STEP_H__
#define __STEP_H__

#include "boundary_conditions.h"
#include "fused_Riemann_Limiter.h"

template <class Riemann_h, class Riemann_v, class Limiter, class BCS>
void step	(pdeParam &param,						// Problem parameters
			 Riemann_h Riemann_pointwise_solver_h,	// Riemann problem solver for vertical interfaces, device function
			 Riemann_v Riemann_pointwise_solver_v,	// Riemann problem solver for horizontal interfaces device function
			 Limiter limiter_phi,					// limiter device function called from a fused Riemann-Limiter function,
													// takes the Riemann solver as parameter along other parameters (waves, and others?)
			 BCS boundary_conditions)				// Boundary conditions put in one object

{
	static real absolute_fastest_speed = 2.0;

	real dt = 0.99/(absolute_fastest_speed/param.dx + absolute_fastest_speed/param.dy); /*/ param.dx/5;	//	awesome failing effects!!/*/

	setBoundaryConditions(param, boundary_conditions);
	
	limited_Riemann_Update(param, dt, Riemann_pointwise_solver_h, Riemann_pointwise_solver_v, limiter_phi);

	cudaMemcpy(&absolute_fastest_speed, param.waveSpeeds, sizeof(real), cudaMemcpyDeviceToHost);
}

#endif