//#include "Riemann_Flux.h"
//
//__device__ void acoustics(int setOfArgs)
//{
//	int doesAfewThings = setOfArgs;
//}
//// As a general rule a Riemann solver must have the following arguments:
//// 2 cells, one left one right (equivalently up and down)
//// 2 sets of coefficients one for the left cell the other for the right
//// a vector amdq
//// a vector apdq
//// a set of waves
////
//// These data will come from both the global and shared memory, this will be decided
//// according to the needs of the limiter. It is certain that amdq and apdq and the waves
//// will be stored in shared memory, however we might need the speeds of the waves and 
//// perhaps the coefficients and even the data itself. The shared memory on 2.x architectures
//// is common with the hardware of the L1 cache, we could potentially leave some of
//// the required data there without control, and control the essentials with from the shared memory
////
//// For the acoustics example, one would need:
//// blockSize * ( q + (amdq apdq) + coeff + waves + speeds ) * sizeof(real) ... computed per cell/thread
//// =
//// 16*16 * (3 + (3 3) + 2 + 2*3 + 1)* (4 or 8) = 18Kbytes or 36Kbytes => 16/48 or 8/48 warps
////
//// Eliminating the need to store at least the coefficients and data q, we can go up to full 48/48
//// warp usage with 16*8 blocks. Remains to see what is needed for the limiters.
////
//// A slightly independent note: The pointers that the Riemann solver will have can point to either global or
//// shared memory. The exact location with offset will be resolved before the kernel calls the device function
//// (the Riemann solver). The writer of the Riemann device function (the user) must be careful how he/she
//// fills the data in amdq, apdq, wave speeds and the waves themselves. A certain convention must be maintained
//// with the layout of the main data q, mainly [state1 state2 state3] [state1 state2 state3] ...
//
//
//struct acoustics_horizontal
//{
//	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
//								/*real* amdq, real* apdq,*/ real* wave, real* waveSpeeds)						// output
//	{
//		float rho_l = u_left[0];
//		float bulk_l = u_left[1];
//
//		float rho_r = u_right[0];
//		float bulk_r = u_right[1];
//
//		float c_l = sqrt(bulk_l/rho_l);	// sound speed
//		float z_l = c_l*rho_l;			// impedance
//
//		float c_r = sqrt(bulk_r/rho_r);
//		float z_r = c_r*rho_r;
//
//		waveSpeeds[0] = c_l;
//		waveSpeeds[1] = c_r;
//
//		float alpha1 = ( q_left[0] - q_right[0] + z_r*(q_right[1] - q_left[1])) / (z_l+z_r);	// ( -(pr-pl) + zr(vr-vl) )/ (zl+zr)
//		float alpha2 = ( q_right[0] - q_left[0] + z_l*(q_right[1] - q_left[1])) / (z_l+z_r);	// (  (pr-pl) + zl(vr-vl) )/ (zl+zr)
//
//		wave[0 + 0*numStates] = -alpha1*z_l;
//		wave[1 + 0*numStates] = alpha1;
//		wave[2 + 0*numStates] = 0;
//
//		wave[0 + 1*numStates] = alpha2*z_r;
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
//struct acoustics_vertical
//{
//	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
//								/*real* amdq, real* apdq,*/ real* wave, real* waveSpeeds)						// output
//	{
//
//
//		float rho_l = u_left[0];
//		float bulk_l = u_left[1];
//
//		float rho_r = u_right[0];
//		float bulk_r = u_right[1];
//
//		float c_l = sqrt(bulk_l/rho_l);	// sound speed
//		float z_l = c_l*rho_l;			// impedance
//
//		float c_r = sqrt(bulk_r/rho_r);
//		float z_r = c_r*rho_r;
//
//		waveSpeeds[0] = c_l;
//		waveSpeeds[1] = c_r;
//
//		float alpha1 = ( q_left[0] - q_right[0] + z_r*(q_right[2] - q_left[2])) / (z_l+z_r);	// ( -(pr-pl) + zr(vr-vl) )/ (zl+zr)
//		float alpha2 = ( q_right[0] - q_left[0] + z_l*(q_right[2] - q_left[2])) / (z_l+z_r);	// (  (pr-pl) + zl(vr-vl) )/ (zl+zr)
//
//		wave[0 + 0*numStates] = -alpha1*z_l;
//		wave[1 + 0*numStates] = 0;
//		wave[2 + 0*numStates] = alpha1;
//
//		wave[0 + 1*numStates] = alpha2*z_r;
//		wave[1 + 1*numStates] = 0;
//		wave[2 + 1*numStates] = alpha2;
//
//		//amdq[0] = -c_l * wave[0 + 0*numStates];
//		//amdq[1] =  0;							// 0   * wave[1 + 0*numStates];
//		//amdq[2] = -c_l * wave[2 + 0*numStates];
//
//		//apdq[0] = c_r * wave[0 + 1*numStates];
//		//apdq[1] = 0;							// 0   * wave[1 + 1*numStates];
//		//apdq[2] = c_r * wave[2 + 1*numStates];
//	}
//};
//__device__ real MC(real theta) // just tp illustrate, bad function
//{
//	return (real)2*theta + theta/(real)2;
//}
//struct limiter
//{
//	template<class limiter_phi>
//	__device__ real operator() (real* wave_left, real* wave_right, real* wave_main_left, real* wave_main_right, limiter_phi phi)	// Need to see what else is needed
//	{
//		// do more stuff
//	}
//};