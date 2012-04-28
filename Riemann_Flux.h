//#ifndef __RIEMANN_FLUX_H__
//#define __RIEMANN_FLUX_H__
//
//#include "common.h"
//#include <math.h>
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
//		/*real* amdq, real* apdq,*/ real* wave, real* waveSpeeds);												// output
//};
//struct acoustics_vertical
//{
//	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
//		/*real* amdq, real* apdq,*/ real* wave, real* waveSpeeds);												// output
//};
//
//__device__ real MC(real theta); // just tp illustrate, bad function
//
//struct limiter
//{
//	template<class limiter_phi>
//	__device__ real operator() (real* wave_left, real* wave_right, real* wave_main_left, real* wave_main_right, limiter_phi phi);	// Need to see what else is needed
//};
//
//#endif