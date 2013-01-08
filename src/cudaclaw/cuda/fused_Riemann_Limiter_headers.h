#ifndef __FUSED_RIEMANN_LIMITER_HEADERS_H__
#define __FUSED_RIEMANN_LIMITER_HEADERS_H__

#define EPSILON (real)0.00000001f	// flying constant, however this is our epsilon

// The Waves and Wave Speeds lie in the shared memory.
// The concern at this stage is not coalescing and alignment (not as it would be in global)
// but bank conflicts, different schemes are likely to yield different performances
// Note however that the Riemann solver depends on the distribution of this data,
// and assumes to have wave1[state1, state2, state3] wave2[state1, state2, state3]
// If this is to remain we must keep the fastest changing components and fiddle
// only with the slower changing ones.
// So, in the case of the waves, the wave number and states must remain as they are,
// and in the case of the wave speeds the wave number must remain.
// Alternatively we must provide a function to the user to set his/her own waves
// such that it would be compatible to the way the framework lays down the memory.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////      Waves    //////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ int getIndex_sharedWaves(int row, int col, int waveNum, int state, int numStates, int numWaves, int blockWidth)
{
	return (row*numStates*numWaves*blockWidth + col*numStates*numWaves + waveNum*numStates + state);
}
inline __device__ real &getSharedWave(real* sharedWaves, int row, int col, int waveNum, int state, int numStates, int numWaves, int blockWidth)
{
	return sharedWaves[getIndex_sharedWaves(row, col, waveNum, state, numStates, numWaves, blockWidth)];
}
inline __device__ void setSharedWave(real* sharedWaves, int row, int col, int waveNum, int state, int numStates, int numWaves, int blockWidth, real newVal)
{
	sharedWaves[getIndex_sharedWaves(row, col, waveNum, state, numStates, numWaves, blockWidth)] = newVal;
}


inline __device__ int getIndex_sharedWaves_flat1(int row, int col, int waveNum, int state, int numStates, int blockSize, int blockWidth)// state 1 of waves1, state 2 of waves1, state 3 of waves1, state 1 of waves2, ...
{
	return (waveNum*blockSize*numStates + state*blockSize + row*blockWidth + col);
}
inline __device__ real &getSharedWave_flat1(real* sharedWaves, int row, int col, int waveNum, int state, int numStates, int blockSize, int blockWidth)
{
	return sharedWaves[getIndex_sharedWaves_flat1(row, col, waveNum, state, numStates, blockSize, blockWidth)];
}
inline __device__ void setSharedWave_flat1(real* sharedWaves, int row, int col, int waveNum, int state, int numStates, int blockSize, int blockWidth, real newVal)
{
	sharedWaves[getIndex_sharedWaves_flat1(row, col, waveNum, state, numStates, blockSize, blockWidth)] = newVal;
}


inline __device__ int getIndex_sharedWaves_flat2(int row, int col, int waveNum, int state, int numWaves, int blockSize, int blockWidth)// state 1 of waves1, state 1 of waves2, state 1 of waves3, state 2 of waves1, ...
{
	return (state*blockSize*numWaves + waveNum*blockSize + row*blockWidth + col);
}
inline __device__ real &getSharedWave_flat2(real* sharedWaves, int row, int col, int waveNum, int state, int numWaves, int blockSize, int blockWidth)
{
	return sharedWaves[getIndex_sharedWaves_flat2(row, col, waveNum, state, numWaves, blockSize, blockWidth)];
}
inline __device__ void setSharedWave_flat2(real* sharedWaves, int row, int col, int waveNum, int state, int numWaves, int blockSize, int blockWidth, real newVal)
{
	sharedWaves[getIndex_sharedWaves_flat2(row, col, waveNum, state, numWaves, blockSize, blockWidth)] = newVal;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////  Wave Speeds  //////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ int getIndex_waveSpeed(int row, int col, int waveNum, int numWaves, int blockWidth)
{
	return (row*numWaves*blockWidth + col*numWaves + waveNum);
}
inline __device__ real &getWaveSpeed(real* waveSpeeds, int row, int col, int waveNum, int numWaves, int blockWidth)
{
	return waveSpeeds[getIndex_waveSpeed(row, col, waveNum, numWaves, blockWidth)];
}
inline __device__ void setWaveSpeed(real* waveSpeeds, int row, int col, int waveNum, int numWaves, int blockWidth, real newSpeed)
{
	waveSpeeds[getIndex_waveSpeed(row, col, waveNum, numWaves, blockWidth)] = newSpeed;
}

inline __device__ int getIndex_waveSpeed_flat(int row, int col, int waveNum, int blockSize, int blockWidth)
{
	return (waveNum*blockSize + row*blockWidth + col);
}
inline __device__ real &getWaveSpeed_flat(real* waveSpeeds, int row, int col, int waveNum, int blockSize, int blockWidth)
{
	return waveSpeeds[getIndex_waveSpeed_flat(row, col, waveNum, blockSize, blockWidth)];
}
inline __device__ void setWaveSpeed_flat(real* waveSpeeds, int row, int col, int waveNum, int blockSize, int blockWidth, real newSpeed)
{
	waveSpeeds[getIndex_waveSpeed_flat(row, col, waveNum, blockSize, blockWidth)] = newSpeed;
}

// As a general rule a Riemann solver must have the following arguments:
// Input:
// - 2 cells, one left one right (equivalently up and down)
// - 2 sets of coefficients one for the left cell the other for the right
// - the number of states (however the user writing the solver would know about this, as well as the number of coefficients)
// Output:
// - a location for storing the set of waves
// - a location for storing the set of wave speeds
//
// The input will come from the global and output will be to shared memory.
//
// A slightly independent note: The Riemann solver's pointer inputs will be take as contiguous arrays.
// That is, eveything to be passed to the solver must first be put in an array format, regardless of
// the global memory distribution. For example, if the global memory distribution is per cell, then
// a pointer to the cell can be used, otherwise, if the memory distribution is per state, as is the
// situation at the moment, we must gather the cell data into an array and pass a pointer to this new
// array as argument.
// The shared memory objects, like the waves and their speeds, are also thought of as arrays in shared.
// Changing this is not advisable, as the current setting works quite well with no bank conflicts (for
// acoustics at least). In case a change must be done, as function must be provided to the user to have
// correct read and write access to the waves, to be compatible with the frameworks view of these objects
// in shared.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////   Riemann Solvers   ////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct acoustics_horizontal
{
	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
								real* wave, real* waveSpeeds)												// output
	{
		real rho_l = u_left[0];
		real bulk_l = u_left[1];

		real rho_r = u_right[0];
		real bulk_r = u_right[1];

		real c_l = sqrt(bulk_l/rho_l);	// sound speed
		real z_l = c_l*rho_l;			// impedance

		real c_r = sqrt(bulk_r/rho_r);
		real z_r = c_r*rho_r;

		waveSpeeds[0] = -c_l;
		waveSpeeds[1] = c_r;

		real alpha1 = ( q_left[0] - q_right[0] + z_r*(q_right[1] - q_left[1])) / (z_l+z_r);	// ( -(pr-pl) + zr(vr-vl) )/ (zl+zr)
		real alpha2 = ( q_right[0] - q_left[0] + z_l*(q_right[1] - q_left[1])) / (z_l+z_r);	// (  (pr-pl) + zl(vr-vl) )/ (zl+zr)

		wave[0 + 0*numStates] = -alpha1*z_l;
		wave[1 + 0*numStates] = alpha1;
		wave[2 + 0*numStates] = 0.0f;

		wave[0 + 1*numStates] = alpha2*z_r;
		wave[1 + 1*numStates] = alpha2;
		wave[2 + 1*numStates] = 0.0f;
	}
};
struct acoustics_vertical
{
	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
								real* wave, real* waveSpeeds)												// output
	{
		real rho_l = u_left[0];
		real bulk_l = u_left[1];

		real rho_r = u_right[0];
		real bulk_r = u_right[1];

		real c_l = sqrt(bulk_l/rho_l);	// sound speed
		real z_l = c_l*rho_l;			// impedance

		real c_r = sqrt(bulk_r/rho_r);
		real z_r = c_r*rho_r;

		waveSpeeds[0] = -c_l;
		waveSpeeds[1] = c_r;

		real alpha1 = ( q_left[0] - q_right[0] + z_r*(q_right[2] - q_left[2])) / (z_l+z_r);	// ( -(pr-pl) + zr(vr-vl) )/ (zl+zr)
		real alpha2 = ( q_right[0] - q_left[0] + z_l*(q_right[2] - q_left[2])) / (z_l+z_r);	// (  (pr-pl) + zl(vr-vl) )/ (zl+zr)

		wave[0 + 0*numStates] = -alpha1*z_l;
		wave[1 + 0*numStates] = 0.0f;
		wave[2 + 0*numStates] = alpha1;

		wave[0 + 1*numStates] = alpha2*z_r;
		wave[1 + 1*numStates] = 0.0f;
		wave[2 + 1*numStates] = alpha2;
	}
};

struct shallow_water_horizontal
{
	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
								real* wave, real* waveSpeeds)												// output
	{
		// try rearanging, or using ul and ur, vl and vr variables instead of dividing every time!
		real g = 9.8;

		real h_left   = q_left[0];
		real hu_left  = q_left[1];
		real hv_left  = q_left[2];
		
		real h_right  = q_right[0];
		real hu_right = q_right[1];
		real hv_right = q_right[2];

		real h_bar = 0.5f*(h_left + h_right);
		real sqrt_h_left  = sqrt(h_left);
		real sqrt_h_right = sqrt(h_right);

		//real sum_sqrt_hleft_hright = sqrt_h_left + sqrt_h_right;

		real u_hat = ((hu_left/sqrt_h_left)+(hu_right/sqrt_h_right))/(sqrt_h_left + sqrt_h_right);
		real v_hat = ((hv_left/sqrt_h_left)+(hv_right/sqrt_h_right))/(sqrt_h_left + sqrt_h_right);

		//real a_left  = h_left*u_hat - hu_left;
		//real a_right = hu_right - h_right*u_hat;
		//real v_hat   = (a_left*(hv_left/h_left) + a_right*(hv_right/h_right)) / (a_left+a_right);

		real c_hat = sqrt(g*h_bar);
		
		waveSpeeds[0] = u_hat - c_hat;
		waveSpeeds[1] = u_hat;
		waveSpeeds[2] = u_hat + c_hat;

		real alpha1 = 0.5f*((u_hat + c_hat)*(h_right - h_left) - (hu_right - hu_left))/c_hat;
		real alpha2 = (hv_right-hv_left)-v_hat*(h_right - h_left);
		real alpha3 = 0.5f*((c_hat - u_hat)*(h_right - h_left) + (hu_right - hu_left))/c_hat;


		wave[0 + 0*numStates] = alpha1;
		wave[1 + 0*numStates] = alpha1*(u_hat - c_hat);
		wave[2 + 0*numStates] = alpha1*v_hat;

		wave[0 + 1*numStates] = 0.0f;
		wave[1 + 1*numStates] = 0.0f;
		wave[2 + 1*numStates] = alpha2;

		wave[0 + 2*numStates] = alpha3;
		wave[1 + 2*numStates] = alpha3*(u_hat + c_hat);
		wave[2 + 2*numStates] = alpha3*v_hat;
	}
};
struct shallow_water_vertical
{
	__device__ void operator() (real* q_left, real* q_right, int numStates, real* u_left, real* u_right,	// input
								real* wave, real* waveSpeeds)												// output
	{
		// try rearanging, or using ul and ur, vl and vr variables instead of dividing every time!
		real g = 9.8;

		real h_left   = q_left[0];
		real hu_left  = q_left[1];
		real hv_left  = q_left[2];
		
		real h_right  = q_right[0];
		real hu_right = q_right[1];
		real hv_right = q_right[2];

		real h_bar = 0.5f*(h_left + h_right);
		real sqrt_h_left  = sqrt(h_left);
		real sqrt_h_right = sqrt(h_right);

		//real sum_sqrt_hleft_hright = sqrt_h_left + sqrt_h_right;

		//real u_hat = ((hu_left/sqrt_h_left)+(hu_right/sqrt_h_right))/(sqrt_h_left + sqrt_h_right);
		//real v_hat = ((hv_left/sqrt_h_left)+(hv_right/sqrt_h_right))/(sqrt_h_left + sqrt_h_right);
		//real a_left  = h_left*u_hat - hu_left;
		//real a_right = hu_right - h_right*u_hat;
		//real v_hat   = (a_left*(hv_left/h_left) + a_right*(hv_right/h_right)) / (a_left+a_right);

		real u_hat = (hu_left/sqrt_h_left + hu_right/sqrt_h_right)/(sqrt_h_left + sqrt_h_right);
		real v_hat = (hv_left/sqrt_h_left + hv_right/sqrt_h_right)/(sqrt_h_left + sqrt_h_right);


		real c_hat = sqrt(g*h_bar);
		
		waveSpeeds[0] = v_hat - c_hat;
		waveSpeeds[1] = v_hat;
		waveSpeeds[2] = v_hat + c_hat;

		real alpha1 = 0.5f*((v_hat + c_hat)*(h_right - h_left) - (hv_right - hv_left))/c_hat;
		real alpha2 = -(hu_right-hu_left) + u_hat*(h_right - h_left);
		real alpha3 = 0.5f*((c_hat - v_hat)*(h_right - h_left) + (hv_right - hv_left))/c_hat;


		wave[0 + 0*numStates] = alpha1;
		wave[1 + 0*numStates] = alpha1*u_hat;
		wave[2 + 0*numStates] = alpha1*(v_hat - c_hat);

		wave[0 + 1*numStates] = 0.0f;
		wave[1 + 1*numStates] = -alpha2;
		wave[2 + 1*numStates] = 0.0f;

		wave[0 + 2*numStates] = alpha3;
		wave[1 + 2*numStates] = alpha3*u_hat;
		wave[2 + 2*numStates] = alpha3*(v_hat + c_hat);
	}
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////   Limiters   //////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int numStates, class Limiter>
__device__ real limiting (Limiter phi, real* main_wave, real* aux_wave)
{
	real main_wave_norm_square = main_wave[0]*main_wave[0];
	real aux_wave_dot_main_wave = aux_wave[0]*main_wave[0];
	#pragma unroll
	for (int i = 1; i < numStates; i++)
	{
		main_wave_norm_square += main_wave[i]*main_wave[i];
		aux_wave_dot_main_wave += aux_wave[i]*main_wave[i];
	}

	if (main_wave_norm_square < EPSILON)
		return (real)1.0f;
	return phi(aux_wave_dot_main_wave/main_wave_norm_square);
}

struct limiter_none
{
	__device__ real operator() (real theta)
	{
		return (real)0.0f;
	}
};
struct limiter_LaxWendroff
{
	__device__ real operator() (real theta)
	{
		return (real)1.0f;
	}
};
struct limiter_MC
{
	__device__ real operator() (real theta)
	{
		real minimum = fmin((real)2.0f, (real)2.0f*theta);
		return fmax((real)0.0f, fmin(((real)1.0f+theta)/(real)2.0f, minimum));
	}
};
struct limiter_superbee
{
	__device__ real operator() (real theta)
	{
		real maximum = fmax((real)0.0f, fmin((real)1.0f,(real)2.0f*theta));
		return fmax(maximum, fmin((real)2.0f,theta));
	}
};
struct limiter_VanLeer
{
	__device__ real operator() (real theta)
	{
		real absTheta = fabs(theta);
		return (theta + absTheta) / ((real)1.0f + absTheta);
	}
};

#endif