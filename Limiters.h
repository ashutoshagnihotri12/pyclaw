#include "common.h"
#include <cmath>


__device__ real none(real theta) // just tp illustrate, bad function
{
	return 0.0;
}
__device__ real LaxWendroff(real theta) // just tp illustrate, bad function
{
	return 1.0;
}
__device__ real MC(real theta) // just tp illustrate, bad function
{
	real minimum = min(2.0, 2.0*theta);
	return max(0, min((1+theta)/2.0, minimum));
}
__device__ real superbee(real theta) // just tp illustrate, bad function
{
	real maximum = max(0.0, min(1.0, 2*theta));
	return max(maximum, min(2,theta));
}
__device__ real VanLeer(real theta) // just tp illustrate, bad function
{
	real absTheta = sqrt(theta*theta);
	return (theta + absTheta) / (1 + absTheta);
}