#ifndef __CUDACLAW_MAIN_H__
#define __CUDACLAW_MAIN_H__

#include "common.h"
#include <math.h>
#include "Visualizer2D.h"
#include "GlutInterface.h"
#include "step.h"

// State initial value functions
void mid_Gaussian_q(pdeParam &);
void circle_q(pdeParam &param);

// Coefficient value functions
void uniform_coefficients(pdeParam &, real* u);

pdeParam setupAcoustics(void(*init_q_function)(pdeParam &), void(*init_coeff_function)(pdeParam &, real* u));

void setupCUDA();

template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute);

void gracefulExit(pdeParam &);

#endif