#ifndef __PROBLEM_SETUP_H__
#define __PROBLEM_SETUP_H__

#define NUMSTATES 3
#define NUMWAVES 3
#define NUMCOEFF 1

#include <math.h>

#include "common.h"

// State initial value functions
void mid_Gaussian_q(pdeParam &);
void circle_q(pdeParam &param);
void centered_circle_q(pdeParam &param);
void off_circle_q(pdeParam &param);

void radial_plateau(pdeParam &);
void separating_streams(pdeParam &);
void dam_break(pdeParam &param);

// Coefficient value functions
void uniform_coefficients(pdeParam &, real* u);
void some_function_coefficients(pdeParam &);

// Problem Setup
pdeParam setupAcoustics(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void(*init_q_function)(pdeParam &), void(*init_coeff_function)(pdeParam &, real* u));
pdeParam setupAcoustics(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void(*init_q_function)(pdeParam &), void(*init_coeff_function)(pdeParam &));

pdeParam setupShallowWater(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void(*init_q_function)(pdeParam &));

void setUpProblem(int argc, char** argv);

#endif