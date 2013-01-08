#include "problem_setup.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  State Settings  //////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void radial_plateau(pdeParam &param)	// For Shallow Water equations specifically
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);

	real plateau_radius = 0.2;
	real x;
	real y;

	for (int row = 0; row < param.cellsY; row++)
	{
		y = (param.height*row)/(real)param.cellsY + param.startY;
		for (int col = 0; col < param.cellsX; col++)
		{
			x = (param.width*col)/(real)param.cellsX + param.startX;
			for (int state = 0; state < param.numStates; state++)
			{
				if (state == 0)
				{
					if (x*x + y*y < plateau_radius*plateau_radius)
						param.setElement_q_cpu(cpu_q, row, col, state, 0.8f);
					else
						param.setElement_q_cpu(cpu_q, row, col, state, 0.5f);
				}
				else
				{
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0f);
				}
			}
		}
	}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}

void separating_streams(pdeParam &param)	// For Shallow Water equations specifically
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);

	real x;
	real y;

	for (int row = 0; row < param.cellsY; row++)
	{
		y = (param.height*row)/(real)param.cellsY + param.startY;
		for (int col = 0; col < param.cellsX; col++)
		{
			x = (param.width*col)/(real)param.cellsX + param.startX;
			param.setElement_q_cpu(cpu_q, row, col, 0, 0.6f);
			for (int state = 1; state < param.numStates; state++)
			{
				if (state == 1)
				{
					if ( x < (param.startX + param.endX)/2.0f )
						param.setElement_q_cpu(cpu_q, row, col, state, -0.2f);
					else
						param.setElement_q_cpu(cpu_q, row, col, state, 0.2f);
				}
				else
				{
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0f );
				}
			}
		}
	}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}

void dam_break(pdeParam &param)	// For Shallow Water equations specifically
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);

	real x;
	real y;

	for (int row = 0; row < param.cellsY; row++)
	{
		y = (param.height*row)/(real)param.cellsY + param.startY;
		for (int col = 0; col < param.cellsX; col++)
		{
			x = (param.width*col)/(real)param.cellsX + param.startX;
			for (int state = 0; state < param.numStates; state++)
			{
				if (state == 0)
				{
					if ( x < (param.startX+param.endX)/2.0f )
						param.setElement_q_cpu(cpu_q, row, col, state, 0.7f);
					else
						param.setElement_q_cpu(cpu_q, row, col, state, 0.4f);
				}
				else
				{
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0f );
				}
			}
		}
	}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}

void mid_Gaussian_q(pdeParam &param)
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);

	real x;
	real y;

	for (int row = 0; row < param.cellsY; row++)
	{
		y = (param.height*row)/(real)param.cellsY + param.startY;
		for (int col = 0; col < param.cellsX; col++)
		{
			x = (param.width*col)/(real)param.cellsX + param.startX;
			for (int state = 0; state < param.numStates; state++)
			{
				if (state == 0)
				{
					param.setElement_q_cpu(cpu_q, row, col, state, exp( -((x*x)+(y*y))/0.1 )  );
				}
				else
				{
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0 );
				}
			}
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

	real x;
	real y;
	real r;

	for (int row = 0; row < param.cellsY; row++)
	{
		y = (param.height*row)/(real)param.cellsY + param.startY;
		for (int col = 0; col < param.cellsX; col++)
		{
			x = (param.width*col)/(real)param.cellsX + param.startX;
			for (int state = 0; state < param.numStates; state++)
			{
				if (state == 0)
				{
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
		}
	}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}

void centered_circle_q(pdeParam &param)
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);
	
	real pi = 3.14159;
	real w = 0.2;	// multiplier that determines the span/largeness of the bell curve

	real center_h = param.width/2.0;
	real center_v = param.height/2.0;
	real incrX = param.width/param.cellsX;
	real incrY = param.height/param.cellsY;
	real x;
	real y = param.startY - center_v;
	real r;

	for (int row = 0; row < param.cellsY; row++)
	{
		x = param.startX - center_h;
		for (int col = 0; col < param.cellsX; col++)
		{
			x += incrX;
			r = sqrt(x*x + y*y);
			for (int state = 0; state < param.numStates; state++)
				if (state == 0)
					if ( abs(r-0.25) <= w )
						param.setElement_q_cpu(cpu_q, row, col, state, (1 + cos( pi*(r-0.25)/w))/4.0f);//exp( -((r-0.25)*(r-0.25))/0.02 )  );
					else
						param.setElement_q_cpu(cpu_q, row, col, state, 0.0);
				else
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0);
		}
		y += incrY;
	}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}

void off_circle_q(pdeParam &param)
{
	size_t size = param.cellsX*param.cellsY*param.numStates*sizeof(real);
	real* cpu_q = (real*)malloc(size);
	
	real pi = 3.14159;
	real w = 0.05f;		// multiplier that determines the width of the bell curve
	real span = 0.2f;	// multiplier that determines the span/radius of the bell curve
	real amplitude = 1.2f;

	real center_h = param.width*0.33;
	real center_v = param.height/2.0;
	real incrX = param.width/param.cellsX;
	real incrY = param.height/param.cellsY;
	real x;
	real y = param.startY - center_v;
	real r;

	for (int row = 0; row < param.cellsY; row++)
	{
		x = param.startX - center_h;
		for (int col = 0; col < param.cellsX; col++)
		{
			x += incrX;
			r = sqrt(x*x + y*y);
			for (int state = 0; state < param.numStates; state++)
				if (state == 0)
					if ( abs(r-span) <= w )
						param.setElement_q_cpu(cpu_q, row, col, state, amplitude*(1 + cos( pi*(r-span)/w))/4.0f);
					else
						param.setElement_q_cpu(cpu_q, row, col, state, 0.0);
				else
					param.setElement_q_cpu(cpu_q, row, col, state, 0.0);
		}
		y += incrY;
	}

	cudaError_t memCpyCheck = cudaMemcpy(param.qNew, cpu_q, size, cudaMemcpyHostToDevice);
	free(cpu_q);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////  Coefficient Settings  //////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

void some_function_coefficients(pdeParam &param)
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////  Problem Setting  /////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pdeParam setupAcoustics(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void (*init_q)(pdeParam &), void (*init_coeff)(pdeParam &,real*) )
{
	pdeParam param(
		CELLSX,		//cellsX
		CELLSY,		//cellsY
		2,			//ghostCells
		NUMSTATES,	//numStates
		NUMWAVES,	//numWaves
		NUMCOEFF,	//numCoeff
		x_start,	//startX
		x_end,		//endX
		y_start,	//startY
		y_end,		//endY
		time_start,	//startTime
		time_end	//endTime
		);
	
	if (snapshotRate != 0.0f)
		param.setSnapshotRate(snapshotRate);

	real u[2] = {1.0, 4.0};
	init_q(param);
	init_coeff(param, u);

	return param;
}

pdeParam setupAcoustics(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void (*init_q)(pdeParam &), void (*init_coeff)(pdeParam &) )
{
	pdeParam param(
		CELLSX,		//cellsX
		CELLSY,		//cellsY
		2,			//ghostCells
		NUMSTATES,	//numStates
		NUMWAVES,	//numWaves
		NUMCOEFF,	//numCoeff
		x_start,	//startX
		x_end,		//endX
		y_start,	//startY
		y_end,		//endY
		time_start,	//startTime
		time_end	//endTime
		);
	
	if (snapshotRate != 0.0f)
		param.setSnapshotRate(snapshotRate);

	init_q(param);
	init_coeff(param);

	return param;
}


pdeParam setupShallowWater(real x_start, real x_end, real y_start, real y_end, real time_start, real time_end, real snapshotRate, void(*init_q)(pdeParam &))
{
	pdeParam param(
		CELLSX,		//cellsX
		CELLSY,		//cellsY
		2,			//ghostCells
		NUMSTATES,	//numStates
		NUMWAVES,	//numWaves
		NUMCOEFF,	//numCoeff
		x_start,	//startX
		x_end,		//endX
		y_start,	//startY
		y_end,		//endY
		time_start,	//startTime
		time_end	//endTime
		);
	
	if (snapshotRate != 0.0f)
		param.setSnapshotRate(snapshotRate);

	init_q(param);

	return param;
}