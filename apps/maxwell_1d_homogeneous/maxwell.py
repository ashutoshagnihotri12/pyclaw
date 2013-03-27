#!/usr/bin/env python
# encoding: utf-8

import numpy as np

# -------- GLOBAL SCALAR DEFINITIONS --------------

# excitation - initial conditoons
ex_type  = 'plane'
alambda  = 1 				# wavelength
t_ex_sig = 1.0*alambda			# width in time (pulse only)
x_ex_sig = 1.0*alambda			# width in the x-direction (pulse)
toff_ex  = 0.0 					# offset in time
xoff_ex	 = 0.0    				# offset in the x-direction
omega 	 = 2.0*np.pi/alambda	# frequency
k 		 = 2.0*np.pi*alambda
amp_Ex	 = 1.0
amp_Hy	 = 1.0


# refractive index n = sqrt(epso*muo)
rip_shape	= 'gaussian1dx'
epsilon_def = 1
epso 		= 1.5
muo			= 1.5
x_vrip_e 	= 0.1
x_vrip_m 	= x_vrip_e
prip 		= 0.1
xoff_rip_e 	= 12
xoff_rip_m 	= xoff_rip_e
sig_rip_eps = 10.0
sig_rip_mu	= sig_rip_eps
deltan = prip*(np.sqrt(epso*muo))
atampe = deltan*(2.0*1.5+deltan)
atampu = deltan*(2.0*1.5+deltan)
# atampe = 0.15
# atampu = 0.15

v = 1/np.sqrt(epso*muo)
vx_ex = v
kx_ex = k

# Grid - mesh settings
x_lower=0.; x_upper=500.; mx = np.floor(50*(x_upper-x_lower)/alambda)

# -------- GLOBAL FUNCTION DEFINITIONS --------------

# refractive index map definition function 
def refind(t,x):
	"""
	deps = refind(t,x)

	This function returns the refractive index map based on general definitions set earlier,
	Gaussian cases support moving RIPs.
	
	x are the coordinate of the grid centers state.grid.e_j.centers, e_j = x 
	"""
	deps = np.empty( (2,len(x)), order='F')

	if rip_shape=='gaussian1dx':
		deps[0,:] = atampe*np.exp(-((x-x_vrip_e*t-xoff_rip_e)**2)/sig_rip_eps**2) + epso
		deps[1,:] = atampu*np.exp(-((x-x_vrip_m*t-xoff_rip_m)**2)/sig_rip_mu**2) + muo
	elif rip_shape=='homogeneous':
		deps[0,:] = epso
		deps[1,:] = muo
	elif rip_shape=='interface':
		deps[0,:] = 1*(x<0.45) + 4*(x>=0.45)
		deps[1,:] = 1*(x<0.45) + 4*(x>=0.45)

	return deps

def update_aux(solver,state):
	x = state.grid.x.centers
	t = state.t
	oldaux = state.aux.copy(order='F')
	state.aux = setaux(t,x)
	state.q = state.q*state.aux/oldaux
	return state

#	next function might be redundant since it already exists as deltan	
def setaux(t,x):
	aux = np.empty( (2,len(x)), order='F')
	aux[:,:] = refind(t,x)
	return aux

def setaux_lower(state,dim,t,auxbc,num_ghost):
	x = state.grid.x.centers_with_ghost(num_ghost)[:num_ghost]
	auxbc[:,:num_ghost] = refind(t,x)
	return auxbc

def setaux_upper(state,dim,t,auxbc,num_ghost):
	x = state.grid.x.centers_with_ghost(num_ghost)[-num_ghost:]
	auxbc[:,-num_ghost:] = refind(t,x)
	return auxbc

def scattering_bc(state,dim,t,qbc,num_ghost):
	"""
	EM scattering boundary conditions with three components Ey, Hz.
	"""
	grid = state.grid
	x = grid.x.centers_with_ghost(num_ghost)[:num_ghost]
	ts = state.t
	aux_left_bc = refind(t,x)
	t0 = 0.05
	if ex_type=='plane':
		pulseshape = 1.0
		harmonic = np.sin(kx_ex*x - omega*ts)
	elif ex_type=='gauss_pulse':
		pulseshape = np.exp(-(x - xoff_ex - vx_ex*(ts-t0))**2/x_ex_sig**2)
		harmonic = np.sin(kx_ex*x - omega*ts)
	elif ex_type=='simple_pulse':
		pulseshape = np.exp(-(x - xoff_ex - vx_ex*(ts-t0))**2/x_ex_sig**2)
		harmonic = 1.0
	
	
	qbc[0,:num_ghost] = amp_Ex*pulseshape*harmonic*aux_left_bc[0,:]
	qbc[1,:num_ghost] = amp_Hy*pulseshape*harmonic*aux_left_bc[1,:]

	return qbc


def qinit(state):
	"""
	Initial conditions in simulation grid for electromagnetic components q
	"""
	grid = state.grid
	x = grid.x.centers
	ts = state.t

	state.q[0,:] = 0.0
	state.q[1,:] = 0.0
#	state.p = np.empty( (2,len(x)), order='F')

	return state


def ffields(state):
	"""
	This function calculates the E and H fields, stripped from the  constitutive relations

	"""
	grid = state.grid
	x = grid.x.centers
	ts = state.t
	fields = np.empty( (2,len(x)), order='F')
	oldaux = state.aux.copy(order='F')
	fields[0,:] = state.q[0,:]/oldaux[0,:]
	fields[1,:] = state.q[1,:]/oldaux[1,:]

	return fields



# -------- MAIN SCRIPT --------------

def em1D(kernel_language='Fortran',iplot=False,htmlplot=False,use_petsc=False,save_outdir='./_trapp1',solver_type='sharpclaw',save_p='./_calculations',beforestep=True):

	if use_petsc:
		import clawpack.petclaw as pyclaw
	else:
		from clawpack import pyclaw


#	Solver settings
	if solver_type=='classic':
		solver=pyclaw.ClawSolver1D()
		solver.dimensional_split=False
		solver.limiters = pyclaw.limiters.tvd.MC
	elif solver_type=='sharpclaw':
		solver=pyclaw.SharpClawSolver1D()
		solver.num_waves = 2
		solver.weno_order = 7

	solver.dt_initial=0.005
	solver.max_steps = 1000000

	import maxwell_1d
	solver.rp = maxwell_1d
	solver.fwave = True
	solver.cfl_max = 2.45
	solver.cfl_desired = 2.4

        solver.call_before_step_each_stage = 1

	if beforestep:
		print 'update aux'
		solver.before_step = update_aux

	
#	define number of waves (eqn) and aux (eps,mu)
	num_eqn = 2
	num_aux = 2

#	print mx
	#	abstract domain and state setup
	x_dime = pyclaw.Dimension('x',x_lower,x_upper,mx)
	domain = pyclaw.Domain([x_dime])
	state = pyclaw.State(domain,num_eqn,num_aux)
	state.mp = 2
	grid = state.grid
	x = grid.x.centers
	tini = state.t
	state.aux = refind(tini,x)
	

	# Boundary conditions
	solver.bc_lower[0] = pyclaw.BC.custom
	solver.bc_upper[0] = pyclaw.BC.extrap
	solver.aux_bc_lower[0]=pyclaw.BC.custom
	solver.aux_bc_upper[0]=pyclaw.BC.custom
	solver.user_bc_lower = scattering_bc
	solver.user_aux_bc_lower = setaux_lower
	solver.user_aux_bc_upper = setaux_upper

	#Initial condition
	qinit(state)

	
	#controller
	claw = pyclaw.Controller()
	claw.tfinal = 750
	claw.num_output_times = 200
	claw.solver = solver
	claw.solution = pyclaw.Solution(state,domain)
	claw.write_aux_always = True
	claw.outdir = save_outdir
#	claw.compute_p = ffields
#	claw.outdir_p = save_p
	
	status = claw.run()

	if htmlplot:  pyclaw.plot.html_plot(outdir=save_outdir,file_format=claw.output_format)
	if iplot:     pyclaw.plot.interactive_plot(outdir=save_outdir,file_format=claw.output_format)

	return claw


if __name__=="__main__":
	import sys
	from clawpack.pyclaw.util import run_app_from_main
	output = run_app_from_main(em1D)
