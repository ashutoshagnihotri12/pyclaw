#!/usr/bin/env python
# encoding: utf-8

import numpy as np

# -------- GLOBAL SCALAR DEFINITIONS --------------

# excitation - initial conditoons
ex_type  = 'plane'
alambda  = 0.1 				# wavelength
t_ex_sig = 1.0*alambda			# width in time (pulse only)
x_ex_sig = 1.0*alambda			# width in the x-direction (pulse)
toff_ex  = 0.0 					# offset in time
xoff_ex	 = 0.02    				# offset in the x-direction
yoff_ex	 = 0.5 					# offset in the y -direction
omega 	 = 2.0*np.pi/alambda	# frequency
k 		 = 2.0*np.pi*alambda
amp_Ex	 = 1.
amp_Ey	 = 1.
amp_Hz	 = 1.


# refractive index n = sqrt(epso*muo)
rip_shape	= 'interlayered'

# set background refractive index 
epsilon_def = 1
epso 		= 1.0
muo			= 1.0

# set moving refractive index parameters
x_vrip_e 	= 0.6
y_vrip_e	= 0.0
x_vrip_m 	= x_vrip_e
y_vrip_m	= y_vrip_e

prip 		= 0.1

xoff_rip_e 	= 0.2
yoff_rip_e	= 0.0
xoff_rip_m 	= xoff_rip_e
yoff_rip_m	= yoff_rip_e
sig_rip_e 	= .1
sig_rip_m	= sig_rip_e
deltan 		= prip*(np.sqrt(epso*muo))
atampe 		= deltan*(2.0*1.5+deltan)
atampu 		= deltan*(2.0*1.5+deltan)

# multilayered definition
N_la 		= 10 			
N_lb 		= N_la -1
t_la		= 0.0015
t_lb		= 0.0050
la_e 		= 2.5
la_m 		= 2.5
lb_e 		= 1.0
lb_m 		= 1.0

# pre-calculations for wave propagation
v = 1/np.sqrt(epso*muo)
vx_ex = v
vy_ex = 0.0
kx_ex = k
ky_ex = 0.0

# Grid - mesh settings
x_lower=0.; x_upper=10.; mx = np.floor(50*(x_upper-x_lower)/alambda)

y_lower=0
if rip_shape=='interlayered':
	y_upper 	= N_la*t_la + N_lb*t_lb
	y_ex_sig 	= 0.95*y_upper
	my 			= np.floor((y_upper-y_lower)*10000)
	mlp 		= (t_la+t_lb)*10000
else:
	y_upper 	= 1
	y_ex_sig  	= x_ex_sig
	my 			= np.floor(50*(y_upper-y_lower)/alambda)


# -------- GLOBAL FUNCTION DEFINITIONS --------------


def refind(t,X,Y):
	"""
	deps = refind(t,x,y)

	This function returns the refractive index map based on general definitions set earlier,
	Gaussian cases support moving RIPs.
	
	x,y are the coordinate of the grid centers state.grid.e_j.centers, e_j = x,y 
	"""
	y,x = np.meshgrid(Y,X)
	deps = np.empty( [2,len(X),len(Y)], order='F')

	if rip_shape=='gaussian2d':
		deps[0,:,:] = atampe*np.exp(-((x-x_vrip_e*t-xoff_rip_e)**2 + (y-y_vrip_e*t-yoff_rip_e)**2)/sig_rip_e**2) + epso
		deps[1,:,:] = atampu*np.exp(-((x-x_vrip_m*t-xoff_rip_m)**2 + (y-y_vrip_m*t-yoff_rip_m)**2)/sig_rip_m**2) + muo
	elif rip_shape=='gaussian1dx':
		deps[0,:,:] = atampe*np.exp(-((x-x_vrip_e*t-xoff_rip_e)**2)/sig_rip_e**2) + epso
		deps[1,:,:] = atampu*np.exp(-((x-x_vrip_m*t-xoff_rip_m)**2)/sig_rip_m**2) + muo
	elif rip_shape=='gaussian1dy':
		deps[0,:,:] = atampe*np.exp(-((y-y_vrip_e*t-yoff_rip_e)**2)/sig_rip_e**2) + epso
		deps[1,:,:] = atampu*np.exp(-((y-y_vrip_m*t-yoff_rip_m)**2)/sig_rip_m**2) + muo	
	elif rip_shape=='homogeneous':
		deps[0,:,:] = epso
		deps[1,:,:] = muo
	elif rip_shape=='interface':
		deps[0,:,:] = 1*(x<x_upper/2) + 4*(x>=x_upper/2)
		deps[1,:,:] = 1*(x<0.45) + 4*(x>=0.45)
	elif rip_shape=='interlayered':
		deps[0,:,:] = lb_e
		deps[1,:,:] = lb_m
		for m in range(0,N_la):
			deps[0,:,m*mlp:m*mlp+(t_la*10000)] = la_e
			deps[1,:,m*mlp:m*mlp+(t_la*10000)] = la_m



	return deps

def update_aux(solver,state):
	grid = state.grid
	y = state.grid.y.centers
	x = state.grid.x.centers
	td = state.t
	oldaux = state.aux.copy(order='F')
	state.aux = setaux(td,x,y)
	state.q = state.q
	
#	next function might be redundant since it already exists as deltan	
def setaux(t,x,y):
	aux = np.empty( [2,len(y),len(x)], order='F')
	aux = refind(t,x,y)
	return aux

def setaux_lower(state,dim,t,auxbc,num_ghost):
	grid = state.grid
	X = grid.x.centers_with_ghost(num_ghost)[:num_ghost]
	Y = grid.y.centers_with_ghost(num_ghost)
	tl = state.t
        auxbc[:,:num_ghost,:] = refind(tl,X,Y)
	return auxbc

def setaux_upper(state,dim,t,auxbc,num_ghost):
	grid = state.grid
	X = grid.x.centers_with_ghost(num_ghost)[-num_ghost:]
	Y = grid.y.centers_with_ghost(num_ghost)
	tu = state.t
        auxbc[:,-num_ghost:,:] = refind(tu,X,Y)
	return auxbc

def scattering_bc(state,dim,t,qbc,num_ghost):
	"""
	EM scattering boundary conditions with three components in  TM-mode Ex, Ey, Hz.
	"""
	grid = state.grid
	X = grid.x.centers_with_ghost(num_ghost)[:num_ghost]
	Y = grid.y.centers_with_ghost(num_ghost)
	ts = state.t
	y,x = np.meshgrid(Y,X)
	aux_left_bc = refind(t,X,Y)
	t0 = 0.05
	pulseshape = np.zeros( [len(X),len(Y)], order='F')
	harmonic = np.zeros( [len(X),len(Y)], order='F')

	if ex_type=='plane':
		pulseshape 	= 1.0
		harmonic 	= np.sin(kx_ex*x + ky_ex*y - omega*ts)
	elif ex_type=='gauss-beam':
		pulseshape 	= np.exp(-(y - yoff_ex)**2/y_ex_sig**2)
		harmonic 	= np.sin(kx_ex*x + ky_ex*y - omega*ts)
	elif ex_type=='gauss_pulse':
		pulseshape 	= np.exp(-(x - xoff_ex - vx_ex*(ts-t0))**2/x_ex_sig**2)*np.exp(-(y - yoff_ex - vy_ex*(ts-t0))**2/y_ex_sig**2)
		harmonic 	= np.sin(kx_ex*x + ky_ex*y - omega*ts)
	elif ex_type=='plane_pulse':
		pulseshape 	= np.exp(-(x - xoff_ex - vx_ex*(ts-t0))**2/x_ex_sig**2)
		harmonic 	= np.sin(kx_ex*x + ky_ex*y - omega*ts)
	elif ex_type=='simple_pulse2D':
		pulseshape 	= np.exp(-(x - xoff_ex - vx_ex*(ts-t0))**2/x_ex_sig**2)*np.exp(-(y - yoff_ex - vy_ex*(ts-t0))**2/y_ex_sig**2)
		harmonic 	= 1.0
	elif ex_type=='simple_pulse2D_x':
		pulseshape 	= np.exp(-(x - xoff_ex - vx_ex*(ts-t0))**2/x_ex_sig**2)
		harmonic 	= 1.0


	qbc[0,:num_ghost,:] = amp_Ex*pulseshape*harmonic*aux_left_bc[0,:,:]
	qbc[1,:num_ghost,:] = amp_Ey*pulseshape*harmonic*aux_left_bc[0,:,:]
	qbc[2,:num_ghost,:] = amp_Hz*pulseshape*harmonic*aux_left_bc[1,:,:]
	return qbc


def qinit(state):
	"""
	Initial conditions in simulation grid for electromagnetic components q
	"""

	grid = state.grid
	X = grid.x.centers
	Y = grid.y.centers
	ts = state.t
	y,x = np.meshgrid(Y,X)
        r2 = (x-1.)**2 + (y-0.5)**2

	state.q[0,:,:] = 0.0 
	state.q[1,:,:] = 0.0
	state.q[2,:,:] = 0.0


# -------- MAIN SCRIPT --------------

def em2D(kernel_language='Fortran',iplot=False,htmlplot=False,use_petsc=True,save_outdir='./_output',solver_type='sharpclaw'):

	if use_petsc:
		import clawpack.petclaw as pyclaw
	else:
		from clawpack import pyclaw

	print v,y_upper,mx,my

#	Solver settings
	if solver_type=='classic':
		solver=pyclaw.ClawSolver2D()
		solver.dimensional_split=False
		solver.limiters = pyclaw.limiters.tvd.MC
	elif solver_type=='sharpclaw':
		solver=pyclaw.SharpClawSolver2D()
		solver.num_waves = 2
		solver.weno_order = 5


#	solver.dt_initial=0.005
#	solver.max_steps = 1000000
	import maxwell_2d
	solver.rp = maxwell_2d


	solver.fwave = True
	solver.cfl_max = 2.45
	solver.cfl_desired = 2.4
#	solver.before_step = update_aux


#	define number of waves (eqn) and aux (eps,mu)
	num_eqn = 3
	num_aux = 2

#	abstract domain and state setup
	x_dime = pyclaw.Dimension('x',x_lower,x_upper,mx)
	y_dime = pyclaw.Dimension('y',y_lower,y_upper,my)
	domain = pyclaw.Domain([x_dime,y_dime])
	state = pyclaw.State(domain,num_eqn,num_aux)
	grid = state.grid
	X = grid.x.centers
	Y = grid.y.centers
	tini = state.t
	state.aux = refind(tini,X,Y)


# 	Boundary conditions
	solver.user_bc_lower = scattering_bc
	solver.bc_lower[0] = pyclaw.BC.custom
	solver.bc_upper[0] = pyclaw.BC.extrap
	solver.bc_lower[1] = pyclaw.BC.extrap
	solver.bc_upper[1] = pyclaw.BC.extrap

	solver.user_aux_bc_lower = setaux_lower
	solver.user_aux_bc_upper = setaux_upper
	solver.aux_bc_lower[0] = pyclaw.BC.custom
	solver.aux_bc_upper[0] = pyclaw.BC.custom
	solver.aux_bc_lower[1] = pyclaw.BC.wall
	solver.aux_bc_upper[1] = pyclaw.BC.wall

#	Initial solution
	qinit(state)


#	controller
	claw = pyclaw.Controller()
	claw.keep_copy = True
	claw.tfinal = 2
	claw.num_output_times = 10
	claw.solver = solver
	claw.solution = pyclaw.Solution(state,domain)
	claw.outdir = save_outdir
	claw.write_aux_always = True

	status = claw.run()

	if htmlplot:  pyclaw.plot.html_plot(outdir=save_outdir,file_format=claw.output_format)
	if iplot:     pyclaw.plot.interactive_plot(outdir=save_outdir,file_format=claw.output_format)

	return claw


if __name__=="__main__":
	import sys
	from clawpack.pyclaw.util import run_app_from_main
	output = run_app_from_main(em2D)
