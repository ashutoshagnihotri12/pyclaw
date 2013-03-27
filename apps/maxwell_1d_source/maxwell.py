#!/usr/bin/env python
# encoding: utf-8
#--------- non-linear em --------------------------

import numpy as np

# -------- GLOBAL SCALAR DEFINITIONS -----------------------------
# ======== all definitions are in m,s,g unit system.
n_frames = 30
x_lower = 0.0e-6
x_upper = 10e-6					# lenght [m]
# ........ material properties ...................................

# vacuum
eo = 8.854187817e-12			# vacuum permittivity   - [F/m]
mo = 4e-7*np.pi 				# vacuum peremeability  - [V.s/A.m]
co = 1/np.sqrt(eo*mo)			# vacuum speed of light - [m/s]
zo = np.sqrt(eo/mo)
# material
mat_shape = 'homogeneous'			# material definition: homogeneous, interface, rip (moving perturbation), multilayered

# background refractive index 
bkg_er = 1.5
bkg_mr = 1.5
bkg_n  = np.sqrt(bkg_er*bkg_mr)
bkg_e  = eo*bkg_er
bkg_m  = mo*bkg_mr

# if interface declare position
x_change = x_upper/2

# set moving refractive index parameters
rip_vx_e 	= 0.0*co	# replace here the value of x
rip_vx_m 	= rip_vx_e

rip_xoff_e 	= 10e-6
rip_xoff_m  = rip_xoff_e

rip_xsig_e 	= 10.0e-6
rip_xsig_m  = rip_xsig_e
s_x_e 		= rip_xsig_e**2
s_x_m 		= rip_xsig_m**2

prip 		= 0.1
deltan 		= prip*(bkg_n) # assumes epsilon = mu
d_e 		= deltan #*(2.0*1.5+deltan)
d_m 		= deltan #*(2.0*1.5+deltan)

# set multilayer parameters

# multilayered definition
n_layers = 2
layers = np.zeros([n_layers,7]) # _layer:  eps mu N t chi2e chi2m chi3e chi3m
layers[0,0] = 1.5
layers[0,1] = 1.5
layers[0,2] = 10
layers[0,3] = 15e-9
layers[1,0] = 2.5
layers[1,1] = 2.5
layers[1,2] = layers[0,2] - 1
layers[1,3] = 50e-9
N_layers = 5
if mat_shape=='multilayer':
	x_upper = N_layers*np.sum(layers[:,3])+layers[0,3]
	tlp = np.sum(layers[:,3])
	mlp = np.floor(tlp/1e-9)

# set non-linear parameters of the material
chi2_e		= 0.0
chi3_e		= 0.0
chi2_m 		= 0.0
chi3_m 		= 0.0

# ........ excitation - initial conditoons .......................
ex_type  = 'plane'
alambda  = 1e-6				# wavelength
ex_t_sig = 1.0*alambda			# width in time (pulse only)
ex_x_sig = 1.0*alambda			# width in the x-direction (pulse)
ex_toff  = 0.0 					# offset in time
ex_xoff	 = 0.0 	    			# offset in the x-direction
omega 	 = 2.0*np.pi*co/alambda	# frequency
k 		 = 2.0*np.pi/alambda
amp_Ey	 = 1.
amp_Hz	 = 1.

# ........ pre-calculations for wave propagation .................
v_r = 1./bkg_n
v = co*v_r
ex_vx = v
ex_kx = k

# Grid - mesh settings
if mat_shape=='multilayer':
	mx = np.floor((x_upper-x_lower)/1e-9)
else:
	mx = np.floor(60*(x_upper-x_lower)/alambda)

ddx = (x_upper-x_lower)/mx
ddt = 0.90/(co*np.sqrt(1.0/(ddx**2)))
max_steps = 250000
t_final = (x_upper-x_lower)/v
# -------- GLOBAL FUNCTION DEFINITIONS --------------

# refractive index map definition function 
def etar(t,x):
	"""
	eta = etar(t,x)

	This function returns the refractive index map based on general definitions set earlier,
	Gaussian cases support moving RIPs.
	
	x are the coordinate of the grid centers state.grid.e_j.centers, e_j = x 
         aux holds:
         0: epsilon
         1: mu
         2: epsilon_t
         3: mu_t
	"""
	
	eta = np.empty( (4,len(x)), order='F')

	if mat_shape=='moving_gauss':
		u_x_e = x - rip_vx_e*t - rip_xoff_e
		u_x_m = x - rip_vx_m*t - rip_xoff_m
		u_e = (u_x_e/rip_xsig_e)**2
		u_m = (u_x_m/rip_xsig_m)**2
		u_e_t = 2*((rip_vx_e*u_x_e)/(rip_xsig_e**2))
		u_m_t = 2*((rip_vx_m*u_x_m)/(rip_xsig_m**2))
		eta[0,:] = d_e*np.exp(-u_e) + bkg_er
		eta[1,:] = d_m*np.exp(-u_m) + bkg_mr
		eta[2,:] = u_e_t*d_e*np.exp(-u_e)
		eta[3,:] = u_m_t*d_m*np.exp(-u_m)
	elif mat_shape=='gaussian':
		u_x_e = x - rip_xoff_e
		u_x_m = x - rip_xoff_m
		u_e = (u_x_e/rip_xsig_e)**2
		u_m = (u_x_m/rip_xsig_m)**2
		eta[0,:] = d_e*np.exp(-u_e) + bkg_er
		eta[1,:] = d_m*np.exp(-u_m) + bkg_mr	
		eta[2,:] = 0.
		eta[3,:] = 0.
	elif mat_shape=='homogeneous':
		eta[0,:] = bkg_er
		eta[1,:] = bkg_mr
		eta[2,:] = 0.
		eta[3,:] = 0.
	elif mat_shape=='interface':
		eta[0,:] = 1*(x<x_change) + 4*(x>=x_change)
		eta[1,:] = 1*(x<x_change) + 4*(x>=x_change)
		eta[2,:] = 0.
		eta[3,:] = 0.
	elif mat_shape=='multilayer':
		for n in range(0,N_layers):
			xi = n*tlp
			for m in range(0,n_layers):
				if m==0:
					eta[0,:] = layers[m,0]*(xi<x)*(x<=xi+layers[m,3])
					eta[1,:] = layers[m,1]*(xi<x)*(x<=xi+layers[m,3])
				else:
					eta[0,:] = layers[m,0]*(xi+layers[m-1,3]<x)*(x<=xi+layers[m,3])
					eta[1,:] = layers[m,1]*(xi+layers[m-1,3]<x)*(x<=xi+layers[m,3])


		eta[0,:] = layers[0,0]*(N_layers*tlp<x)*(x<=N_layers*tlp+layers[0,3])
		eta[1,:] = layers[0,1]*(N_layers*tlp<x)*(x<=N_layers*tlp+layers[0,3])
		eta[2,:] = 0.0
		eta[3,:] = 0.0	


	return eta

def update_aux(solver,state):
	x = state.grid.x.centers
	t = state.t
#	oldaux = state.aux.copy(order='F')
	state.aux = setaux(t,x)
#   state.q = state.q*state.aux[0:2,:]/oldaux[0:2,:]
	return state

#	next function might be redundant since it already exists as deltan	
def setaux(t,x):
	aux = np.empty( (4,len(x)), order='F')
	aux[:,:] = etar(t,x)
	return aux

def setaux_lower(state,dim,t,auxbc,num_ghost):
	x = state.grid.x.centers_with_ghost(num_ghost)[:num_ghost]
	auxbc[:,:num_ghost] = etar(t,x)
	return auxbc

def setaux_upper(state,dim,t,auxbc,num_ghost):
	x = state.grid.x.centers_with_ghost(num_ghost)[-num_ghost:]
	auxbc[:,-num_ghost:] = etar(t,x)
	return auxbc

def scattering_bc(state,dim,t,qbc,num_ghost):
	"""
	EM scattering boundary conditions with three components Ey, Hz.
	"""
	grid = state.grid
	x = grid.x.centers_with_ghost(num_ghost)[:num_ghost]
	ts = state.t
	t0 = 0.05

	if ex_type=='plane':
		pulseshape = 1.0
		harmonic = np.sin(ex_kx*x - omega*ts)
	elif ex_type=='gauss_pulse':
		pulseshape = np.exp(-(x - ex_xoff - ex_vx*(ts-t0))**2/ex_x_sig**2)
		harmonic = np.sin(ex_kx*x - omega*ts)
	elif ex_type=='simple_pulse':
		pulseshape = np.exp(-(x - ex_xoff - ex_vx*(ts-t0))**2/ex_x_sig**2)
		harmonic = 1.0
	elif ex_type=='off':
		pulseshape = 0.
		harmonic = 0.	
	
	qbc[0,:num_ghost] = amp_Ey*pulseshape*harmonic
	qbc[1,:num_ghost] = amp_Hz*pulseshape*harmonic

	return qbc


def qinit(state):
	"""
	Initial conditions in simulation grid for electromagnetic components q
	"""
	
	if ex_type=='off':
		grid = state.grid
		x = grid.x.centers
		dd = x_upper-x_lower
		state.q[0,:] = 0.
		state.q[1,:] = np.exp(-(x-dd/2)**2/(dd/10)**2)
	else:
		state.q[0,:] = 0.0
		state.q[1,:] = 0.0

#	state.p = np.empty( (2,len(x)), order='F')
	
	return state

def psir(solver,state,dt):
	"""
	This function calculates psi(x,y,t)
	"""
	grid 	= state.grid
	x 	 	= grid.x.centers_with_ghost(num_ghost)
	ts   	= state.t
	etartim = etar(ts-dt/2,x)
	etarti 	= etar(ts+dt/2,x)
	etart 	= (etarti - etartim) / dt

	return etart

def kappa(solver,state,dt):
	"""
	This function calculates the capacity function kappa
	"""
	grid = state.grid
	x = grid.x.centers_with_ghost(num_ghost)
	ts = state.t
	eta = etar(ts,x)
	nlfields = np.empty( (2,len(x)), order='F')
	kap = np.empty( (2,len(x)), order='F')
	nlfields[0,:] = 2*chi2*state.q[0,:] + 3*chi3*state.q[0,:]**2
	nlfields[0,:] = 2*chi2m*state.q[0,:] + 3*chi3m*state.q[1,:]**2

	kap[0,:] = eta[0,:] + nlfields[0,:]
	kap[1,:] = eta[1,:] + nlfields[1,:]

	return kap

# -------- MAIN SCRIPT --------------

def em1D(kernel_language='Fortran',iplot=False,htmlplot=False,use_petsc=False,save_outdir='./_trap',solver_type='sharpclaw',save_p='./_calculations',before_step=False):

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
		solver.weno_order = 5

	solver.dt_initial = ddt
	solver.max_steps = max_steps
	
	import maxwell_1d_nl
	solver.rp = maxwell_1d_nl
	solver.fwave = True
	solver.cfl_max = 0.45
	solver.cfl_desired = 0.4
	print 'setup information:'
	print 'v_wave=',v
	print 'x_lim=',x_upper,' t_f=',t_final 
	print 'mx=',mx,'dx=',ddx,'dt=',ddt,'N_max=',max_steps
	print 'lambda=',alambda,'freq=',omega
	if before_step:
		print 'update aux'
		solver.call_before_step_each_stage = 1
		solver.before_step = update_aux
	
#	define number of waves (eqn) and aux (eps,mu)
	num_eqn = 2
	num_aux = 4

#	print mx
	#	abstract domain and state setup
	x_dime = pyclaw.Dimension('x',x_lower,x_upper,mx)
	domain = pyclaw.Domain([x_dime])
	state = pyclaw.State(domain,num_eqn,num_aux)
	state.mp = 2
	grid = state.grid
	x = grid.x.centers
	tini = state.t
	state.aux = etar(tini,x)
	
	state.problem_data['dx'] = x_dime.delta
	state.problem_data['chi2_e_r'] = chi2_e
	state.problem_data['chi3_e_r'] = chi3_e
	state.problem_data['chi2_m_r'] = chi2_m
	state.problem_data['chi3_m_r'] = chi3_m
	state.problem_data['eo'] = eo
	state.problem_data['mo'] = mo
	state.problem_data['co'] = co
	state.problem_data['zo'] = zo

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
	claw.tfinal = t_final
	claw.num_output_times = n_frames
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
