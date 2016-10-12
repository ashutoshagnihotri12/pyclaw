#!/usr/bin/env python
# encoding: utf-8
    
def acoustics(use_petsc=False,kernel_language='Fortran',solver_type='classic',iplot=False,htmlplot=False,outdir='./_output',weno_order=5):
    """
    This example solves the 1-dimensional acoustics equations in a homogeneous
    medium.
    """
    import numpy as np
    from clawpack import riemann

    #=================================================================
    # Import the appropriate classes, depending on the options passed
    #=================================================================
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    riemann_solver = riemann.acoustics_1D

    if solver_type=='classic':
        solver = pyclaw.ClawSolver1D(riemann_solver)
    elif solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(riemann_solver)
        solver.weno_order=weno_order
        solver.time_integrator = 'RK'
        from nodepy import rk
        import numpy as np
        #ssp3 = rkm.SSPRK3(4).__num__()
        rkm = rk.loadRKM('DP5').__num__()
        solver.A = rkm.A
        solver.b = rkm.b
        #solver.b_hat = np.array([1./3,1./3,1./3,0.])
        solver.b_hat = rkm.bhat
        solver.c = rkm.c
        solver.error_tolerance = 1.e-2
        solver.dt_variable = True
        solver.cfl_max = 10.
        solver.cfl_desired = 1.5
        solver.dt_initial = 1.e-3
    else: raise Exception('Unrecognized value of solver_type.')

    #========================================================================
    # Instantiate the solver and define the system of equations to be solved
    #========================================================================
    solver.kernel_language=kernel_language
 
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall

    #solver.cfl_desired = 1.0
    #solver.cfl_max     = 1.0

    #========================================================================
    # Instantiate the grid and set the boundary conditions
    #========================================================================
    x = pyclaw.Dimension('x',0.0,1.0,200)
    domain = pyclaw.Domain(x)
    num_eqn = 2
    state = pyclaw.State(domain,num_eqn)

    #========================================================================
    # Set problem-specific variables
    #========================================================================
    rho = 1.0
    bulk = 1.0
    state.problem_data['rho']=rho
    state.problem_data['bulk']=bulk
    state.problem_data['zz']=np.sqrt(rho*bulk)
    state.problem_data['cc']=np.sqrt(bulk/rho)

    #========================================================================
    # Set the initial condition
    #========================================================================
    xc = domain.grid.x.centers
    state.q[0,:] = np.cos(2*np.pi*xc)
    state.q[1,:] = 0.
    solver.dt_initial=domain.grid.delta[0]/state.problem_data['cc']*0.01

    #========================================================================
    # Set up the controller object
    #========================================================================
    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.num_output_times = 40
    claw.tfinal = 2.

    # Solve
    status = claw.run()

    # Plot results
    #if htmlplot:  pyclaw.plot.html_plot(outdir=outdir)
    #if iplot:     pyclaw.plot.interactive_plot(outdir=outdir)

    return claw

if __name__=="__main__":
    from setplot import setplot
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(acoustics,setplot)
