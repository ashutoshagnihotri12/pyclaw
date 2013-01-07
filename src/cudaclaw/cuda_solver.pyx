# distutils: language = c++

"""
cuda_solver.pyx

Cythonized bindings to CUDA-enabled hyperbolic solvers for PyClaw

"""

cdef extern from "params.h" :
    # this is hard-coded intentionally, for now
    ctypedef double real 
    cdef cppclass pdeParam:
        pdeParam(int, int, int, int, int, int,
                 double, double, double, double) except +
import cython
import sys

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# external CUDA interfaces
cdef extern int shallow_water_solver_allocate(pdeParam* params)
cdef extern int shallow_water_solver_setup (int bc_left, 
                                            int bc_right, 
                                            int bc_up, 
                                            int bc_down, 
                                            int limiter)

cdef extern int hyperbolic_solver_2d_step (double dt, double* dt) 
cdef extern int hyperbolic_solver_2d_get_qbc (double* qbc)
cdef extern int hyperbolic_solver_2d_set_qbc (double* qbc)

def check_err(err):

    if (not err):
        return 
    else:
        caller = sys._getframe(1).f_code.co_name
        raise Exception(
            "Error code %d returned from %s" % err, caller)

class CUDAState(clawpack.pyclaw.State):
    r"""  See the corresponding PyClaw State documentation."""

    cdef np.ndarray q

    @property
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_q_from_qbc(self, int num_ghost, np.ndarray qbc):
        r"""
        Set the value of q using the array qbc. For CUDASolver, this
        involves retrieving from device memory into qbc, then copying 
        the appropriate data into q
        """

        cdef int err

        err = hyperbolic_solver_2d_get_qbc(&self.qbc)
        check_err(err)

        self.q = qbc[:,num_ghost:-num_ghost,num_ghost:-num_ghost]

    def get_qbc_from_q(self,num_ghost,qbc):
        """
        Fills in the interior of qbc by copying q to it.  For CUDASolver,
        this involves copying the appropriate data into qbc, then copying
        qbc into device memory.
        """

        cdef int err

        qbc[:,num_ghost:-num_ghost,num_ghost:-num_ghost] = self.q

        err = hyperbolic_solver_2d_set_qbc(&self.qbc)
        check_err(err)

        return qbc

class CUDASolver(clawpack.pyclaw.Solver):
    r"""  See the corresponding PyClaw Solver documentation."""

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def allocate_workspace(self, solution):
    r"""
    allocate_workspace (solution):

    Allocates memory for CUDA subroutines based on solution
    """

    state = solution.state
    grid  = state.grid

    int ghostCells = self.num_ghost

    int cellsX = grid.num_cells[0] + ghostCells*2
    int cellsY = grid.num_cells[1] + ghostCells*2
    int numStates = 3
    int numWaves  = 3
    int numCoeff  = 1
    double startX = grid.x.lower 
    double endX   = grid.x.upper
    double startY = grid.y.lower
    double endY   = grid.y.upper

    pdeParam params = pdeParam(cellsX,
                               cellsY,
                               ghostCells,
                               numStates,
                               numWaves,
                               numCoeff,
                               startX,
                               endX,
                               startY,
                               endY)

    shallow_water_solver_allocate(&params)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setup(self, solution):
    r"""
    setup (solution):

    Sets up a 2-D Shallow-Water Riemann solver with appropriate boundary 
    conditions, limiter, grid size, and ghost cells.
    """

    cdef int err    

    # PyClaw boundary conditions
    # bc_lower = [left, down]
    # bc_upper = [right, up]

    # CUDACLAW boundary conditions
    # bc = [left, right, up, down]
    # left, ... columns ... , right 
    # up, ... rows ... , down

    # Note that CUDACLAW assumes an upper-left hand origin,
    # instead of PyClaw's lower-left hand origin, we flip
    # the CUDACLAW boundary conditions here so that the arrays line up

    bc_left  = self.bc_lower[0]
    bc_right = self.bc_upper[0]
    bc_up    = self.bc_lower[1]
    bc_down  = self.bc_upper[1]

    limiter  = self.limiters

    err = shallow_water_solver_setup(bc_left,
                                     bc_right,
                                     bc_up,
                                     bc_down,
                                     limiter)
    check_err(err)

    self.allocate_workspace(solution)

@cython.boundscheck(False)
@cython.wraparound(False)
def step(double dt):
    """
    step (dt):
    
    Given input timestep dt, applies boundary conditions, advances the 2-D 
    Shallow-Water Riemann solution by the requested time dt, then returns a
    suggested next_dt that obeys the CFL condition for the current solution by
    a safety factor of 0.9.

    param: dt 
    param: next_dt
    """

    cdef double next_dt

    err = shallow_water_solver_step(dt, &next_dt)
    check_err(err)    
    return next_dt


