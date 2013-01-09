from cudaclaw import CUDASolver2D

# so super() init works in the generic solver.py, I really need to fix this code
CUDASolver2D.__module__ = 'clawpack.cudaclaw.classic.solver'