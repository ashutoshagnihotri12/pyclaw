import os
import logging, logging.config

# Default logging configuration file
_DEFAULT_LOG_CONFIG_PATH = os.path.join(os.path.dirname(__file__),'log.config')
del os

# Setup loggers
logging.config.fileConfig(_DEFAULT_LOG_CONFIG_PATH)
__all__ = []

__all__.extend(['Controller','Dimension','Patch','Domain',
	            'Solution','State','CFL','riemann'])

from clawpack.pyclaw.controller import Controller
from clawpack.pyclaw.geometry import Patch, Domain, Dimension
from clawpack.pyclaw.solution import Solution 
from clawpack.pyclaw.cfl import CFL
from clawpack.cudaclaw.solver import CUDASolver2D
from clawpack.cudaclaw.state import State

__all__.append('BC')
from clawpack.pyclaw.solver import BC

# Sub-packages
import limiters
from limiters import *
__all__.extend(limiters.__all__)

from clawpack.pyclaw import plot
__all__.append('plot')
