"""
multiply.pyx

simple cython test of accessing a numpy array's data

the C function: c_multiply multiplies all the values in a 2-d array by a scalar, in place.

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_multiply (double* array, double value, int m, int n) 
cdef extern void cuda_multiply (double* array, double value, int m, int n) 

@cython.boundscheck(False)
@cython.wraparound(False)
def multiply(np.ndarray[double, ndim=2, mode="c"] input not None, double value):
    """
    multiply (arr, value)
    
    Takes a numpy array as input, and multiplies each element by value, in place
    
    param: array -- a 2-d numpy array of np.float64
    param: value -- a number that will be multiplied by each element in the array
    
    """
    cdef int m, n
    
    m, n = input.shape[0], input.shape[1]
    
    c_multiply (&input[0,0], value, m, n)
    
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
def multiply_cuda(np.ndarray[double, ndim=2, mode="c"] input not None, double value):
    """
    cuda_multiply (arr, value)
    
    Takes a numpy array as input, and multiplies each element by value, in place
    
    param: array -- a 2-d numpy array of np.float64
    param: value -- a number that will be multiplied by each element in the array
    
    """
    cdef int m, n
    
    m, n = input.shape[0], input.shape[1]
    
    cuda_multiply (&input[0,0], value, m, n)
    
    return None
