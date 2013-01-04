#!/usr/bin/env python
"""CUDAClaw

adapted from http://stackoverflow.com/a/13300714/122022
by Robert McGibbon (used under StackOverflow CC-BY license)
"""

import os
from os.path import join as pjoin
import subprocess
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.core import setup
import numpy

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted by Robert from 
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    This functionality should really be brought in from numpy or PyCUDA.

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib'
    and values giving the absolute path to each directory.

    Also returns 'cuflags', which allows for customizing compiler flags,
    these are currently hardcoded to 64-bit CUDA 5 flags

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib': pjoin(home, 'lib')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    cudaconfig['cuflags'] = '-m64 -gencode arch=compute_10,code=sm_10' + \
    					    ' -gencode arch=compute_20,code=sm_20' + \
    					    ' -gencode arch=compute_30,code=sm_30' + \
    					    ' -gencode arch=compute_35,code=sm_35' 

    return cudaconfig

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

cuda_multiply = Extension('cuda_multiply',
						  sources=['cuda_multiply.cu'])


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(
    	'cudaclaw', 
    	parent_package, 
    	top_path,
    	)
    return config

def customize_compiler_for_nvcc(self):
    """decorates numpy build_ext class to handle .cu source files

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. Instead, we take 
    advantage of Python's dynamism to over-ride the class function directly
    """

    # tell the compiler it can process .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    super_compile = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
    	postargs = []
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda compiler for .cu files
            # currently hard-coded to OS X CUDA 5 options
            self.set_executable('compiler_so', 
            	                CUDA['nvcc'] + ' -Xcompiler -fPIC ' + CUDA['cuflags'])
            # set postargs for either '.cu' or '.c'
            # from the extra_compile_args in the Extension class
            if '.cu' in extra_postargs:
            	postargs = extra_postargs['.cu']
        else:
            if '.c' in extra_postargs:
            	postargs = extra_postargs['.c']

        super_compile(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# decorate build_ext
class cuda_build_ext(build_ext):
    def build_extensions(self):
    	# this is a bit hacky
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

if __name__ == '__main__':
    setup(cmdclass = {'build_ext': cuda_build_ext},
	      ext_modules = [Extension("multiply_cuda",
 	  				     sources=["multiply.pyx", 
 	  				     		  "c_multiply.c",
 	  				     		  "cuda_multiply.cu"],
 	  				     library_dirs=[CUDA['lib']],
 	  				     libraries=['cudart'],
 	  				     runtime_library_dirs=[CUDA['lib']],
          				 include_dirs=[numpy.get_include()])],
		  **configuration(top_path='').todict())
