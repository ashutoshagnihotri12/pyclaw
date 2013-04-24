.. _cudaclaw:

*********
CUDACLAW
*********


PyClaw now features speedups through integration with the
CUDACLAW package, which provides GPU acceleration through the CUDA
programming interface.

Getting Started
===============

CUDACLAW integration is currently in pre-release testing.  Please
contact us on claw-dev@googlegroups.com if you are planning to use
CUDACLAW for any non-trivial applications or research.

You will need to obtain the following software prerequisites (beyond
the PyClaw requirements):

    * CUDA 4 SDK or `CUDA 5 <https://developer.nvidia.com/cuda-downloads>`_

    * `Cython <http://cython.org/>`_

Developer's Setup
=================

Currently, CUDACLAW development is tracked in the `ahmadia/clawpack cudaclaw
<https://github.com/ahmadia/clawpack/commits/cudaclaw>`_ branch
hosted on github.

At this point, we strongly recommend that you download the git
repository, then manually build and install.

If you are planning on development, first go to
`Aron's clawpack superrepository fork at <https://github.com/ahmadia/clawpack>`_,
create your own `superrepository fork
<https://github.com/ahmadia/clawpack/fork>`_, then clone it:  ::

    git clone -b cudaclaw --recursive git@github.com:your_username/clawpack.git
    cd clawpack
    pip install -e .

When making commits, ensure that you always commit and push changes to
the subrepositories before committing changes to the superrepository.

Alternatively, if you are planning on simply tracking changes, you
can follow Aron's fork directly: ::

    git clone -b cudaclaw --recursive git://github.com/ahmadia/clawpack.git
    cd clawpack
    pip install -e .

Check your basic installation by navigating to the pyclaw subdirectory and
running nose: ::

    cd pyclaw
    nosetests

Check the CUDACLAW installation by (from the pyclaw directory) heading
to the apps/shallow_2d subdirectory and running the cudaclaw test code: ::

    cd apps/shallow_2d
    python cudaclaw_shallow2D.py iplot=1
