import numpy as np
import numpy.linalg as npl
import time
import acoustics_weno as acoustics
from matplotlib import pylab
f = open('workfile_weno', 'a')

def calc_p(test,f):
    p = np.log(test[1,1:]/test[1,0:-1])/np.log(test[0,1:]/test[0,0:-1])
    np.savetxt(f,p)
    return


# initialize arrays and number of points
idx = 0
j = np.arange(4,18)
N = 2**j
ctest = np.zeros([4,3,len(N)])

for i,k in enumerate(range(2,6,1)):
    porder = 2*k + 1
    for n in N:
        start = time.clock()
        claw = acoustics.acoustics(solver_type='sharpclaw',nx=n,norder=porder,use_petsc=1)
        dx = claw.frames[-1].state.grid.delta[0]
        t = time.clock() - start
        ctest[i,1,idx] = dx*npl.norm(claw.frames[-1].q[0,:]-claw.frames[0].q[0,:],1)
        ctest[i,0,idx] = n
        ctest[i,2,idx] = t
        print ctest[i,0,idx],ctest[i,1,idx],t
        idx +=1
    

    calc_p(ctest[i,:,:],f)
    ap = 'weno_test' + str(k) + '.txt'
    fp = open(ap, 'w')
    np.savetxt(fp,ctest[i,:,:]) 
    fp.close() 
    idx = 0 
    # pylab.figure()
    # pylab.plot(ctest)
    # pylab.show()

# close workfile
f.close()

# plot results
pylab.clf()
pylab.hold(True)
for i in range(ctest.shape[0]):
    pylab.loglog(N,ctest[i,1,:],'o-')
    print i
    pylab.hold(False)
    a = 'test-weno' + str(i+1) + '.pdf'
    pylab.savefig(a)


#for i in range(8):
#    pylab.figure()
#    pylab.contourf(pml[i,:,:].copy())
#    pylab.colorbar()
#pylab.show()
