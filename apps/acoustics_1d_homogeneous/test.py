import numpy as np
import numpy.linalg as npl
import time
import acoustics
from matplotlib import pylab
f = open('workfile', 'w')
xf = 10100
#ctest = np.empty([1,10])
idx = 0

j = np.arange(3,10)
N = 2**j

ctest = np.zeros([1,3,len(N)])

for i,k in enumerate(range(2,3,1)):
	porder = 2*k
        for n in N:
		start = time.clock()
		claw = acoustics.acoustics(solver_type='sharpclaw',nx=n,norder=porder)
                dx = claw.frames[-1].state.grid.delta[0]
		t = time.clock() - start
		ctest[i,1,idx] = dx*npl.norm(claw.frames[-1].q[0,:]-claw.frames[0].q[0,:],1)
		ctest[i,0,idx] = n
		ctest[i,2,idx] = t
		# print ctest[0,idx],ctest[1,idx],t
		idx +=1
	
	a = 'ctest' + str(k) + '.txt'
	#np.savetxt(a,ctest)	
	idx = 0	
	# pylab.figure()
	# pylab.plot(ctest)
	# pylab.show()


pylab.clf()
pylab.hold(True)
for i in range(ctest.shape[0]):
    pylab.loglog(N,ctest[i,1,:],'o-')
    print i
pylab.hold(False)
pylab.savefig('ctest.pdf')

#for i in range(8):
#    pylab.figure()
#    pylab.contourf(pml[i,:,:].copy())
#    pylab.colorbar()
#pylab.show()
