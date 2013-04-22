import numpy as np
import numpy.linalg as npl
import time
import acoustics
from matplotlib import pylab
f = open('workfile', 'w')
xf = 10100
#ctest = np.empty([1,10])
idx = 0
ctest = np.zeros([3,(xf-100)/100])
for k in range(2,6,1):
	porder = 2*k
	for n in range(100,xf,100):
		start = time.clock()
		claw = acoustics.acoustics(solver_type='sharpclaw',nx=n,norder=porder)
		t = time.clock() - start
		ctest[1,idx] = npl.norm(claw.frames[-1].q[0,:]-claw.frames[0].q[0,:])
		ctest[0,idx] = n
		ctest[2,idx] = t
		# print ctest[0,idx],ctest[1,idx],t
		idx +=1
	
	a = 'ctest' + str(k) + '.txt'
	np.savetxt(a,ctest)	
	idx = 0	
	# pylab.figure()
	# pylab.plot(ctest)
	# pylab.show()



#for i in range(8):
#    pylab.figure()
#    pylab.contourf(pml[i,:,:].copy())
#    pylab.colorbar()
#pylab.show()