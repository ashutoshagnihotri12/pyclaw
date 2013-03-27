    subroutine rp1(maxm,meqn,mwaves,mbc,mx,ql,qr,auxl,auxr, &
    fwave,s,amdq,apdq,num_aux)
!     =====================================================

!     # This version uses interleaved arrays.
!     # This version outputs f-waves.

!     # Riemann solver for the time dependent nonlinear em equations in 1d,
!     # in this case eps and mu dependend on time and position,
!     #  variable coefficients
!     #   (q1)_t + (q2/mu)_x = 0
!     #   (q2)_t +  (q1/eps)_x = 0
!     # where q1=eps*E, q2=mu*H, eps=f(x,t), and mu=g(x,t)

!     # aux(1,i) = rho(i)
!     # function f(x_i,t_i) gives the permittivity value at the ith cell at step t_i
!     # function g(x_i,t_i) gives the permeability value at the ith cell at step t_i
!     #    For RIP:   f(x_i,t_i)=f(x_i-v*t_i), and g(x_i,t_i)=g(x_i-v*t_i)
!     # the system assumes the em functions to be some constant value + transient part

!     # On input, ql contains the state vector at the left edge of each cell
!     #           qr contains the state vector at the right edge of each cell

!     # On output, fwave contains the waves as jumps in f,
!     #            s the speeds,
!     #
!     #            amdq = A^- Delta q,
!     #            apdq = A^+ Delta q,
!     #                   the decomposition of the flux difference
!     #                       f(qr(i-1)) - f(ql(i))
!     #                   into leftgoing and rightgoing parts respectively.
!     #

!     # Note that the ith Riemann problem has left state qr(:,i-1)
!     #                                    and right state ql(:,i)
!     # From the basic clawpack routines, this routine is called with ql = qr


    implicit double precision (a-h,o-z)

    dimension auxl(num_aux,1-mbc:maxm+mbc)
    dimension auxr(num_aux,1-mbc:maxm+mbc)
    dimension fwave(mwaves,meqn,1-mbc:maxm+mbc)
    dimension    s(mwaves,1-mbc:maxm+mbc)
    dimension   ql(meqn,1-mbc:maxm+mbc)
    dimension   qr(meqn,1-mbc:maxm+mbc)
    dimension apdq(meqn,1-mbc:maxm+mbc)
    dimension amdq(meqn,1-mbc:maxm+mbc)

    common /comxyt/ dtcom,dxcom,dycom,tcom,icom,jcom


!     # split the jump in q at each interface into waves

    do 20 i = 2-mbc, mx+mbc
        epsi   = auxl(1,i  )
        epsim  = auxr(1,i-1)
        xmui   = auxl(2,i  )
        xmuim  = auxr(2,i-1)
        epsEi  = ql(1,i  )
        epsEim = qr(1,i-1)
        xmuHi  = ql(2,i  )
        xmuHim = qr(2,i-1)

!     # calculate velocity c = 1/sqrt(eps*mu) and impedance Z = sqrt(eps/mu)
        ci = 1.d0 / dsqrt(epsi * xmui)
        cim = 1.d0 / dsqrt(epsim * xmuim)
        zi = dsqrt(epsi / xmui)
        zim = dsqrt(epsim / xmuim)

!     # flux difference
        df1 = xmuHi/xmui - xmuHim/xmuim
        df2 = epsEi/epsi - epsEim/epsim

        b1 = (zi * df2 - df1) / (zim + zi)
        b2 = (zim * df2 + df1) / (zim + zi)

!        a1 = -b1 / cim
!        a2 = b2 / ci
  
!     # Compute the waves.
    
        fwave(1,1,i) = b1 *(-zim)
        fwave(2,1,i) = b1
        s(1,i) = -cim
    
        fwave(1,2,i) = b2*(zi)
        fwave(2,2,i) = b2
        s(2,i) = ci


    20 END DO


!     # compute the leftgoing and rightgoing fluctuations:
!     # Note s(1,i) < 0   and   s(2,i) > 0.

    do 220 m=1,meqn
        do 220 i = 2-mbc, mx+mbc
            amdq(m,i) = fwave(m,1,i)
            apdq(m,i) = fwave(m,2,i)
    220 END DO

    return
    end subroutine rp1
