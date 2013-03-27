    subroutine rpn2(ixy,maxm,meqn,mwaves,mbc,mx,ql,qr,auxl,auxr, &
    fwave,s,amdq,apdq,num_aux)
!     =====================================================

!     # This version outputs f-waves.

!     # Riemann solver for the time dependent nonlinear em equations in 1d,
!     # in this case eps and mu dependend on time and position,
!     #  variable coefficients
!     #   kappa1(q,t,x) (q1)_t +  (q2)_x  = -eps_t E
!     #   kappa2(q,t,x) (q2)_t +  (q1)_x = -mu_t H
!     # where q1=E, q2=H, eps=f(x,t), and mu=g(x,t)
!     # and kappa1 = eps + 2*chi2_e_r*E + 3*chi3_e*E^2
!     # and kappa2 = mu  + 2*chi2_m_r*H + 3*chi3_m*H^2

!     # aux(1,i) = eps(i)
!     # aux(2,i) = mu(i)
!     # aux(3,i) = eps_t(i)
!     # aux(4,i) = mu_t(i)

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
!     #                   the decomposition of the flux difference minus the source term
!     #                       f(qr(i-1)) - f(ql(i)) - \psi(q,x,t)
!     #                   into leftgoing and rightgoing parts respectively.
!     #

!     # Note that the ith Riemann problem has left state qr(:,i-1)
!     #                                    and right state ql(:,i)
!     # From the basic clawpack routines, this routine is called with ql = qr


    implicit none

    double precision :: auxl(num_aux,1-mbc:maxm+mbc)
    double precision :: auxr(num_aux,1-mbc:maxm+mbc)
    double precision :: fwave(mwaves,meqn,1-mbc:maxm+mbc)
    double precision ::    s(mwaves,1-mbc:maxm+mbc)
    double precision ::   ql(meqn,1-mbc:maxm+mbc)
    double precision ::   qr(meqn,1-mbc:maxm+mbc)
    double precision :: apdq(meqn,1-mbc:maxm+mbc)
    double precision :: amdq(meqn,1-mbc:maxm+mbc)
    double precision :: wave(meqn,mwaves,1-mbc:maxm+mbc)

    integer :: i, mx, mbc, maxm, num_aux, meqn, mwaves, m, ixy

    double precision :: epsi_r, epsim_r, mui_r, muim_r, q1i, q1im, q2i, q2im, q3i, q3im, ci, cim, zi, zim
    double precision :: epsti_r, epstim_r, muti_r, mutim_r, kappa1, kappa2, kappa3, eo, mo, zo, co
    double precision :: df1, df2, df3, psi1, psi2, psi3, b1, b2, dx, dy, chi2_e_r, chi2_m_r, chi3_e_r, chi3_m_r
    double precision :: dq1, dq2, dq3, beta1, beta2, beta3
    common /cparam/  dx, dy, chi2_e_r, chi2_m_r, chi3_e_r, chi3_m_r, eo, mo, co, zo

!     # split the jump in q at each interface into waves

    do 20 i = 2-mbc, mx+mbc
        epsi_r      = auxl(1,i  )
        epsim_r     = auxr(1,i-1)
        mui_r       = auxl(2,i  )
        muim_r      = auxr(2,i-1)
        epsti_r     = auxl(3,i  )
        epstim_r    = auxr(3,i-1)
        muti_r      = auxl(4,i  )
        mutim_r     = auxr(4,i-1)

        q1i    = ql(1,i)
        q1im   = qr(1,i-1)
        q2i    = ql(2,i)
        q2im   = qr(2,i-1)
        q3i    = ql(3,i)
        q3im   = qr(3,i-1)

!     # calculate velocity c = 1/sqrt(eps*mu) and impedance Z = sqrt(eps/mu)
        ci  = co 
        cim = co
        zi  = zo
        zim = zo

        psi1 = - 0.5d0*(epsti_r*q1i + epstim_r*q1im)
        psi2 = - 0.5d0*(epsti_r*q2i + epstim_r*q2im)
        psi3 = - 0.5d0*(muti_r*q3i  + mutim_r*q3im )

!     # flux difference
        
        df1 = (q1i - q1im)/eo
        df2 = (q2i - q2im)/eo
        df3 = (q3i - q3im)/mo

        kappa1 = 0.5d0*(epsi_r + epsim_r + 2*chi2_e_r*(q1i + q1im) + 3*chi3_e_r*((q1i + q1im)**2))
        kappa1 = 0.5d0*(epsi_r + epsim_r + 2*chi2_e_r*(q2i + q2im) + 3*chi3_e_r*((q2i + q2im)**2))
        kappa3 = 0.5d0*(mui_r  + muim_r  + 2*chi2_m_r*(q3i + q3im) + 3*chi3_m_r*((q3i + q3im)**2))
        !   Normal & perpendicular waves
!   ------------
        if (ixy==1) then
            dq2 = df2 - dx*psi3
            dq3 = df3 - dx*psi2   
            beta1 = (-dq3+dq2*zi)/(zi+zim)
            beta2 = 0.
            beta3 = (dq3+dq2*zim)/(zi+zim)
            wave(1,1,i) = 0.
            wave(2,1,i) = beta1 * (-zim) / kappa2
            wave(3,1,i) = beta1 / kappa3
            wave(1,2,i) = 0.
            wave(2,2,i) = beta3 * (zi) / kappa2
            wave(3,2,i) = beta3 / kappa3
            s(1,i) = -cim
            s(2,i) = ci 
        else
            dq1 = -df1 - dy*psi3
            dq3 = -df3 - dy*psi1
            beta1 = (dq3+dq1*zi)/(zi+zim)
            beta2 = 0
            beta3 = (-dq3+dq1*zim)/(zi+zim)
            wave(1,1,i) = beta1 * (zim) / kappa1
            wave(2,1,i) = 0.
            wave(3,1,i) = beta1 / kappa3
            wave(1,2,i) = beta3 * (-zi) / kappa1
            wave(2,2,i) = 0.
            wave(3,2,i) = beta3 / kappa3
            s(1,i) = -cim
            s(2,i) = ci 
            !if (beta1.gt.1.e-10) then
            !    write(*,*) s(1,i), beta1
            !endif 
        endif

    20 END DO


!     # compute the leftgoing and rightgoing fluctuations:
!     # Note s(1,i) < 0   and   s(2,i) > 0.

    do 220 m=1,meqn
        do 220 i = 2-mbc, mx+mbc
            amdq(m,i) = fwave(m,1,i)
            apdq(m,i) = fwave(m,2,i)
    220 END DO

    return
    end subroutine rpn2
