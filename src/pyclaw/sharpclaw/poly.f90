module poly
contains

    ! ===================================================================
    subroutine poly4(q,ql,qr,num_eqn,maxnx,num_ghost)
    ! ===================================================================

        implicit none

        integer,          intent(in) :: num_eqn, maxnx, num_ghost
        double precision, intent(in) :: q(num_eqn,maxnx+2*num_ghost)
        double precision, intent(out) :: ql(num_eqn,maxnx+2*num_ghost),qr(num_eqn,maxnx+2*num_ghost)

        integer :: i,n

        do n = 1,num_eqn
            do i = num_ghost,maxnx+num_ghost+1

                ql(n,i) = (-5.d0*q(n,-2+i)+60.d0*q(n,-1+i)+90.d0*q(n,i)-20.d0*q(n,1+i)+3.d0*q(n,2+i))/128.d0
                qr(n,i) = (3.d0*q(n,-2+i)-20.d0*q(n,-1+i)+90.d0*q(n,i)+60.d0*q(n,1+i)-5.d0*q(n,2+i))/128.d0

            end do
        end do

    end subroutine poly4

    ! ===================================================================
    subroutine poly6(q,ql,qr,num_eqn,maxnx,num_ghost)
    ! ===================================================================

        implicit none

        integer,          intent(in) :: num_eqn, maxnx, num_ghost
        double precision, intent(in) :: q(num_eqn,maxnx+2*num_ghost)
        double precision, intent(out) :: ql(num_eqn,maxnx+2*num_ghost),qr(num_eqn,maxnx+2*num_ghost)

        integer :: i,n

        do n = 1,num_eqn
            do i = num_ghost+1,maxnx+num_ghost+1

                ql(n,i) = (7.d0*(q(n,-3+i) - 10.d0*q(n,-2+i) + 75.d0*q(n,-1+i) + 100.d0*q(n,i) &
                           - 25.d0*q(n,1+i) + 6.d0*q(n,2+i)) - 5.d0*q(n,3+i))/1024.d0
                qr(n,i) = (-5.d0*q(n,-3+i) + 7.d0*(6*q(n,-2+i) - 25.d0*q(n,-1+i) + 100.d0*q(n,i) &
                           + 75.d0*q(n,1+i) - 10.d0*q(n,2+i) + q(n,3+i)))/1024.d0

            end do
        end do
    
    end subroutine poly6

    ! ===================================================================
    subroutine poly8(q,ql,qr,num_eqn,maxnx,num_ghost)
    ! ===================================================================

        implicit none

        integer,          intent(in) :: num_eqn, maxnx, num_ghost
        double precision, intent(in) :: q(num_eqn,maxnx+2*num_ghost)
        double precision, intent(out) :: ql(num_eqn,maxnx+2*num_ghost),qr(num_eqn,maxnx+2*num_ghost)

        integer :: i,n

        do n = 1,num_eqn
            do i = num_ghost+1,maxnx+num_ghost+1
                        
                ql(n,i) = (-45.d0*q(n,-4+i) + 42.d0*(12*q(n,-3+i) + 7.d0*(-10*q(n,-2+i) + 60.d0*q(n,-1+i) &
                           + 75.d0*q(n,i) - 20.d0*q(n,1+i) + 6.d0*q(n,2+i))) - 360.d0*q(n,3+i) + 35.d0*q(n,4+i))/32768.d0
                qr(n,i) = (35.d0*q(n,-4+i) - 360.d0*q(n,-3+i) + 294.d0*(6*q(n,-2+i) - 20.d0*q(n,-1+i) + 75.d0*q(n,i) &
                           + 60.d0*q(n,1+i) - 10.d0*q(n,2+i)) + 504.d0*q(n,3+i) - 45.d0*q(n,4+i))/32768.d0

            end do
        end do
    
    end subroutine poly8

    ! ===================================================================
    subroutine poly10(q,ql,qr,num_eqn,maxnx,num_ghost)
    ! ===================================================================

        implicit none

        integer,          intent(in) :: num_eqn, maxnx, num_ghost
        double precision, intent(in) :: q(num_eqn,maxnx+2*num_ghost)
        double precision, intent(out) :: ql(num_eqn,maxnx+2*num_ghost),qr(num_eqn,maxnx+2*num_ghost)

        integer :: i,n

        do n = 1,num_eqn
            do i = num_ghost+1,maxnx+num_ghost+1

                ql(n,i) = (77.d0*q(n,-5+i) - 99.d0*(10.d0*q(n,-4+i) - 7.d0*(9.d0*q(n,-3+i) - 40.d0*q(n,-2+i) &
                           + 210.d0*q(n,-1+i) + 252.d0*q(n,i) - 70.d0*q(n,1+i) + 24.d0*q(n,2+i)) + 45.d0*q(n,3+i)) &
                           + 770.d0*q(n,4+i) - 63.d0*q(n,5+i))/262144.d0
                qr(n,i) = (-63.d0*q(n,-5+i) + 11.d0*(70.d0*q(n,-4+i) - 405.d0*q(n,-3+i) + 63.d0*(24.d0*q(n,-2+i) & 
                           - 70.d0*q(n,-1+i) + 252.d0*q(n,i) + 210.d0*q(n,1+i) - 40.d0*q(n,2+i) + 9.d0*q(n,3+i)) &
                           - 90.d0*q(n,4+i) + 7.d0*q(n,5+i)))/262144.d0
            end do
        end do
    end subroutine poly10

end module poly
