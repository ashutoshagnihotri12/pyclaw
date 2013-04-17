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

                ql(n,i) = (-5*q(n,-2+i)+60*q(n,-1+i)+90*q(n,i)-20*q(n,1+i)+3*q(n,2+i))/128.d0
                qr(n,i) = (3*q(n,-2+i)-20*q(n,-1+i)+90*q(n,i)+60*q(n,1+i)-5*q(n,2+i))/128.d0

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

                ql(n,i) = (7*(q(n,-3+i) - 10*q(n,-2+i) + 75*q(n,-1+i) + 100*q(n,i) &
                           - 25*q(n,1+i) + 6*q(n,2+i)) - 5*q(n,3+i))/1024.d0
                qr(n,i) = (-5*q(n,-3+i) + 7*(6*q(n,-2+i) - 25*q(n,-1+i) + 100*q(n,i) &
                           + 75*q(n,1+i) - 10*q(n,2+i) + q(n,3+i)))/1024.d0

            end do
        end do
    
    end subroutine poly6

end module poly
