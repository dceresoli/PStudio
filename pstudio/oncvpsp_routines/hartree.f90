!
! Copyright (c) 1989-2019 by D. R. Hamann, Mat-Sim Research LLC and Rutgers
! University
!
! 
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!
! modified from vout.f90 by D. Ceresoli to calculate only the Hartree potential

 subroutine hartree(rho,vo,zion,rr,mmax)

!rho  total charge density or valence/pseudovalence charge density
!vo  output total potential
!zion  charge of screened ion
!rr  log radial mesh

 implicit none

 integer, parameter :: dp=kind(1.0d0)
 real(kind=dp), parameter :: pi=3.141592653589793238462643383279502884197_dp
 real(kind=dp), parameter :: pi4=4.0_dp*pi

!Input vaiables
 integer, intent(in) :: mmax
 real(kind=dp), intent(in) :: rho(mmax), rr(mmax)
 real(kind=dp), intent(in) :: zion

!Output variables
 real(kind=dp), intent(out) :: vo(mmax)

!Local variables
 integer ii
 real(kind=dp) :: al, tv

!Function
 real(kind=dp) :: aii
 real(kind=dp), allocatable :: rvp(:), rv(:)

 allocate(rvp(mmax),rv(mmax))

 al = 0.01d0 * dlog(rr(101) / rr(1))

! integration for electrostatic potential
 do ii=1,mmax
   rvp(ii)=rho(ii)*al*rr(ii)**3
 end do

 rv(mmax)=zion
 rv(mmax-1)=zion
 rv(mmax-2)=zion

 do ii=mmax-2,2,-1
   rv(ii-1)=rv(ii)+aii(rvp,ii)
 end do

 do ii=1,mmax
   rvp(ii)=rho(ii)*al*rr(ii)**2
 end do

 tv=0.0d0
 do ii=mmax-2,2,-1
   tv=tv+aii(rvp,ii)
   rv(ii-1)=rv(ii-1)-rr(ii-1)*tv
 end do

 do ii=1,mmax
   vo(ii)=rv(ii)/rr(ii)
 end do

 deallocate(rvp,rv)

 return

 end subroutine hartree
