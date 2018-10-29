*****************************************************************      
      program mainqtet
      implicit none
      integer i,npt,npart,icon,mcon,ind,nbin,ibin
      parameter (npt=60000,mcon=100,nbin=1000)
      double precision x(npt),y(npt),z(npt)
      double precision del
      double precision qtet(npt),avgqtet
      integer qbin(nbin)
      external calqtet

      integer maxframes,maxatoms,natoms,nframes
      parameter (maxframes=25000,maxatoms=60000)
      write(*,*)""
      real*4 rx(maxframes,maxatoms)
      real*4 ry(maxframes,maxatoms)
      real*4 rz(maxframes,maxatoms)
      double precision xbox,ybox,zbox
      real start,finish
*****************************************************************
      open(222,file="qtet.dat")
      open(100,file="avgqtet.dat")
 
      nframes=mcon

      call cpu_time(start)
      call readdcd
     *(maxframes,maxatoms,rx,ry,rz,xbox,ybox,zbox,natoms,nframes)
*****************************************************************
      npart=natoms
      del=4.0d0/dble(nbin)
      avgqtet=0.0d0
*******************************************
c      open(unit=10,status='unknown',file='./fort.423')
**************************************************************************

      qbin=0   
      do  icon=1,mcon
        do i=1,npart
           x(i)=rx(icon,i)
           y(i)=ry(icon,i)
           z(i)=rz(icon,i)
        enddo
 
       call  calcqtet (x,y,z,qtet,xbox,ybox,zbox,npart)
       if(mod(icon,10).eq.0) then
         write(*,*)'configuration',icon,'of ',mcon,' is done...'
       endif
c          write(111,'(1f10.5)')(qtet(i),i=1,npart)
        do i=1,npart
          ibin=(3+qtet(i))/del
          avgqtet=avgqtet+qtet(i)
          if((ibin.ge.1).and.(ibin.le.nbin)) then
            qbin(ibin)=qbin(ibin)+1
          else
            write(*,*)qtet(i),ibin
          endif
        enddo

      enddo
      
      do i=1,nbin
        if(qbin(i).gt.0) then
          write(222,*)dble(i)*del-3.0d0,dble(qbin(i))/dble(nbin*mcon)
        endif
      enddo
      avgqtet=avgqtet/dble(mcon*npart)
      write(*,*)
      write(100,'(a,1f9.4)')"Avg qtet ",avgqtet
      write(*,'(a,1f9.4)')"Avg qtet ",avgqtet

      call cpu_time(finish)
      print*,"Qtet calculation started at",start,"ended at",finish

       stop
       end


      subroutine readdcd
     *(maxframes,maxatoms,rx,ry,rz,xbox,ybox,zbox,natoms,nframes)
       integer i,j
       integer maxframes,maxatoms

       double precision d(6),xbox,ybox,zbox
       real*4 rx(maxframes,maxatoms)
       real*4 ry(maxframes,maxatoms)
       real*4 rz(maxframes,maxatoms)
       real*4 dummyr
       integer*4 nset, natoms, dummyi,nframes,tframes
       character*4 dummyc

       open(10,file='o.traj.dcd',status='old',form='unformatted')
       read(10) dummyc, tframes,(dummyi,i=1,8),dummyr, (dummyi,i=1,9)
       read(10) dummyi, dummyr,dummyr
       read(10) natoms
       print*,"Total number of frames are",tframes
           do i = 1,nframes
           read(10) (d(j),j=1, 6)

           read(10) (rx(i,j),j=1,natoms)
           read(10) (ry(i,j),j=1,natoms)
           read(10) (rz(i,j),j=1,natoms)
          end do
          xbox=d(1)
          ybox=d(3)
          zbox=d(6)
          print*,"File reading is done"
          return

       end subroutine readdcd

