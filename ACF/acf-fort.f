      program ionicconductivity
      implicit none
      integer i,r,nconf,maxnconf,t,tcor,t0,tt0max,tt0,w
      parameter (nconf=5000000)
      parameter (maxnconf=5000000)
      parameter (tcor=1000000)
      double precision q,temp
      real*8 ptx(0:maxnconf),pty(0:maxnconf),ptz(0:maxnconf)
      double precision boltz,vol,const,norm(0:tcor)
      real*8 acf(0:tcor)
      double precision tstep,sumionic,intacf(0:tcor)
      double precision xbox,ybox,zbox
      write(*,*)"Enter the boxlengths for x, y, and z direction"
      read(*,*) xbox,ybox,zbox
      write(*,*)"Enter the temperature (K) and timestep (fs)"
      read(*,*) temp,tstep
      parameter (boltz=1.38d-23)
      external   calcacf
      real start,finish

      
      open(400,file="acf.dat")
 
      call cpu_time(start)
      vol = (xbox*ybox*zbox)*(1.0d-10)**3
      const = vol/(boltz*temp)
      open (1,file='flux.dat',status='old')
      read(1,*)
      do i=1,nconf
         read (1,*)ptx(i-1),pty(i-1),ptz(i-1)
      enddo
      print*,"File reading..... Done!"
      call calcacf(ptx,pty,ptz,acf,nconf,tcor)
      
      print*,"ACF calculation ..... Done!"

      sumionic=0.0d0
      do t=0,tcor
         acf(t)=acf(t)/dble(3*(nconf-t))
         sumionic=sumionic+acf(t)
c         sumionic=sumionic-0.5*(acf(0)+acf(t))
         intacf(t)=sumionic*const*tstep*(1.0d-15)*((101325.0)**2.0)
         write(400,*)t*tstep,acf(t),intacf(t)
      enddo
      
      call cpu_time(finish)
      
      print*,"ACF calculation started at",start,"ended at",finish
      stop
      end

