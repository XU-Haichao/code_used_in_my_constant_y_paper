
      SUBROUTINE REFLCT(Ear,Ne,Param,Ifl,Photar)
 
      IMPLICIT NONE
      INTEGER Ne , Ifl
      REAL Ear(0:Ne) , Param(*) , Photar(Ne)
 
c     Driver for angle-dependent reflection from a neutral medium.
c     See Magdziarz & Zdziarski, 1995, MNRAS.
c
c     The output spectrum is the sum of the input (the original contents
c     of Photar) and the reflection component.
c     The reflection component alone can be obtained
c     for scale (see below) = rel_refl < 0. Then the actual
c     reflection normalization is |scale|. Note that you need to
c     change then the limits of rel_refl. The range of rel_refl in that case
c     should exclude zero (as then the direct component appears).
c
c     Version with variable iron abundance and new opacities of Balucinska
c     & McCammon (1992, and 1994, private communication). As expected in AGNs,
c     H and He are assumed to be fully ionized.
c
c     This version allows for changes of the vector 'ear' between subsequent
c       calls.
c
c
c     number of model parameters: 5
c     1: scale, scaling factor for reflection; if <0, no direct component
c        (scale=1 for isotropic source above disk)
c     2: redshift, z
c     3: abundance of elements heavier than He relative to
c        the solar abundances
c     4: iron abundance relative to the solar iron abundance
c     5: cosine of inclination angle
c     algorithm:
c          a[E*(1+z)]=E^{-Gamma}*exp(-E/E_c)+scale*reflection
c     Normalization is the photon flux at 1 keV (photons keV^-1 cm^-2 s^-1)
c     of the cutoff power law only (without reflection)
c     and in the earth frame.
c
c
      INCLUDE 'xspec.inc'

      INTEGER isptot, ispref, ispinc, ix, iphotaux

      INTEGER i , j , ierr, nesave
      LOGICAL firstcall
      CHARACTER contxt*127

      SAVE isptot, ispref, ispinc, ix
      SAVE firstcall, nesave

      DATA isptot, ispref, ispinc, ix, nesave/5*-1/
      DATA firstcall/.TRUE./
 
      IF ( firstcall ) THEN
         CALL xwrite('Compton reflection from neutral medium.',5)
         CALL xwrite('If you use results of this model in a paper,',5)
         CALL xwrite
     &  ('please refer to Magdziarz & Zdziarski 1995 MNRAS, 273, 837'
     &              ,5)
         firstcall = .FALSE.
      ENDIF

      IF ( Ne .NE. nesave ) THEN
         CALL udmget(Ne, 6, isptot, ierr)
         contxt = 'Failed to get memory for sptot'
         IF ( ierr .NE. 0 ) GOTO 999
         CALL udmget(Ne, 6, ispref, ierr)
         contxt = 'Failed to get memory for spref'
         IF ( ierr .NE. 0 ) GOTO 999
         CALL udmget(Ne, 6, ispinc, ierr)
         contxt = 'Failed to get memory for sppow'
         IF ( ierr .NE. 0 ) GOTO 999
         CALL udmget(Ne, 6, ix, ierr)
         contxt = 'Failed to get memory for x'
         IF ( ierr .NE. 0 ) GOTO 999
         nesave = Ne
      ENDIF

c   x is in the source frame and in units of m_e c^2.

      DO 100 i = 1 , Ne
         MEMR(ix+i-1) = ((Ear(i)+Ear(i-1))/2./511.)*(1.+Param(2))
 100  CONTINUE

C    Generate the spinc array from the input photar array.

      DO i = 1, Ne
         MEMR(ispinc+i-1) = photar(i) * (ear(i)+ear(i-1))/2.
      ENDDO

c     x is the energy array (units m_e c^2)
c     spinc is the input spectrum array (E F_E)
c     spref is the reflected spectrum array (E F_E)
c     sptot is the total spectrum array (E F_E), = spinc if no reflection
c     all dimensions = Ne

      CALL DOREFA(Param,MEMR(ix),Ne,MEMR(isptot),MEMR(ispinc),
     &             MEMR(ispref))

      DO 300 i = 1 , Ne
         Photar(i) = MEMR(isptot+i-1) / ((ear(i)+ear(i-1))/2.)
 300  CONTINUE

 999  CONTINUE
      IF ( ierr .NE. 0 ) THEN
         CALL xwrite(contxt, 10)
      ENDIF
 
      RETURN
      END
**==powrefa.spg  processed by SPAG 4.50J  at 11:22 on 27 Feb 1996
 
 
      SUBROUTINE DOREFA(Param,X,Jmax,Sptot,Spinc,Spref)
c
      IMPLICIT NONE
      REAL gamma , ec , scale , z_red , xm , xn
      REAL ab_ , ab0
      REAL Param(*) , Sptot(*) , Spref(*) , Spinc(*) , X(*)
      INTEGER Jmax , j
 
c--------
C  INPUT:
c the scaling factor for the reflected spectrum;
c 1 corresponds to seeing equal
c contributions from the reflected and direct spectra
      scale = Param(1)
c redshift
      z_red = Param(2)
c abundance
      ab_ = LOG10(Param(3))
c Iron abundance
      ab0 = LOG10(Param(4))
c Cosine of inclination angle
      xm = Param(5)
c--------
 
      IF ( scale.NE.0. ) THEN
         CALL DOGREEN(x,Jmax,Spref,xm,ab_,ab0,Spinc)
         DO 120 j = 1 , Jmax
            Spref(j) = Spref(j)*ABS(scale)
 120     CONTINUE
      ELSE
         DO 140 j = 1 , Jmax
            Spref(j) = 0
 140     CONTINUE
      ENDIF
      IF ( scale.GE.0 ) THEN
         DO 160 j = 1 , Jmax
            Sptot(j) = Spinc(j) + Spref(j)
 160     CONTINUE
      ELSE
         DO 180 j = 1 , Jmax
            Sptot(j) = Spref(j)
 180     CONTINUE
      ENDIF
 
      RETURN
      END
 
c ------------------------------------------------------------------ c
 
      SUBROUTINE DOGREEN(X,Jmax,Spref,Xm,Ab_,Ab0,Spinc)
c     calculates the reflected spectrum
c     ab0 - iron log10 abundance relative to hydrogen
      IMPLICIT NONE
      REAL Ab0 , Ab_ , ddy , dy , dym , Ec , fjc , Gamma , GNR , GRXYH , 
     &     GRXYL , SIGMABFA , SPP , sr , srp , xjd , xjl , xkabs , Xm , 
     &     xmin , xmo
      REAL xnor , xrefmax , xtransh , xtransl , y , y0 , ymin
      INTEGER i , j , Jmax , jmin , jrefmax , jtransh , jtransl , nc , 
     &        ncp
      REAL X(*) , Spref(*), Spinc(*)
      REAL pm1(19) , pmx(26) , ap(3)
      REAL SPINTERP
      EXTERNAL SPINTERP
 
c  precision factor for Green's function integration
      ncp = 100
c  ranges for different methods
      xmin = 2.E-4
      xtransl = 0.01957
      xtransh = 0.02348
      ymin = .03333
c  angle dependent parameters
      xmo = MAX(.05,MIN(.95,Xm))
      CALL PM1Y(xmo,pm1)
      CALL PMXY(xmo,pmx)
      ap(2) = 0.802 - 1.019*xmo + 2.528*xmo**2 - 3.198*xmo**3 + 
     &        1.457*xmo**4 + (.030/xmo)**4
      ap(3) = 0.381
      xrefmax = 1/(ymin+pmx(26))
 
      jmin = 0
      DO 100 j = 1 , Jmax
         IF ( X(j).GT.xmin ) GOTO 200
         jmin = j
 100  CONTINUE
 200  jtransl = 0
      DO 300 j = MAX(1,jmin) , Jmax
         IF ( X(j).GT.xtransl ) GOTO 400
         jtransl = j
 300  CONTINUE
 400  jtransh = 0
      DO 500 j = MAX(1,jtransl) , Jmax
         IF ( X(j).GT.xtransh ) GOTO 600
         jtransh = j
 500  CONTINUE
 600  jrefmax = 0
      DO 700 j = MAX(1,jtransh) , Jmax
         IF ( X(j).GT.xrefmax ) GOTO 800
         jrefmax = j
 700  CONTINUE
 
 800  DO 900 j = 1 , jmin
         Spref(j) = 0
 900  CONTINUE
c-----------------------
      DO 1000 j = jmin + 1 , jtransl
         xkabs = SIGMABFA(j,X,jtransh,Ab_,Ab0,xnor)
         Spref(j) = Spinc(j)*GNR(xkabs,xmo)
 1000 CONTINUE
c-----------------------
      IF ( jmin.GE.jtransl ) xkabs = SIGMABFA(1,X,MAX(1,jtransh),Ab_,
     &                               Ab0,xnor)
      ap(1) = xnor/1.21
      xjl = ALOG(xtransl)
      xjd = ALOG(xtransh) - xjl
      DO 1100 j = jtransl + 1 , jtransh
         fjc = .5*SIN(3.14159*((ALOG(X(j))-xjl)/xjd-.5))
         xkabs = SIGMABFA(j,X,jtransh,Ab_,Ab0,xnor)
         y = 1/X(j)
         Spref(j) = Spinc(j)*GNR(xkabs,xmo)
         dym = y - ymin
         dy = MIN(2.,dym)
         ddy = (dy-pmx(26))/(ncp+1)
         y0 = y - dy
         sr = SPINTERP(y0,jmax,X,Spinc)*GRXYL(pm1,pmx,dy,y0,ap)
         dy = pmx(26)
         y0 = y - dy
         sr = .5*(sr+SPINTERP(y0,jmax,X,Spinc)*GRXYL(pm1,pmx,dy,y0,ap))
         DO 1050 i = 1 , ncp
            dy = dy + ddy
            y0 = y - dy
            sr = sr + SPINTERP(y0,jmax,X,Spinc)*GRXYL(pm1,pmx,dy,y0,ap)
 1050    CONTINUE
         sr = sr*ddy
         IF ( dym.GT.2. ) THEN
            ddy = dym - 2
            nc = INT(ncp*ddy/(dym-pmx(26)))
            ddy = ddy/(nc+1)
            dy = dym
            y0 = y - dy
            srp = SPINTERP(y0,jmax,X,Spinc)*GRXYH(pm1,pmx,dy,y0,ap)
            dy = 2
            y0 = y - dy
            srp = .5*(srp+SPINTERP(y0,jmax,X,Spinc)
     &                    *GRXYH(pm1,pmx,dy,y0,ap))
            DO 1060 i = 1 , nc
               dy = dy + ddy
               y0 = y - dy
               srp = srp + SPINTERP(y0,jmax,X,Spinc)
     &                     *GRXYH(pm1,pmx,dy,y0,ap)
 1060       CONTINUE
            sr = sr + srp*ddy
         ENDIF
         Spref(j) = (.5-fjc)*Spref(j) + (.5+fjc)*sr*X(j)
 1100 CONTINUE
c-----------------------
      DO 1200 j = jtransh + 1 , jrefmax
         y = 1/X(j)
         dym = y - ymin
         dy = MIN(2.,dym)
         ddy = (dy-pmx(26))/(ncp+1)
         y0 = y - dy
         sr = SPINTERP(y0,jmax,X,Spinc)*GRXYL(pm1,pmx,dy,y0,ap)
         dy = pmx(26)
         y0 = y - dy
         sr = .5*(sr+SPINTERP(y0,jmax,X,Spinc)*GRXYL(pm1,pmx,dy,y0,ap))
         DO 1150 i = 1 , ncp
            dy = dy + ddy
            y0 = y - dy
            sr = sr + SPINTERP(y0,jmax,X,Spinc)*GRXYL(pm1,pmx,dy,y0,ap)
 1150    CONTINUE
         sr = sr*ddy
         IF ( dym.GT.2. ) THEN
            ddy = dym - 2
            nc = INT(ncp*ddy/(dym-pmx(26)))
            ddy = ddy/(nc+1)
            dy = dym
            y0 = y - dy
            srp = SPINTERP(y0,jmax,X,Spinc)*GRXYH(pm1,pmx,dy,y0,ap)
            dy = 2
            y0 = y - dy
            srp = .5*(srp+SPINTERP(y0,jmax,X,Spinc)
     &                    *GRXYH(pm1,pmx,dy,y0,ap))
            DO 1160 i = 1 , nc
               dy = dy + ddy
               y0 = y - dy
               srp = srp + SPINTERP(y0,jmax,X,Spinc)
     &                     *GRXYH(pm1,pmx,dy,y0,ap)
 1160       CONTINUE
            sr = sr + srp*ddy
         ENDIF
         Spref(j) = sr*X(j)
 1200 CONTINUE
c----------------------
      DO 1300 j = jrefmax + 1 , Jmax
         Spref(j) = 0
 1300 CONTINUE
 
      RETURN
      END

c ------------------------------------------------------------------- 

      function spinterp(invx, jmax, x, spinc)

      INTEGER jmax
      REAL spinterp, invx, x(*), spinc(*)

c Performs binary search on spinc given array x and input invx (=1/x).

      INTEGER ix, ixlo, ixhi
      REAL xtarg


      xtarg = 1./invx

      IF ( xtarg .LT. x(1) ) THEN
         spinterp = 0.
         RETURN
      ENDIF

      IF ( xtarg .GT. x(jmax) ) THEN
         spinterp = 0.
         RETURN
      ENDIF

      ixlo = 0
      ixhi = jmax + 1

      DO WHILE ( (ixhi-ixlo) .GT. 1 )

         ix = (ixhi + ixlo) / 2
         IF ( xtarg .GT. x(ix) ) THEN
            ixlo = ix
         ELSE
            ixhi = ix
         ENDIF

      ENDDO

      IF ( ixlo .NE. ixhi ) THEN
         spinterp = ( spinc(ixlo)*(x(ixhi)-xtarg) + 
     &                spinc(ixhi)*(xtarg-x(ixlo)) ) / (x(ixhi)-x(ixlo))
      ELSE
         spinterp = spinc(ixlo)
      ENDIF

      RETURN
      END


