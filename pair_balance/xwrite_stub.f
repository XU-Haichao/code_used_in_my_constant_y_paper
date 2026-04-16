      SUBROUTINE xwrite(msg, level)
      IMPLICIT NONE

      CHARACTER*(*) msg
      INTEGER level

      WRITE(*,*) msg
      RETURN
      END
