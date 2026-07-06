      SUBROUTINE xwrite(msg, level)
      IMPLICIT NONE

      CHARACTER*(*) msg
      INTEGER level

      IF (level .GE. 5) WRITE(*,*) msg
      RETURN
      END
