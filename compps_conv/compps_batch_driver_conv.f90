program compps_batch_driver_conv
  implicit none
  include 'mscomppsq_free.inc'
  include 'mscomppsj_free.inc'

  integer, parameter :: ne = 260
  integer, parameter :: ne_seed = 900
  integer, parameter :: n_tau = 3
  integer, parameter :: n_theta = 2
  integer, parameter :: n_injection = 2
  integer, parameter :: n_target_angle = 3
  integer, parameter :: n_angle_bin = 3
  integer, parameter :: n_dense_mu = 256
  integer, parameter :: max_scatter_override_value = 800
  double precision, parameter :: mec2_kev = 511.0d0
  double precision, parameter :: seed_kT_kev = 5.0d-3
  double precision, parameter :: pi = 3.1415926535897932384626433832795d0
  double precision, parameter :: cmin = 1.0d-300
  double precision, parameter :: tau_input_scale = 1.0d0
  double precision, parameter :: cov_fac_value = 1.0d0

  double precision :: tau_values(n_tau), theta_values(n_theta)
  double precision :: target_theta_deg(n_target_angle), target_mu(n_target_angle)
  double precision :: bin_theta_low(n_angle_bin), bin_theta_high(n_angle_bin)
  double precision :: bin_mu_low(n_angle_bin), bin_mu_high(n_angle_bin)
  double precision :: bin_solid_angle(n_angle_bin)
  character(len=32) :: injection_names(n_injection)
  double precision :: injection_geom(n_injection)

  double precision :: ekev(ne), eev(ne), obj(ne), parm(10)
  double precision :: phcon(ne), phblb(ne), phref(ne), phnorm
  double precision :: e_seed(0:ne_seed), spec_seed(ne_seed)
  double precision :: up_int(ne), down_int(ne), total_int(ne)
  double precision :: up_target(ne), down_target(ne)
  double precision :: up_bin(ne), down_bin(ne), total_bin(ne)
  double precision :: log10emin, log10emax, loge
  integer :: i, itau, ith, inj, ia, ibin
  integer :: maxsc_override
  logical :: flagfre, flagspnc
  character(len=256) :: outdir

  common /MSMAXSCOVR/ maxsc_override

  outdir = 'compps_run/output_bb5ev_slab'
  call execute_command_line('mkdir -p ' // trim(outdir))

  tau_values = (/ 0.1d0, 1.0d0, 5.0d0 /)
  theta_values = (/ 0.1d0, 0.5d0 /)
  injection_names = (/ character(len=32) :: 'lambert', 'internal_iso' /)
  injection_geom = (/ 1.0d0, -1.0d0 /)
  target_theta_deg = (/ 0.0d0, 30.0d0, 60.0d0 /)
  do i = 1, n_target_angle
     target_mu(i) = dcos(target_theta_deg(i) * pi / 180.0d0)
  end do
  bin_theta_low = (/ 0.0d0, 30.0d0, 60.0d0 /)
  bin_theta_high = (/ 30.0d0, 60.0d0, 90.0d0 /)
  do i = 1, n_angle_bin
     bin_mu_low(i) = dcos(bin_theta_high(i) * pi / 180.0d0)
     bin_mu_high(i) = dcos(bin_theta_low(i) * pi / 180.0d0)
     bin_solid_angle(i) = 2.0d0 * pi * (bin_mu_high(i) - bin_mu_low(i))
  end do

  log10emin = -4.0d0
  log10emax = 5.6989700043360187d0
  do i = 1, ne
     loge = log10emin + (log10emax - log10emin) * dble(i - 1) / dble(ne - 1)
     ekev(i) = 10.0d0**loge
     eev(i) = 1.0d3 * ekev(i)
     obj(i) = ekev(i) / mec2_kev
  end do
  call build_blackbody_seed_grid(e_seed, spec_seed)

  maxsc_override = max_scatter_override_value
  call write_metadata(trim(outdir) // '/compps_slab_bb5ev_metadata.csv')

  open(unit=31, file=trim(outdir) // '/compps_slab_bb5ev_angle_integrated_EdNdE.csv', status='replace')
  write(31,'(A)') 'injection_mode,theta_e,tau,E_keV,E_eV,up_dNdE,down_dNdE,total_dNdE,up_EdNdE,down_EdNdE,total_EdNdE'

  open(unit=32, file=trim(outdir) // '/compps_slab_bb5ev_angle_resolved_EdNdE_dOmega.csv', status='replace')
  write(32,'(A)') 'injection_mode,theta_e,tau,boundary,theta_deg,mu,E_keV,E_eV,dNdE_dOmega,EdNdE_dOmega'

  open(unit=33, file=trim(outdir) // '/compps_slab_bb5ev_lambert_tau1_theta0p1_angle_bin_integrated_EdNdE.csv', status='replace')
  write(33,'(A)') 'theta_low_deg,theta_high_deg,mu_low,mu_high,solid_angle_sr,boundary,E_keV,E_eV,dNdE_integrated,EdNdE_integrated'

  do inj = 1, n_injection
     do ith = 1, n_theta
        do itau = 1, n_tau
           call run_case(theta_values(ith), tau_input_scale * tau_values(itau), injection_geom(inj), &
                         obj, ne, parm, phcon, phblb, phref, phnorm, e_seed, spec_seed, ne_seed)
           call integrate_dense_mu_angles(obj, ne, up_int, down_int, total_int)

           do i = 1, ne
              write(31,'(A,",",ES24.16,",",ES24.16,",",8(ES24.16,:,","))') trim(injection_names(inj)), &
                   theta_values(ith), tau_values(itau), ekev(i), eev(i), up_int(i), down_int(i), total_int(i), &
                   eev(i) * up_int(i), eev(i) * down_int(i), eev(i) * total_int(i)
           end do

           if (dabs(theta_values(ith) - 0.1d0) .lt. 1d-12 .and. dabs(tau_values(itau) - 1.0d0) .lt. 1d-12) then
              do ia = 1, n_target_angle
                 call remap_boundary_at_mu(obj, ne, target_mu(ia), 1, up_target)
                 call remap_boundary_at_mu(obj, ne, target_mu(ia), 2, down_target)
                 do i = 1, ne
                    write(32,'(A,",",ES24.16,",",ES24.16,",",A,",",7(ES24.16,:,","))') trim(injection_names(inj)), &
                         theta_values(ith), tau_values(itau), 'up', target_theta_deg(ia), target_mu(ia), &
                         ekev(i), eev(i), up_target(i), eev(i) * up_target(i)
                    write(32,'(A,",",ES24.16,",",ES24.16,",",A,",",7(ES24.16,:,","))') trim(injection_names(inj)), &
                         theta_values(ith), tau_values(itau), 'down', target_theta_deg(ia), target_mu(ia), &
                         ekev(i), eev(i), down_target(i), eev(i) * down_target(i)
                 end do
              end do

              if (trim(injection_names(inj)) .eq. 'lambert') then
                 do ibin = 1, n_angle_bin
                    call integrate_mu_interval(obj, ne, bin_mu_low(ibin), bin_mu_high(ibin), 1, up_bin)
                    call integrate_mu_interval(obj, ne, bin_mu_low(ibin), bin_mu_high(ibin), 2, down_bin)
                    total_bin = up_bin + down_bin
                    do i = 1, ne
                       write(33,'(5(ES24.16,","),A,",",4(ES24.16,:,","))') bin_theta_low(ibin), bin_theta_high(ibin), &
                            bin_mu_low(ibin), bin_mu_high(ibin), bin_solid_angle(ibin), 'up', ekev(i), eev(i), &
                            up_bin(i), eev(i) * up_bin(i)
                       write(33,'(5(ES24.16,","),A,",",4(ES24.16,:,","))') bin_theta_low(ibin), bin_theta_high(ibin), &
                            bin_mu_low(ibin), bin_mu_high(ibin), bin_solid_angle(ibin), 'down', ekev(i), eev(i), &
                            down_bin(i), eev(i) * down_bin(i)
                       write(33,'(5(ES24.16,","),A,",",4(ES24.16,:,","))') bin_theta_low(ibin), bin_theta_high(ibin), &
                            bin_mu_low(ibin), bin_mu_high(ibin), bin_solid_angle(ibin), 'total', ekev(i), eev(i), &
                            total_bin(i), eev(i) * total_bin(i)
                    end do
                 end do
              end if
           end if
        end do
     end do
  end do

  close(31)
  close(32)
  close(33)

  call execute_command_line('mkdir -p compps_run/output_bb5ev_slab_truncdiag')
  maxsc_override = 2000
  call run_case(0.5d0, tau_input_scale * 5.0d0, -1.0d0, &
                obj, ne, parm, phcon, phblb, phref, phnorm, e_seed, spec_seed, ne_seed)
  call write_native_angle_resolved_spectrum( &
       'compps_run/output_bb5ev_slab_truncdiag/compps_internal_iso_theta0p5_tau5_native_angle_resolved_maxsc2000.csv', &
       0.5d0, 5.0d0, 2000)
  call write_native_angle_integrated_spectrum( &
       'compps_run/output_bb5ev_slab_truncdiag/compps_internal_iso_theta0p5_tau5_native_angle_integrated_maxsc2000.csv', &
       0.5d0, 5.0d0, 2000)
  maxsc_override = max_scatter_override_value

  print *, 'Wrote compPS-conv spectra to ', trim(outdir)

contains

  subroutine build_blackbody_seed_grid(edges, spec)
    implicit none
    double precision, intent(out) :: edges(0:ne_seed), spec(ne_seed)
    double precision :: logemin, logemax, ec, width, x
    integer :: i
    logemin = -8.0d0
    logemax = 2.0d0
    do i = 0, ne_seed
       edges(i) = 10.0d0**(logemin + (logemax - logemin) * dble(i) / dble(ne_seed))
    end do
    do i = 1, ne_seed
       ec = dsqrt(edges(i - 1) * edges(i))
       width = edges(i) - edges(i - 1)
       x = ec / seed_kT_kev
       if (x .gt. 700.0d0) then
          spec(i) = 0.0d0
       else if (x .lt. 1.0d-8) then
          spec(i) = ec * seed_kT_kev * width
       else
          spec(i) = ec * ec / (dexp(x) - 1.0d0) * width
       end if
    end do
  end subroutine build_blackbody_seed_grid

  subroutine run_case(theta_e, tau, geom, obj, ne, parm, phcon, phblb, phref, phnorm, e_seed, spec_seed, ne_seed)
    implicit none
    integer, intent(in) :: ne, ne_seed
    double precision, intent(in) :: theta_e, tau, geom, obj(ne), e_seed(0:ne_seed), spec_seed(ne_seed)
    double precision, intent(out) :: parm(10), phcon(ne), phblb(ne), phref(ne), phnorm
    logical :: flagfre, flagspnc
    parm = 0.0d0
    parm(1) = theta_e * mec2_kev
    parm(2) = 2.0d0
    parm(3) = -1.0d0
    parm(4) = 1000.0d0
    parm(5) = tau
    parm(6) = geom
    parm(7) = 1.0d0
    parm(8) = 0.5d0
    parm(9) = cov_fac_value
    parm(10) = 0.0d0
    flagfre = .true.
    flagspnc = .true.
    call MSISMCO(parm, obj, phcon, phblb, phref, ne, phnorm, flagfre, flagspnc, spec_seed, e_seed, ne_seed)
  end subroutine run_case

  subroutine integrate_dense_mu_angles(obj, ne, up_int, down_int, total_int)
    implicit none
    integer, intent(in) :: ne
    double precision, intent(in) :: obj(ne)
    double precision, intent(out) :: up_int(ne), down_int(ne), total_int(ne)
    double precision :: up_spec(ne), down_spec(ne), mu, weight
    integer :: imu, i
    up_int = 0.0d0
    down_int = 0.0d0
    weight = 1.0d0 / dble(n_dense_mu)
    do imu = 1, n_dense_mu
       mu = (dble(imu) - 0.5d0) / dble(n_dense_mu)
       call remap_boundary_at_mu(obj, ne, mu, 1, up_spec)
       call remap_boundary_at_mu(obj, ne, mu, 2, down_spec)
       do i = 1, ne
          up_int(i) = up_int(i) + weight * up_spec(i)
          down_int(i) = down_int(i) + weight * down_spec(i)
       end do
    end do
    up_int = 2.0d0 * pi * up_int
    down_int = 2.0d0 * pi * down_int
    total_int = up_int + down_int
  end subroutine integrate_dense_mu_angles

  subroutine write_native_angle_resolved_spectrum(path, theta_e, tau_report, max_scatter)
    implicit none
    include 'mscomppsq_free.inc'
    character(len=*), intent(in) :: path
    double precision, intent(in) :: theta_e, tau_report
    integer, intent(in) :: max_scatter
    double precision :: mu, theta_deg, dnde, ednde
    integer :: ix, ia, k, boundary
    character(len=8) :: boundary_name

    open(unit=41, file=path, status='replace')
    write(41,'(A)') 'max_scatter,max_tau_grid,max_angle_grid,injection_mode,theta_e,tau,boundary,angle_index,mu,theta_deg,mu_weight,E_keV,E_eV,dNdE_dOmega_native,EdNdE_dOmega_native'
    do boundary = 1, 2
       if (boundary .eq. 1) then
          boundary_name = 'up'
       else
          boundary_name = 'down'
       end if
       do ia = 1, MAXANG
          mu = UANG(ia)
          theta_deg = dacos(mu) * 180.0d0 / pi
          do ix = 1, MAXFRE
             k = ia + (ix - 1) * MAXANG
             if (boundary .eq. 1) then
                dnde = (SUIPL(k) + DINTPL(k)) * mu / XEN(ix)
             else
                dnde = (SUIMI(k) + DINTMI(k)) * mu / XEN(ix)
             end if
             ednde = XKEV(ix) * 1.0d3 * dnde
             write(41,'(I0,",",I0,",",I0,",",A,",",ES24.16,",",ES24.16,",",A,",",I0,",",7(ES24.16,:,","))') &
                  max_scatter, MAXTAU, MAXANG, 'internal_iso', theta_e, tau_report, trim(boundary_name), ia, &
                  mu, theta_deg, AANG(ia), XKEV(ix), XKEV(ix) * 1.0d3, dnde, ednde
          end do
       end do
    end do
    close(41)
  end subroutine write_native_angle_resolved_spectrum

  subroutine write_native_angle_integrated_spectrum(path, theta_e, tau_report, max_scatter)
    implicit none
    include 'mscomppsq_free.inc'
    character(len=*), intent(in) :: path
    double precision, intent(in) :: theta_e, tau_report
    integer, intent(in) :: max_scatter
    double precision :: up_dnde, down_dnde, up_ednde, down_ednde, mu
    integer :: ix, ia, k

    open(unit=42, file=path, status='replace')
    write(42,'(A)') 'max_scatter,max_tau_grid,max_angle_grid,injection_mode,theta_e,tau,E_keV,E_eV,up_dNdE_native,down_dNdE_native,total_dNdE_native,up_EdNdE_native,down_EdNdE_native,total_EdNdE_native'
    do ix = 1, MAXFRE
       up_dnde = 0.0d0
       down_dnde = 0.0d0
       do ia = 1, MAXANG
          mu = UANG(ia)
          k = ia + (ix - 1) * MAXANG
          up_dnde = up_dnde + AANG(ia) * (SUIPL(k) + DINTPL(k)) * mu / XEN(ix)
          down_dnde = down_dnde + AANG(ia) * (SUIMI(k) + DINTMI(k)) * mu / XEN(ix)
       end do
       up_dnde = 2.0d0 * pi * up_dnde
       down_dnde = 2.0d0 * pi * down_dnde
       up_ednde = XKEV(ix) * 1.0d3 * up_dnde
       down_ednde = XKEV(ix) * 1.0d3 * down_dnde
       write(42,'(I0,",",I0,",",I0,",",A,",",ES24.16,",",ES24.16,",",8(ES24.16,:,","))') &
            max_scatter, MAXTAU, MAXANG, 'internal_iso', theta_e, tau_report, XKEV(ix), XKEV(ix) * 1.0d3, &
            up_dnde, down_dnde, up_dnde + down_dnde, up_ednde, down_ednde, up_ednde + down_ednde
    end do
    close(42)
  end subroutine write_native_angle_integrated_spectrum

  subroutine remap_boundary_at_mu(obj, ne, target_mu, boundary, out)
    implicit none
    include 'mscomppsq_free.inc'
    integer, intent(in) :: ne, boundary
    double precision, intent(in) :: obj(ne), target_mu
    double precision, intent(out) :: out(ne)
    double precision :: ds(MAXFRE), y2(MAXFRE), y
    integer :: i
    call build_log_spectrum_at_mu(target_mu, boundary, ds)
    call MSQSPLINE(XLOG, ds, MAXFRE, 1.0d30, 1.0d30, y2)
    do i = 1, ne
       if (obj(i) .lt. XEN(1) .or. obj(i) .ge. XEN(MAXFRE)) then
          out(i) = 0.0d0
       else
          call MSQSPLINT(XLOG, ds, y2, MAXFRE, dlog10(obj(i)), y)
          out(i) = dexp(y) / obj(i)
       end if
    end do
  end subroutine remap_boundary_at_mu

  subroutine build_log_spectrum_at_mu(target_mu, boundary, ds)
    implicit none
    include 'mscomppsq_free.inc'
    integer, intent(in) :: boundary
    double precision, intent(in) :: target_mu
    double precision, intent(out) :: ds(MAXFRE)
    double precision :: fint(MAXANG), y, polerr, val
    integer :: ix, ia, k
    do ix = 1, MAXFRE
       do ia = 1, MAXANG
          k = ia + (ix - 1) * MAXANG
          if (boundary .eq. 1) then
             val = (SUIPL(k) + DINTPL(k)) * UANG(ia)
          else
             val = (SUIMI(k) + DINTMI(k)) * UANG(ia)
          end if
          fint(ia) = dlog(dmax1(val, cmin))
       end do
       call MSPOLINTQ(UANG, fint, MAXANG, target_mu, y, polerr)
       ds(ix) = y
    end do
  end subroutine build_log_spectrum_at_mu

  subroutine integrate_mu_interval(obj, ne, mu_low, mu_high, boundary, out)
    implicit none
    integer, intent(in) :: ne, boundary
    double precision, intent(in) :: obj(ne), mu_low, mu_high
    double precision, intent(out) :: out(ne)
    integer, parameter :: nq = 16
    double precision :: x(nq), w(nq), spec(ne), mu, half_width, mid
    integer :: q
    call gauss_legendre_0_1_16(x, w)
    out = 0.0d0
    half_width = 0.5d0 * (mu_high - mu_low)
    mid = 0.5d0 * (mu_high + mu_low)
    do q = 1, nq
       mu = mid + half_width * (2.0d0 * x(q) - 1.0d0)
       call remap_boundary_at_mu(obj, ne, mu, boundary, spec)
       out = out + w(q) * spec
    end do
    out = 2.0d0 * pi * (mu_high - mu_low) * out
  end subroutine integrate_mu_interval

  subroutine gauss_legendre_0_1_16(x, w)
    implicit none
    double precision, intent(out) :: x(16), w(16)
    x = (/ &
      5.299532504175033d-3, 2.771245750426802d-2, 6.673604915905862d-2, 1.220297844498134d-1, &
      1.910618777986780d-1, 2.709916111713863d-1, 3.591982246103705d-1, 4.526937451081222d-1, &
      5.473062548918778d-1, 6.408017753896295d-1, 7.290083888286137d-1, 8.089381222013220d-1, &
      8.779702155501866d-1, 9.332639508409414d-1, 9.722875424957320d-1, 9.947004674958250d-1 /)
    w = (/ &
      1.357622970587704d-2, 3.112676196932394d-2, 4.757925584124643d-2, 6.231448562776694d-2, &
      7.479799440828837d-2, 8.457825969750126d-2, 9.130170752246180d-2, 9.472530522753424d-2, &
      9.472530522753424d-2, 9.130170752246180d-2, 8.457825969750126d-2, 7.479799440828837d-2, &
      6.231448562776694d-2, 4.757925584124643d-2, 3.112676196932394d-2, 1.357622970587704d-2 /)
  end subroutine gauss_legendre_0_1_16

  subroutine write_metadata(path)
    implicit none
    character(len=*), intent(in) :: path
    open(unit=40, file=path, status='replace')
    write(40,'(A)') 'key,value'
    write(40,'(A)') 'source,compps_conv'
    write(40,'(A)') 'geometry,slab'
    write(40,'(A)') 'seed_spectrum,blackbody'
    write(40,'(A,ES24.16)') 'seed_kT_keV,', seed_kT_kev
    write(40,'(A,ES24.16)') 'tau_input_scale,', tau_input_scale
    write(40,'(A,ES24.16)') 'cov_fac,', cov_fac_value
    write(40,'(A,I0)') 'max_scatter_override,', max_scatter_override_value
    write(40,'(A,I0)') 'dense_mu_points,', n_dense_mu
    write(40,'(A,I0)') 'max_tau_grid_points,', MAXTAU
    write(40,'(A,I0)') 'max_frequency_points,', MAXFRE
    write(40,'(A,ES24.16)') 'seed_kT_eV,', seed_kT_kev * 1.0d3
    write(40,'(A,I0)') 'energy_points,', ne
    write(40,'(A)') 'angle_integrated_grid_tau,0.1;1;5'
    write(40,'(A)') 'angle_integrated_grid_theta_e,0.1;0.5'
    write(40,'(A)') 'injection_modes,lambert;internal_iso'
    write(40,'(A)') 'angle_resolved_case,tau=1 theta_e=0.1 theta=0;30;60 deg'
    write(40,'(A)') 'lambert_angle_bins,tau=1 theta_e=0.1 bins=0to30;30to60;60to90 deg'
    close(40)
  end subroutine write_metadata

end program compps_batch_driver_conv

subroutine xwrite(message, level)
  implicit none
  character(len=*), intent(in) :: message
  integer, intent(in) :: level
  if (level .ge. 5) then
     write(*,'(A)') trim(message)
  end if
end subroutine xwrite
