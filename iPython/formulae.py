import sympy.physics.units as u
from sympy import log, Min, Max, sqrt, sign, ceiling, floor
from numpy import pi
from parameter_definitions import ImagingModes

class Formulae:
    def __init__(self):
        pass

    @staticmethod
    def compute_derived_parameters(telescope_parameters, imaging_mode):
        """
        Computes a host of important values from the originally supplied telescope parameters, using the parametric
        equations. These equations are based on the PDR05 document
        @param o:
        @param imaging_mode:
        @raise Exception:
        """
        o = telescope_parameters  # Used for shorthand in the equations below

        o.wl_max = u.c / o.freq_min             # Maximum Wavelength
        o.wl_min = u.c / o.freq_max             # Minimum Wavelength
        o.wl = 0.5*(o.wl_max + o.wl_min)          # Representative Wavelength
        o.Theta_fov = 7.66 * o.wl * o.Qfov / (pi * o.Ds * o.Nfacet) # added Nfacet dependence (c.f. PDR05 uses lambda max not mean here)
        o.Theta_beam = 3 * o.wl/(2*o.Bmax) #bmax here is for the overall experiment (so use Bmax), not the specific bin... (Consistent with PDR05 280115)
        o.Theta_pix = o.Theta_beam/(2*o.Qpix) #(ConsistenNf_out with PDR05 280115)
        o.Npix_linear = o.Theta_fov / o.Theta_pix  # The linear number of pixels along the image's side (assumed to be square) (Consistent with PDR05 280115)
        o.Rfft = o.Nfacet**2 * 5 * o.Npix_linear**2 * log(o.Npix_linear,2) / o.Tsnap # added Nfacet dependence (Consistent with PDR05 280115, with 5 prefactor rather than 10 (late change; somewhat controvertial and in need of review after PDR dicsussions re. Hermiticity))

        o.DeltaW_max = o.Qw * Max(o.Bmax_bin*o.Tsnap*o.Omega_E/(o.wl*2), o.Bmax_bin**2/(o.R_Earth*o.wl*8)) #W deviation catered for by W kernel, in units of typical wavelength, for the specific baseline bin being considered (Consistent with PDR05 280115, but with lambda not lambda min)
        o.Ngw = 2*o.Theta_fov * sqrt((o.DeltaW_max**2 * o.Theta_fov**2/4.0)+(o.DeltaW_max**1.5 * o.Theta_fov/(o.epsilon_w*pi*2))) #size of the support of the w kernel evaluated at maximum w (Consistent with PDR05 280115)
        o.Ncvff = o.Qgcf*sqrt(o.Naa**2+o.Ngw**2) #The total linear kernel size (Consistent with PDR05 280115)
        o.Nf_no_smear = log(o.wl_max/o.wl_min) / log(3*o.wl/(2*o.Bmax_bin)/(o.Theta_fov*o.Qbw)+1)
        o.epsilon_f_approx = sqrt(6*(1-(1.0/o.amp_f_max))) #first order expansion of sin used here to solve epsilon = arcsinc(1/amp_f_max). Checked as valid for amp_f_max 1.001, 1.01, 1.02. 1% error at amp_f_max=1.03 anticipated. See Skipper memo (REF needed)
        o.Tdump_skipper = o.epsilon_f_approx * o.wl/(o.Theta_fov * o.Nfacet * o.Omega_E * o.Bmax_bin) * u.s #multiply theta_fov by Nfacet so averaging time is set by total field of view, not faceted FoV. See Skipper memo (REF needed).
        o.Tdump = Min(o.Tdump_skipper, 1.2 * u.s) # Visibility integration time; limit this at 1.2s maximum.

        if imaging_mode == ImagingModes.Continuum:
            o.Rrp  = o.Nfacet**2 * 50 * o.Npix_linear**2 / o.Tsnap # Reprojection Flop rate, per output channel (Consistent with PDR05 280115)
        elif imaging_mode == ImagingModes.Spectral:
            o.Nf_out  = o.Nf_max    # TODO: Not sure if this is the correct expression
            o.Rrp  = o.Nfacet**2 * 50 * o.Npix_linear**2 / o.Tsnap # Reprojection Flop rate, per output channel (Consistent with PDR05 280115)
        elif imaging_mode == ImagingModes.SlowTrans:
            o.Rrp  = 0*u.s / u.s #(Consistent with PDR05 280115)
        else:
            raise Exception("Unknown Imaging Mode %s" % imaging_mode)

        # Workaround to avoid recursive errors...effectively is Max(Nf_out,Nf_no_smear). TODO: fix this.
        o.Nf_vis=(o.Nf_out*sign(floor(o.Nf_out/o.Nf_no_smear)))+(o.Nf_no_smear*sign(floor(o.Nf_no_smear/o.Nf_out)))

        o.Nvis = o.binfrac*o.Na*(o.Na-1)*o.Nf_vis/(2*o.Tdump) * u.s # Number of visibilities per second to be gridded (after averaging short baselines to coarser freq resolution). Note multiplication by u.s to get rid of /s
        o.Rgrid = (o.Nfacet**2 + o.Nfacet)*0.5*8*o.Nmm*o.Nvis*(o.Ngw**2+o.Naa**2) #added Nfacet dependence. Linear becuase there are Nfacet^2 facets but can integrate Nfacet times longer at gridding as fov is lower. Needs revisiting. (Consistent with PDR05 280115)

        o.Rccf = o.Nfacet**2 * 5 * o.binfrac *(o.Na-1)* o.Na * o.Nmm * o.Ncvff**2 * log(o.Ncvff,2)/(o.Tion*o.Qfcv) #reduce by multiplication by o.binfrac (RCB), add in extra multiplication by Nfacet-squared.

        o.Rphrot = 2 * o.Nmajor * o.Npp * o.Nbeam * o.Nvis * o.Nfacet**2 * 25 * sign(o.Nfacet-1)  # Last factor ensures that answer is zero if Nfacet is 1.

        o.Gcorr = o.Na * (o.Na - 1) * o.Nf_vis * o.Nbeam * o.Nw * o.Npp / o.Tdump  # Minimum correlator output data rate, after baseline dependent averaging (THINK THIS IS REDUNDANT)
        o.Mbuf_vis = 2 * o.Mvis * o.Nbeam * o.Npp * o.Nvis * o.Tobs / u.s # Note the division by u.s to get rid of pesky SI unit. Also note the factor 2 -- we have a double buffer (allowing storage of a full observation while simultaneously capturing the next)
        o.Mw_cache = o.Ngw**3 * o.Qgcf**3 * o.Nbeam * o.Nf_vis * 8
        o.Rio = o.Mvis * (o.Nmajor+1) * o.Nbeam * o.Npp * o.Nvis * o.Nfacet**2 #added o.Nfacet dependence; changed Nmajor factor to Nmajor+1 as part of post PDR fixes.

        # Split the FLOP rate into constituent parts, for plotting
        Rflop_common_factor = 2 * o.Nmajor * o.Nbeam * o.Npp
        o.Rflop_grid = Rflop_common_factor * o.Rgrid
        o.Rflop_conv = Rflop_common_factor * o.Nf_vis * o.Rccf
        o.Rflop_fft  = Rflop_common_factor * o.Nf_out * o.Rfft
        o.Rflop_proj = Rflop_common_factor * o.Nf_out * o.Rrp
        o.Rflop_phrot = o.Rphrot

        o.Rflop = o.Rflop_phrot + o.Rflop_proj + o.Rflop_fft + o.Rflop_conv + o.Rflop_grid  # Overall flop rate
