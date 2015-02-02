import sympy.physics.units as u
from astropy import constants as const
import numpy as np
from sympy import symbols, pi, log, ln, Min, Max, sqrt, sign, lambdify, ceiling, floor, evalf

class formulae:
    @staticmethod
    def compute_derived_parameters(o, mode):
        o.Tdump = Min(o.Tdump_ref * floor(o.Bmax / o.Bmax_bin), 1.2 * u.s) # Correlator dump time; limit this at 1.2s maximum
        o.wl_max = u.c / o.freq_min             # Maximum Wavelength
        o.wl_min = u.c / o.freq_max             # Minimum Wavelength
        o.wl = 0.5*(o.wl_max + o.wl_min)          # Representative Wavelength
        o.Theta_fov = 7.66 * o.wl * o.Qfov / (pi * o.Ds * o.Nfacet) # added Nfacet dependence (c.f. PDR05 uses lambda max not mean here)
        o.Theta_beam = 3 * o.wl/(2*o.Bmax) #bmax here is for the overall experiment (so use Bmax), not the specific bin... (Consistent with PDR05 280115)
        o.Theta_pix = o.Theta_beam/(2*o.Qpix) #(ConsistenNf_outt with PDR05 280115)
        o.Npix_linear = o.Theta_fov / o.Theta_pix  # The linear number of pixels along the image's side (assumed to be square) (Consistent with PDR05 280115)
        o.Rfft = o.Nfacet**2 * 5 * o.Npix_linear**2 * log(o.Npix_linear,2) / o.Tsnap # added Nfacet dependence (Consistent with PDR05 280115, with 5 prefactor rather than 10 (late change))
        o.Qw2  = 1  # Obsolete variable (ask Rosie about what it meant)
        o.Qw32 = 1  # Obsolete variable (ask Rosie about what it meant)

        o.DeltaW_max = o.Qw * Max(o.Bmax_bin*o.Tsnap*o.Omega_E/(o.wl*2), o.Bmax_bin**2/(o.R_Earth*o.wl*8)) #W deviation catered for by W kernel, in units of typical wavelength, for the specific baseline bin being considered (Consistent with PDR05 280115, but with lambda not lambda min)
        o.Ngw = 2*o.Theta_fov * sqrt((o.Qw2 * o.DeltaW_max**2 * o.Theta_fov**2/4.0)+(o.Qw32 * o.DeltaW_max**1.5 * o.Theta_fov/(o.epsilon_w*pi*2))) #size of the support of the w kernel evaluated at maximum w (Consistent with PDR05 280115)
        o.Ncvff = o.Qgcf*sqrt(o.Naa**2+o.Ngw**2) #The total linear kernel size (Consistent with PDR05 280115)

        if mode == 'Continuum':
            o.Nf_used  = log(o.wl_max/o.wl_min) / log(3*o.wl/(2*o.Bmax_bin)/(o.Theta_fov*o.Qbw)+1) #Number of channels for gridding at longest baseline
            o.Rrp  = o.Nfacet**2 * 50 * o.Npix_linear**2 / o.Tsnap #(Consistent with PDR05 280115)
        elif mode == 'Spectral':
            o.Nf_used  = o.Nf_max
            o.Rrp  = o.Nfacet**2 * 50 * o.Npix_linear**2 / o.Tsnap #(Consistent with PDR05 280115)
        elif mode == 'SlowTrans':
            o.Nf_used  = log(o.wl_max/o.wl_min) / log(3*o.wl/(2*o.Bmax_bin)/(o.Theta_fov*o.Qbw)+1) #Number of bands for gridding at longest baseline
            o.Rrp  = 0*u.s / u.s #(Consistent with PDR05 280115)
        else:
            raise Exception()

        o.Nf_no_smear  = log(o.wl_max/o.wl_min) / log(3*o.wl/(2*o.Bmax_bin)/(o.Theta_fov*o.Qbw)+1)
        #o.Nf_vis = max(o.Nf_out, o.Nf_no_smear)

        #The following workaround is no longer needed
        o.Nf_vis=(o.Nf_out*sign(floor(o.Nf_out/o.Nf_no_smear)))+(o.Nf_no_smear*sign(floor(o.Nf_no_smear/o.Nf_out))) #Workaround to avoid recursive errors...effectively is Max(Nf_out,Nf_no_smear)

        o.Nvis = o.binfrac*o.Na*(o.Na-1)*o.Nf_vis/(2*o.Tdump) * u.s # Number of visibilities per second to be gridded (after averaging short baselines to coarser freq resolution). Note multiplication by u.s to get rid of /s
        o.Rgrid = (o.Nfacet**2 + o.Nfacet)*0.5*8*o.Nmm*o.Nvis*(o.Ngw**2+o.Naa**2) #added Nfacet dependence. Linear becuase there are Nfacet^2 facets but can integrate Nfacet times longer at gridding as fov is lower. Needs revisiting. (Consistent with PDR05 280115)

        o.Rccf = o.Nfacet**2 * 5 * o.binfrac *(o.Na-1)* o.Na * o.Nmm * o.Ncvff**2 * log(o.Ncvff,2)/(o.Tion*o.Qfcv) #reduce by multiplication by o.binfrac (RCB), add in extra multiplication by Nfacet-squared.

        o.Rphrot = 2 * o.Nmajor * o.Npp * o.Nbeam * o.Nvis * o.Nfacet**2 * 25 * sign(o.Nfacet-1)  # Last factor ensures that answer is zero if Nfacet is 1.

        # Rrp is handled in the telescope parameters file

        o.Gcorr = o.Na * (o.Na - 1) * o.Nf_max * o.Nbeam * o.Nw * o.Npp / o.Tdump  # Minimum correlator output data rate, after baseline dependent averaging
        o.Mbuf_vis = 2 * o.Mvis * o.Nbeam * o.Npp * o.Nvis * o.Tobs / u.s / 1.0e15# Note the division by u.s to get rid of pesky SI unit. Also note the factor 2 -- we have a double buffer (allowing storage of a full observation while simultaneously capturing the next)
        o.Mw_cache = o.Ngw**3 * o.Qgcf**3 * o.Nbeam * o.Nf_vis * 8
        o.Rflop = (o.Rphrot + 2 * o.Nmajor*o.Nbeam*o.Npp*(o.Nf_out*(o.Rrp + o.Rfft) + (o.Nf_vis*o.Rccf) + o.Rgrid))/1.0e15 # Overall flop rate
        o.Rio = o.Mvis * o.Nmajor * o.Nbeam * o.Npp * o.Nvis * o.Nfacet**2 / 1.0e15 #added o.Nfacet dependence

        # Split the FLOP rate into constituent parts, for plotting
        Rflop_common_factor = 2 * o.Nmajor * o.Nbeam * o.Npp / 1.0e15
        o.Rflop_grid = Rflop_common_factor * o.Rgrid
        o.Rflop_conv = Rflop_common_factor * o.Nf_vis * o.Rccf
        o.Rflop_fft  = Rflop_common_factor * o.Nf_out  * o.Rfft
        o.Rflop_proj = Rflop_common_factor * o.Nf_out  * o.Rrp
        o.Rflop_phrot = o.Rphrot/1.0e15