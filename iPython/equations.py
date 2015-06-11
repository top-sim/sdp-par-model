"""
This class contains the actual equations that are used to compute the telescopes' performance values and computional
requirements from the supplied basic parameters defined in ParameterDefinitions.

This class contains a method for (symbolically) computing derived parameters using imaging equations described
in PDR05 (version 1.85).
"""

from sympy import log, Min, Max, sqrt, floor, sign
from numpy import pi
from parameter_definitions import ImagingModes
from parameter_definitions import ParameterContainer

class Equations:
    def __init__(self):
        pass

    @staticmethod
    def apply_imaging_equations(telescope_parameters, imaging_mode, bl_dep_time_av, on_the_fly=False, verbose=False):
        """
        (Symbolically) computes a set of derived parameters using imaging equations described in PDR05 (version 1.85).

        The derived parameters are added to the supplied telescope_parameter object (locally referred to as o).
        Where parameters are only described symbolically (using sympy) they can be numerically evaluated at a later
        stage, when unknown symbolic variables are suitably substituted. This is typically done in external code
        contained, e.g. in implementation.py.

        @param telescope_parameters: ParameterContainer object containing the telescope parameters.
               This ParameterContainer object is modified in-place by appending / overwriting the relevant fields
        @param imaging_mode: The telecope's imaging mode
        @param bl_dep_time_av: True iff baseline dependent time averaging should be used.
        @param on_the_fly: True iff using on-the-fly kernels
        @param verbose: displays verbose command-line output
        @raise Exception:
        """
        o = telescope_parameters  # Used for shorthand in the equations below
        assert isinstance(o, ParameterContainer)
        assert hasattr(o, "c")  # Checks initialization by proxy of whether the speed of light is defined

        o.wl_max = o.c / o.freq_min  # Maximum Wavelength
        o.wl_min = o.c / o.freq_max  # Minimum Wavelength
        o.wl = 0.5 * (o.wl_max + o.wl_min)  # Representative Wavelength

        # ===============================================================================================
        # PDR05 (version 1.85) Sec 9.2
        # ===============================================================================================

        # TODO: In line below: PDR05 uses *wl_max* instead of wl. Also uses 7.6 instead of 7.66. Is this correct?
        o.Theta_fov = 7.66 * o.wl * o.Qfov / (pi * o.Ds * o.Nfacet)  # Eq 6 - Facet Field-of-view
        # TODO: In the two lines below, PDR05 uses *wl_min* instead of wl
        o.Theta_beam = 3 * o.wl / (2. * o.Bmax)     # Synthesized beam. Called Theta_PSF in PDR05.
        o.Theta_pix = o.Theta_beam / (2. * o.Qpix)  # Eq 7 - Pixel size
        o.Npix_linear = o.Theta_fov / o.Theta_pix   # Eq 8 - Number of pixels on side of facet
        o.epsilon_f_approx = sqrt(6 * (1 - (1. / o.amp_f_max)))  # expansion of sine solves eps = arcsinc(1/amp_f_max).
        o.Qbw = 1.47 / o.epsilon_f_approx  # Equation nr?

        if verbose:
            print "Image Characteristics:"
            print "----------------------\n"
            print "Facet FOV: ", o.Theta_fov, " rads"
            print "PSF size:  ", o.Theta_beam, " rads"
            print "Pixel size:", o.Theta_pix, " rads"
            print "No. pixels on facet side:", o.Npix_linear
            print "Epsilon approx :", o.epsilon_f_approx
            print "Found Qbw = %8.3f, and cell frac error, epsilon,  %8.3f" % (o.Qbw, o.epsilon_f_approx)
            print "\n---------------------\n"

        # ===============================================================================================
        # PDR05 Sec 9.1
        # ===============================================================================================

        log_wl_ratio = log(o.wl_max / o.wl_min)
        # The two equations below => combination of Eq 4 and Eq 5 for full and facet FOV at max baseline respectively.
        # These limit bandwidth smearing to within a fraction (epsilon_f_approx) of a uv cell.
        # Done: PDR05 Eq 5 says o.Nf = log_wl_ratio / (1 + 0.6 * o.Ds / (o.Bmax * o.Q_fov * o.Qbw)). This is fine - substituting in the equation for theta_fov shows it is indeed correct.
        o.Nf_no_smear_predict =  log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax * o.Theta_fov * o.Qbw * o.Nfacet)))
        o.Nf_no_smear_backward = log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax_bin * o.Theta_fov * o.Qbw)))

        # correlator output averaging time scaled for max baseline.
        o.Tdump_scaled = o.Tdump_ref * o.B_dump_ref / o.Bmax
        o.combine_time_samples = Max(
            floor(o.epsilon_f_approx * o.wl / (o.Theta_fov * o.Nfacet * o.Omega_E * o.Bmax_bin * o.Tdump_scaled)), 1.)
        o.Tdump_skipper = o.Tdump_scaled * o.combine_time_samples

        if bl_dep_time_av:
            # Don't let any bl-dependent time averaging be for longer than either 1.2s or Tion.
            o.Tdump_predict = Min(o.Tdump_skipper, 1.2, o.Tion)
            # For backward step at gridding only, allow coalescance of visibility points at Facet FoV
            # smearing limit only for BLDep averaging case.
            o.Tdump_backward = Min(o.Tdump_skipper * o.Nfacet, o.Tion)
        else:
            o.Tdump_predict = o.Tdump_scaled
            o.Tdump_backward = o.Tdump_scaled

        if verbose:
            print "Channelization Characteristics:"
            print "-------------------------------\n"
            print "Ionospheric timescale: ", o.Tion, " sec"
            print "T_dump predict: ", o.Tdump_predict, " sec"
            print "T_dump backward: ", o.Tdump_backward, " sec"
            print ""
            print "No. freq channels for predict: ", o.Nf_no_smear_predict
            print "No. freq channels for backward step: ", o.Nf_no_smear_backward
            print ""
            if bl_dep_time_av:
                print "USING BASELINE DEPENDENT TIME AVERAGING, combining %g time samples: " % o.combine_time_samples
            else:
                print "NOT IMPLEMENTING BASELINE DEPENDENT TIME AVERAGING"
            if on_the_fly:
                print "On-the-fly kernels..."
            else:
                print "Not using on-the-fly kernels..."
            print "\n------------------------------\n"

        # ===============================================================================================
        # PDR05 Sec 12.2 - 12.5
        # ===============================================================================================

        o.DeltaW_Earth = o.Bmax_bin ** 2 / (8. * o.R_Earth * o.wl)  # Eq. 19
        # TODO: in the two lines below, PDR05 uses lambda_min, not mean.
        o.DeltaW_SShot = o.Bmax_bin * o.Omega_E * o.Tsnap / (2. * o.wl) # Eq. 26 : W-deviation for snapshot.
        o.DeltaW_max = o.Qw * Max(o.DeltaW_SShot, o.DeltaW_Earth)
        # w-kernel support size **Note difference in cellsize assumption**
        o.Ngw = 2 * o.Theta_fov * sqrt((o.DeltaW_max * o.Theta_fov / 2.) ** 2 +
                                       (o.DeltaW_max ** 1.5 * o.Theta_fov / (2 * pi * o.epsilon_w)))  # Eq. 25

        # The following two TODO lines were added by Anna and/or Rosie
        # TODO: Test what happens if we calculate convolution functions on the fly for each visibility?
        # TODO: Make smaller kernels, no reuse.
        Nkernel2 = o.Ngw ** 2 + o.Naa ** 2  # squared linear size of combined W and A kernels; used in eqs 23 and 32
        o.Ncvff = sqrt(Nkernel2)  # Partially Eq. 23 : combined kernel support size
        if not on_the_fly:
            o.Ncvff *= o.Qgcf  # Completes Eq 23.

        o.Nf_vis_backward = Max(o.Nf_out, o.Nf_no_smear_backward)
        o.Nf_vis_predict = Max(o.Nf_out, o.Nf_no_smear_predict)

        if verbose:
            print "Geometry Assumptions:"
            print "-------------------------------"
            print ""
            print "Delta W Earth: ", o.DeltaW_Earth, " lambda"
            print "Delta W Snapshot: ", o.DeltaW_SShot, " lambda"
            print "Delta W max: ", o.DeltaW_max, " lambda"
            print ""
            print "------------------------------"
            print ""
            print "Kernel Sizes:"
            print "-------------------------------"
            print ""
            print "Support of w-kernel: ", o.Ngw, " pixels"
            print "Support of combined GCF: ", o.Ncvff, " sub-pixels"
            print ""
            print "------------------------------"
            print ""
            if on_the_fly:
                print "WARNING! On the fly kernels in use. Experimental!:  (Set on_the_fly = False to disable)"
                print "On the fly kernels is a new option forcing convolution kernels to be recalculated"
                print "for each and every viibility point, but only at the actual size needed  - i.e. not"
                print "oversampled by a factor of Qgcf (8)."

        # ===============================================================================================
        # PDR05 Sec 12.8
        # ===============================================================================================

        ncomb = o.Na * (o.Na - 1) / 2.0
        # Eq. 31 Visibility rate for backward step, allow coalescing in time and freq prior to gridding
        o.Nvis_backward = o.binfrac * ncomb * o.Nf_vis_backward / o.Tdump_backward
        # Eq. 31 Visibility rate for predict step
        o.Nvis_predict  = o.binfrac * ncomb * o.Nf_vis_predict  / o.Tdump_predict

        # Eq. 30 : R_flop = 2 * N_maj * N_pp * N_beam * ( R_grid + R_fft + R_rp + R_ccf)
        # no factor 2 in the line below, because forward & backward steps are both in Rflop numbers
        Rflop_common_factor = o.Nmajor * o.Npp * o.Nbeam

        # Gridding:
        # --------
        o.Rgrid_backward = 8. * o.Nvis_backward * Nkernel2 * o.Nmm  # Eq 32; FLOPS
        o.Rgrid_predict  = 8. * o.Nvis_predict  * Nkernel2 * o.Nmm  # Eq 32; FLOPS, per half cycle, per polarisation, per beam, per facet
        o.Rgrid = o.Rgrid_backward + o.Rgrid_predict
        o.Rflop_grid = Rflop_common_factor * o.Rgrid *o.Nfacet**2 #TODO: Multiply by Nfacet**2. Pretty major bug - how did we manage that / double check.

        # FFT:
        # ---
        if imaging_mode in (ImagingModes.Continuum, ImagingModes.FastImg):
            # make only enough FFT grids to extract necessary spectral info and retain distributability.
            o.Nf_FFT_backward = o.minimum_channels
        elif imaging_mode == ImagingModes.Spectral:
            o.Nf_out = o.Nf_max
            o.Nf_FFT_backward = o.Nf_max
        else:
            raise Exception("Unknown Imaging Mode defined : %s" % imaging_mode)

        o.Nf_FFT_predict = o.Nf_vis_predict
        Nfacet_x_Npix = o.Nfacet * o.Npix_linear #This is mathematically correct below but potentially misleading (lines 201,203) as the Nln(N,2) is familiar to many users.

        # Eq. 33, per output grid (i.e. frequency)
        # TODO: please check correctness of 2 eqns below.
        # TODO: Note the Nf_out factor is only in the backward step of the final cycle.
        # note: o.binfrac serves to handle the fact that the FFT step is done once for all baselines and not on a baseline-bin basis.
        o.Rfft_backward = o.binfrac * 5. * Nfacet_x_Npix ** 2 * log(o.Npix_linear, 2) / o.Tsnap
        # Eq. 33 per predicted grid (i.e. frequency)
        o.Rfft_predict  = o.binfrac * 5. * Nfacet_x_Npix ** 2 * log(Nfacet_x_Npix, 2) / o.Tsnap #Predict step is at full FoV (NfacetXNpix)
        o.Rfft_intermediate_cycles = (o.Nf_FFT_backward * o.Rfft_backward) + (o.Nf_FFT_predict * o.Rfft_predict)
        # final major cycle, create final data products (at Nf_out channels)
        o.Rfft_final_cycle = (o.Nf_out * o.Rfft_backward) + (o.Nf_FFT_predict * o.Rfft_predict)

        # do Nmajor-1 cycles before doing the final major cycle.
        o.Rflop_fft = o.Npp * o.Nbeam* ((o.Nmajor - 1) * o.Rfft_intermediate_cycles + o.Rfft_final_cycle)

        # Re-Projection:
        # -------------
        if imaging_mode in (ImagingModes.Continuum, ImagingModes.Spectral):
            o.Rrp = 50. * Nfacet_x_Npix ** 2 / o.Tsnap  # Eq. 34
        elif imaging_mode == ImagingModes.FastImg:
            o.Rrp = 0  # (Consistent with PDR05 280115)
        else:
            raise Exception("Unknown Imaging Mode : %s" % imaging_mode)

        # Reproj intermediate major cycle FFTs (Nmaj -1) times,
        # then do the final ones for the last cycle at the full output spectral resolution.
        o.Rflop_proj = o.Rrp * (o.Nbeam * o.Npp) * ((o.Nmajor - 1) * o.Nf_FFT_backward + o.Nf_out)

        # Convolution kernels:
        # --------------------
        o.grid_cell_error = o.epsilon_f_approx
        o.dfonF = o.grid_cell_error * o.Qgcf / (o.Qkernel * o.Ncvff)

        # allow uv positional errors up to grid_cell_error * 1/Qkernel of a cell from frequency smearing.
        o.Nf_gcf_backward_nosmear = log(o.wl_max / o.wl_min) / log(o.dfonF + 1.)
        o.Nf_gcf_predict_nosmear  = o.Nf_gcf_backward_nosmear #TODO: Placeholder since we don't know how many frequecy kernels we need for predict step.

        if on_the_fly:
            o.Nf_gcf_backward = o.Nf_vis_backward
            o.Nf_gcf_predict  = o.Nf_vis_predict
            o.Tkernel_backward = o.Tdump_backward
            o.Tkernel_predict  = o.Tdump_predict
        else:
            # For both of the following, maintain distributability; need at least minimum_channels (500) kernels.
            o.Nf_gcf_backward = Max(o.Nf_gcf_backward_nosmear, o.minimum_channels)
            o.Nf_gcf_predict  = Max(o.Nf_gcf_predict_nosmear,  o.minimum_channels)
            o.Tkernel_backward = o.Tion
            o.Tkernel_predict  = o.Tion

        if verbose:
            print "Number of kernels to cover freq axis is Nf_FFT_backward: ", o.Nf_gcf_backward
            print "Number of kernels to cover freq axis is Nf_FFT_predict: ", o.Nf_gcf_predict

        # The following two equations correspond to Eq. 35
        o.Rccf_backward = o.binfrac * 5. * o.Nf_gcf_backward * ncomb * o.Nfacet**2 * o.Ncvff**2 * log(o.Ncvff, 2) * o.Nmm / o.Tkernel_backward
        o.Rccf_predict  = o.binfrac * 5. * o.Nf_gcf_predict  * ncomb * o.Nfacet**2 * o.Ncvff**2 * log(o.Ncvff, 2) * o.Nmm / o.Tkernel_predict
        o.Rccf = o.Rccf_backward + o.Rccf_predict
        o.Rflop_conv = Rflop_common_factor * o.Rccf

        # Phase rotation (for the facetting):
        # --------------
        # Eq. 29. The sign() statement below serves as an "if > 1" statement for this symbolic equation.
        # 25 FLOPS per visiblity. Only do it if we need to facet.
        o.Rflop_phrot = sign(o.Nfacet - 1) * 25 * Rflop_common_factor * (o.Nvis_predict + o.Nvis_backward) * o.Nfacet ** 2

        # Calculate overall flop rate : revised Eq. 30
        # ================================================================================
        o.Rflop = o.Rflop_grid + o.Rflop_fft + o.Rflop_proj + o.Rflop_conv + o.Rflop_phrot

        # ===============================================================================================
        # Compute the Buffer sizes - section 12.15 in PDR05
        # ===============================================================================================

        o.Mw_cache = (o.Ngw ** 3) * (o.Qgcf ** 3) * o.Nbeam * o.Nf_vis_predict * 8.0  # Eq 48.

        # Note the factor 2 in the line below -- we have a double buffer
        # (allowing storage of a full observation while simultaneously capturing the next)
        # TODO: The o.Nbeam factor in eqn below is not mentioned in PDR05 eq 49. Why?
        o.Mbuf_vis = 2 * o.Npp * o.Nvis_predict * o.Nbeam * o.Mvis * o.Tobs  # Eq 49

        # added o.Nfacet dependence; changed Nmajor factor to Nmajor+1 as part of post PDR fixes.
        # TODO: Differs quite substantially from Eq 50, by merit of the Nbeam and Npp, as well as Nfacet ** 2 factors.
        # TODO: PDR05 lacking in this regard and must be updated.
        o.Rio = o.Nbeam * o.Npp * (1 + o.Nmajor) * o.Nvis_predict * o.Mvis * o.Nfacet ** 2  # Eq 50

        # TODO : In this block of code, the last action is a modification of o.Npix_linear that looks suspect to me.
        # TODO : Why is the value overwritten? Where is it used in its new form? Only in api_ipython it seems. Is this correct / intended?
        # TODO : Agreed - this is an ugly workaround because Npix_linear is the same in all baseline bins and we just want the one value of it fed back in the results, just do this at the very end so it doesn't screw up other calculations.
        # TODO : since binfrac is normalised to unity this "works" but it's ugly. Previously, we were getting numbers N_bins x too big.
        o.Npix_linear = o.Npix_linear * o.binfrac

        return o