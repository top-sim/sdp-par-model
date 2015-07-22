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
        
        #TODO: make this section (below0 work so that if Nfacet=1 we do not make the maps larger than they need to be
        #i.e. only
        #if o.Nfacet ==1:
        #    o.facet_overlap_frac=0.0 #If we are not using facets, we mustn't unneccesarily increase the FoV!
        #print "using Nfacet, facet_overlap:", o.Nfacet, o.facet_overlap_frac
        using_facet_overlap_frac=sign(o.Nfacet - 1)*o.facet_overlap_frac
        
        

        o.wl_max = o.c / o.freq_min  # Maximum Wavelength
        o.wl_min = o.c / o.freq_max  # Minimum Wavelength
        o.wl = 0.5 * (o.wl_max + o.wl_min)  # Representative Wavelength

        # ===============================================================================================
        # PDR05 (version 1.85) Sec 9.2
        # ===============================================================================================

        # TODO: In line below: PDR05 uses *wl_max* instead of wl. Also uses 7.6 instead of 7.66. Is this correct?
        o.Theta_fov = 7.66 * o.wl * o.Qfov *  (1+using_facet_overlap_frac) / (pi * o.Ds * o.Nfacet)  # Eq 6 - Facet Field-of-view (linear)
        o.Total_fov = 7.66 * o.wl * o.Qfov / (pi * o.Ds) # Total linear field of view of map
        # TODO: In the two lines below, PDR05 uses *wl_min* instead of wl
        o.Theta_beam = 3 * o.wl / (2. * o.Bmax)     # Synthesized beam. Called Theta_PSF in PDR05.
        o.Theta_pix = o.Theta_beam / (2. * o.Qpix)  # Eq 7 - Pixel size
        o.Npix_linear = o.Theta_fov / o.Theta_pix   # Eq 8 - Number of pixels on side of facet
        o.epsilon_f_approx = sqrt(6 * (1 - (1. / o.amp_f_max)))  # expansion of sine solves eps = arcsinc(1/amp_f_max).
        o.Qbw = 1.47 / o.epsilon_f_approx  # See notes on https://confluence.ska-sdp.org/display/PIP/Frequency+resolution+and+smearing+effects+in+the+iPython+SDP+Parametric+model

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
        o.Nf_no_smear_predict =  log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax * o.Total_fov * o.Qbw)))
        o.Nf_no_smear_backward = log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax_bin * o.Theta_fov * o.Qbw)))

        # correlator output averaging time scaled for max baseline.
        o.Tdump_scaled = o.Tdump_ref * o.B_dump_ref / o.Bmax
        o.combine_time_samples = Max(
            floor(o.epsilon_f_approx * o.wl / (o.Total_fov * o.Omega_E * o.Bmax_bin * o.Tdump_scaled)), 1.)
        o.Tcoal_skipper = o.Tdump_scaled * o.combine_time_samples #coalesce visibilities in time.

        if bl_dep_time_av:
            # Don't let any bl-dependent time averaging be for longer than either 1.2s or Tion. ?Why 1.2s?
            o.Tcoal_predict = Min(o.Tcoal_skipper, 1.2, o.Tion)
            # For backward step at gridding only, allow coalescance of visibility points at Facet FoV
            # smearing limit only for BLDep averaging case.
            o.Tcoal_backward = Min(o.Tcoal_skipper * o.Nfacet/(1+using_facet_overlap_frac), o.Tion) #scale Skipper time to smaller field of view of facet, rather than full FoV.
        else:
            o.Tcoal_predict = o.Tdump_scaled
            o.Tcoal_backward = o.Tdump_scaled

        if verbose:
            print "Channelization Characteristics:"
            print "-------------------------------\n"
            print "Ionospheric timescale (for updating kernels and limiting any time averaging): ", o.Tion, " sec"
            print "Coalesce Time predict: ", o.Tcoal_predict, " sec"
            print "Coalesce Time backward: ", o.Tcoal_backward, " sec"
            print ""
            print "No. freq channels for predict: ", o.Nf_no_smear_predict
            print "No. freq channels for backward step: ", o.Nf_no_smear_backward
            print ""
            if bl_dep_time_av:
                print "USING BASELINE DEPENDENT TIME AVERAGING"
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
        o.Ngw_backward = 2 * o.Theta_fov * sqrt((o.DeltaW_max * o.Theta_fov / 2.) ** 2 +
                                       (o.DeltaW_max ** 1.5 * o.Theta_fov / (2 * pi * o.epsilon_w)))  # Eq. 25, for Facet FoV
                                       
        o.Ngw_predict = 2 * o.Total_fov * sqrt((o.DeltaW_max * o.Total_fov / 2.) ** 2 +
                                        (o.DeltaW_max ** 1.5 * o.Total_fov / (2 * pi * o.epsilon_w)))  # Eq. 25, for full FoV


        # TODO: Check split of kernel size for backward and predict steps.
        Nkernel2_backward = o.Ngw_backward ** 2 + o.Naa ** 2  # squared linear size of combined W and A kernels; used in eqs 23 and 32
        Nkernel2_predict = o.Ngw_predict ** 2 + o.Naa ** 2  # squared linear size of combined W and A kernels; used in eqs 23 and 32
        if on_the_fly:
            o.Qgcf = 1.0
        
        o.Ncvff_backward = sqrt(Nkernel2_backward)*o.Qgcf  #  Eq. 23 : combined kernel support size and oversampling
        o.Ncvff_predict = sqrt(Nkernel2_predict)*o.Qgcf  #  Eq. 23 : combined kernel support size and oversampling

        o.Nf_vis_backward = Min(Max(o.Nf_out, o.Nf_no_smear_backward),o.Nf_max) #TODO:can't be bigger than the channel count from the correlator
        o.Nf_vis_predict = Min(Max(o.Nf_out, o.Nf_no_smear_predict),o.Nf_max) #TODO:can't be bigger than the channel count from the correlator
        
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
            print "Support of w-kernel: ", o.Ngw_predict, " pixels"
            print "Support of combined, oversampled GCF at far field: ", o.Ncvff_predict, " sub-pixels"
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

        nbaselines = o.Na * (o.Na - 1) / 2.0
        # Eq. 31 Visibility rate for backward step, allow coalescing in time and freq prior to gridding
        o.Nvis_backward = o.binfrac * nbaselines * o.Nf_vis_backward / o.Tcoal_backward
        # Eq. 31 Visibility rate for predict step
        o.Nvis_predict  = o.binfrac * nbaselines * o.Nf_vis_predict  / o.Tcoal_predict

        # Eq. 30 : R_flop = 2 * N_maj * N_pp * N_beam * ( R_grid + R_fft + R_rp + R_ccf)
        # no factor 2 in the line below, because forward & backward steps are both in Rflop numbers
        Rflop_common_factor = o.Nmajor * o.Npp * o.Nbeam

        # Gridding:
        # --------
        o.Rgrid_backward = 8. * o.Nvis_backward * Nkernel2_backward * o.Nmm * Rflop_common_factor *o.Nfacet**2# Eq 32; FLOPS
        o.Rgrid_predict  = 8. * o.Nvis_predict  * Nkernel2_predict * o.Nmm * Rflop_common_factor # Eq 32; FLOPS, per half cycle, per polarisation, per beam, per facet - only one facet for predict
        o.Rflop_grid = o.Rgrid_backward + o.Rgrid_predict

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
        o.Rfft_predict  = o.binfrac * 5. * Nfacet_x_Npix ** 2 * log(Nfacet_x_Npix, 2) / o.Tsnap #Predict step is at full FoV (NfacetXNpix) TODO: PIP.IMG check this
        o.Rfft_intermediate_cycles = (o.Nf_FFT_backward * o.Rfft_backward) + (o.Nf_FFT_predict * o.Rfft_predict)
        # final major cycle, create final data products (at Nf_out channels)
        o.Rfft_final_cycle = (o.Nf_out * o.Rfft_backward) + (o.Nf_FFT_predict * o.Rfft_predict)

        # do Nmajor-1 cycles before doing the final major cycle.
        o.Rflop_fft = o.Npp * o.Nbeam* (((o.Nmajor - 1) * o.Rfft_intermediate_cycles) + o.Rfft_final_cycle)

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
        o.Rflop_proj = o.Rrp * (o.Nbeam * o.Npp) * ((o.Nmajor - 1) * o.Nf_FFT_backward + o.Nf_out) #TODO: does this account for backward and predict steps correctly? PIP.IMG check please.

        # Generate Convolution kernels:
        # --------------------

        o.dfonF_backward = o.epsilon_f_approx / (o.Qkernel * sqrt(Nkernel2_backward))
        o.dfonF_predict = o.epsilon_f_approx / (o.Qkernel * sqrt(Nkernel2_predict))

        # allow uv positional errors up to o.epsilon_f_approx * 1/Qkernel of a cell from frequency smearing.(But not more than Nf_max channels...)
        o.Nf_gcf_backward_nosmear = Min(log(o.wl_max / o.wl_min) / log(o.dfonF_backward + 1.), o.Nf_max) #TODO: PIP.IMG check please
        o.Nf_gcf_predict_nosmear  = Min(log(o.wl_max / o.wl_min) / log(o.dfonF_predict + 1.), o.Nf_max) #TODO: PIP.IMG check please

        if on_the_fly:
            o.Nf_gcf_backward = o.Nf_vis_backward
            o.Nf_gcf_predict  = o.Nf_vis_predict
            o.Tkernel_backward = o.Tcoal_backward
            o.Tkernel_predict  = o.Tcoal_predict
        else:
            # For both of the following, maintain distributability; need at least minimum_channels (500) kernels.
            o.Nf_gcf_backward = Max(o.Nf_gcf_backward_nosmear, o.minimum_channels)
            o.Nf_gcf_predict  = Max(o.Nf_gcf_predict_nosmear,  o.minimum_channels)
            o.Tkernel_backward = o.Tion #TODO: some baseline dependent re-use limits along the same lines as the frequency re-use? PIP.IMG check please.
            o.Tkernel_predict  = o.Tion

        if verbose:
            print "Number of kernels to cover freq axis is Nf_gcf_backward: ", o.Nf_gcf_backward
            print "Number of kernels to cover freq axis is Nf_gcf_predict: ", o.Nf_gcf_predict

        # The following two equations correspond to Eq. 35
        o.Rccf_backward = o.binfrac * 5. * o.Nf_gcf_backward * nbaselines * o.Nfacet**2 * o.Ncvff_backward**2 * log(o.Ncvff_backward, 2) * o.Nmm / o.Tkernel_backward
        o.Rccf_predict  = o.binfrac * 5. * o.Nf_gcf_predict  * nbaselines * o.Ncvff_predict**2 * log(o.Ncvff_predict, 2) * o.Nmm / o.Tkernel_predict #TODO we assume Nfacet=1 for predict step, so do we need Nfacet^2 multiplier in here? PIP.IMG check please
        o.Rccf = o.Rccf_backward + o.Rccf_predict
        o.Rflop_conv = Rflop_common_factor * o.Rccf

        # Phase rotation (for the facetting):
        # --------------
        # Eq. 29. The sign() statement below serves as an "if > 1" statement for this symbolic equation.
        # 25 FLOPS per visiblity. Only do it if we need to facet.
        # TODO: check line below - is it correct if we don't facet in the predict step? PIP.IMG check please
        o.Rflop_phrot = sign(o.Nfacet - 1) * 25 * Rflop_common_factor * (o.Nvis_predict + o.Nvis_backward) * o.Nfacet ** 2

        # Calculate overall flop rate : revised Eq. 30
        # ================================================================================
        o.Rflop = o.Rflop_grid + o.Rflop_fft + o.Rflop_proj + o.Rflop_conv + o.Rflop_phrot

        # ===============================================================================================
        # Compute the Buffer sizes - section 12.15 in PDR05
        # ===============================================================================================

        o.Mw_cache = (o.Ngw_predict ** 3) * (o.Qgcf ** 3) * o.Nbeam * o.Nf_vis_predict * 8.0  # Eq 48. TODO: re-implement this equation within a better description of where kernels will be stored etc.
        # Note the factor 2 in the line below -- we have a double buffer
        # (allowing storage of a full observation while simultaneously capturing the next)
        # TODO: The o.Nbeam factor in eqn below is not mentioned in PDR05 eq 49. Why?
        o.Mbuf_vis = 2 * o.Npp * o.Nvis_predict * o.Nbeam * o.Mvis * o.Tobs  # Eq 49

        # added o.Nfacet dependence; changed Nmajor factor to Nmajor+1 as part of post PDR fixes.
        # TODO: Differs quite substantially from Eq 50, by merit of the Nbeam and Npp, as well as Nfacet ** 2 factors.
        # TODO: PDR05 lacking in this regard and must be updated.
        o.Rio = o.Nbeam * o.Npp * (1 + o.Nmajor) * o.Nvis_predict * o.Mvis * o.Nfacet ** 2  # Eq 50 TODO: is this correct if we have only got facets for the backward step and use Nfacet=1 for predict step?

        return o