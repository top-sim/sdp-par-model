"""
This class contains the actual equations that are used to compute the telescopes' performance values and computional
requirements from the supplied basic parameters defined in ParameterDefinitions.

This class contains a method for (symbolically) computing derived parameters using imaging equations described
in PDR05 (version 1.85).
"""

from sympy import log, Min, Max, sqrt, floor, sign, Symbol, Lambda, Add
from numpy import pi, round
import math
from parameter_definitions import ImagingModes
from parameter_definitions import ParameterContainer

class Equations:
    def __init__(self):
        pass

    @staticmethod
    def apply_imaging_equations(telescope_parameters, imaging_mode,
                                bl_dep_time_av, bins, binfracs,
                                on_the_fly=False, verbose=False):
        """
        (Symbolically) computes a set of derived parameters using imaging
        equations described in PDR05 (version 1.85).

        The derived parameters are added to the supplied
        telescope_parameter object (locally referred to as o).  Where
        parameters are only described symbolically (using sympy) they
        can be numerically evaluated at a later stage, when unknown
        symbolic variables are suitably substituted. This is typically
        done in external code contained, e.g. in implementation.py.

        @param telescope_parameters: ParameterContainer object
            containing the telescope parameters.  This
            ParameterContainer object is modified in-place by
            appending / overwriting the relevant fields
        @param imaging_mode: The telecope's imaging mode
        @param bl_dep_time_av: True iff baseline dependent time
            averaging should be used.
        @param on_the_fly: True iff using on-the-fly kernels
        @param verbose: displays verbose command-line output
        """
        o = telescope_parameters  # Used for shorthand in the equations below
        assert isinstance(o, ParameterContainer)
        assert hasattr(o, "c")  # Checks initialization by proxy of whether the speed of light is defined

        # Store parameters
        o.imaging_mode = imaging_mode
        o.bl_dep_time_av = bl_dep_time_av
        o.Bmax_bins = list(bins)
        o.frac_bins = list(binfracs)
        o.on_the_fly = on_the_fly

        # Check parameters
        if o.Tobs < 10.0:
            o.Tsnap_min = o.Tobs
            if verbose:
                print 'Warning: Tsnap_min overwritten in equations.py file because observation was shorter than 10s'

        # Derive simple parameters
        #o.Na = 512 * (35.0/o.Ds)**2 #Hack to make station diameter and station number inter-related...comment it out after use
        o.nbaselines = o.Na * (o.Na - 1) / 2.0

        o.wl_max = o.c / o.freq_min  # Maximum Wavelength
        o.wl_min = o.c / o.freq_max  # Minimum Wavelength
        o.wl = 0.5 * (o.wl_max + o.wl_min)  # Representative Wavelength

        # Apply general imaging equations
        Equations._apply_image_equations(o)
        Equations._apply_channel_equations(o)
        Equations._apply_coalesce_equations(o)
        Equations._apply_geometry_equations(o)

        # Apply function equations
        Equations._apply_grid_fft_equations(o)
        Equations._apply_reprojection_equations(o)
        Equations._apply_kernel_equations(o)
        Equations._apply_phrot_equations(o)

        # Apply summary equations
        Equations._apply_flop_equations(o)
        Equations._apply_io_equations(o)

        return o

    @staticmethod
    def _sum_bl_bins(o, bcount, bmax, expr):
        """Helper for dealing with baseline dependence. For a term
        depending on the given symbols, "sum_bl_bins" will build a
        sum term over all baseline bins."""

        # Replace in concrete values for baseline fractions and
        # length. Using Lambda here is a solid 25% faster than
        # subs(). Unfortunately very slow nonetheless...
        results = []
        lam = Lambda((bcount,bmax), expr)
        for (frac_val, bmax_val) in zip(o.frac_bins, o.Bmax_bins):
            results.append(lam(frac_val*o.nbaselines, bmax_val))
        return Add(*results, evaluate=False)

    @staticmethod
    def _apply_image_equations(o):
        """
        Calculate image parameters, such as resolution and size

        References: PDR05 (version 1.85) Sec 9.2
        """

        # Facet overlap is only needed if Nfacet > 1
        o.using_facet_overlap_frac=sign(o.Nfacet - 1)*o.facet_overlap_frac

        # TODO: In line below: PDR05 uses *wl_max* instead of wl. Also
        # uses 7.6 instead of 7.66. Is this correct?
        o.Number_imaging_subbands = math.ceil(log(o.wl_max/o.wl_min)/log(o.max_subband_freq_ratio))
        subband_frequency_ratio = (o.wl_max/o.wl_min)**(1./o.Number_imaging_subbands)

        # max subband wavelength to set image FoV
        o.wl_sb_max = o.wl *sqrt(subband_frequency_ratio)
        # min subband wavelength to set pixel size
        o.wl_sb_min = o.wl_sb_max / subband_frequency_ratio

        # Eq 6 - Facet Field-of-view (linear) at max sub-band wavelength
        o.Theta_fov = 7.66 * o.wl_sb_max * o.Qfov * (1+o.using_facet_overlap_frac) \
                      / (pi * o.Ds * o.Nfacet)
        # Total linear field of view of map (all facets)
        o.Total_fov = 7.66 * o.wl_sb_max * o.Qfov / (pi * o.Ds)
        # TODO: In the two lines below, PDR05 uses *wl_min* instead of wl
        # Synthesized beam at fiducial wavelength. Called Theta_PSF in PDR05.
        o.Theta_beam = 3 * o.wl_sb_min / (2. * o.Bmax)
        # Eq 7 - Pixel size at fiducial wavelength.
        o.Theta_pix = o.Theta_beam / (2. * o.Qpix)

        # Eq 8 - Number of pixels on side of facet in subband.
        o.Npix_linear = (o.Theta_fov / o.Theta_pix)
        o.Npix_linear_total_fov = (o.Total_fov / o.Theta_pix)
        # expansion of sine solves eps = arcsinc(1/amp_f_max).
        o.epsilon_f_approx = sqrt(6 * (1 - (1. / o.amp_f_max)))
        # See notes on https://confluence.ska-sdp.org/display/PIP/Frequency+resolution+and+smearing+effects+in+the+iPython+SDP+Parametric+model
        o.Qbw = 1.47 / o.epsilon_f_approx

    @staticmethod
    def _apply_channel_equations(o):
        """
        Determines the number of frequency channels to use in backward &
        predict steps.

        References: PDR05 Sec 9.1
        """

        bmax = Symbol('bmax')
        log_wl_ratio = log(o.wl_max / o.wl_min)

        # The two equations below => combination of Eq 4 and Eq 5 for
        # full and facet FOV at max baseline respectively.  These
        # limit bandwidth smearing to within a fraction
        # (epsilon_f_approx) of a uv cell.
        # Done: PDR05 Eq 5 says o.Nf = log_wl_ratio / (1 + 0.6 * o.Ds / (o.Bmax * o.Q_fov * o.Qbw)).
        # This is fine - substituting in the equation for theta_fov shows it is indeed correct.
        o.Nf_no_smear_predict = \
            log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax * o.Total_fov * o.Qbw)))
        #Use full FoV for de-grid (predict) for high accuracy
        
        o.Nf_no_smear_backward = Lambda(bmax,
            log_wl_ratio / log(1 + (3 * o.wl / (2. * bmax * o.Theta_fov * o.Qbw))))
        

        # Actual frequency channels for backward & predict
        # steps. Bound by minimum parallism and input channel count.
        o.Nf_vis_backward = Lambda(bmax,
            Min(Max(o.Nf_out, o.Nf_no_smear_backward(bmax)),o.Nf_max))
        o.Nf_vis_predict = \
            Min(Max(o.Nf_out, o.Nf_no_smear_predict),o.Nf_max)

        # Number of frequency channels depends on imaging mode
        if o.imaging_mode in (ImagingModes.Continuum, ImagingModes.FastImg):
            # make only enough FFT grids to extract necessary spectral
            # info and retain distributability.
            o.Nf_FFT_backward = o.minimum_channels
        elif o.imaging_mode == ImagingModes.Spectral:
            o.Nf_out = o.Nf_max
            o.Nf_FFT_backward = o.Nf_max
        else:
            raise Exception("Unknown Imaging Mode defined : %s" % imaging_mode)
        o.Nf_FFT_predict = o.N_taylor_terms * o.Number_imaging_subbands #This is an important and substantial change - we made FFT grids corresponding to the sky model and from these then interpolate and de-grid. We do not make an FFT sky at each predict frequency.
            #o.Nf_FFT_predict = o.Nf_vis_predict

    @staticmethod
    def _apply_coalesce_equations(o):

        # correlator output averaging time scaled for max baseline.
        o.Tdump_scaled = o.Tdump_ref * o.B_dump_ref / o.Bmax
        bmax = Symbol('bmax')
        o.combine_time_samples = Lambda(bmax,
            Max(floor(o.epsilon_f_approx * o.wl /
                      (o.Total_fov * o.Omega_E * bmax * o.Tdump_scaled)), 1.))
        # coalesce visibilities in time.
        o.Tcoal_skipper = Lambda(bmax,
            o.Tdump_scaled * o.combine_time_samples(bmax))

        if o.bl_dep_time_av:
            # Don't let any bl-dependent time averaging be for longer
            # than either 1.2s or Tion. ?Why 1.2s?
            o.Tcoal_predict = Lambda(bmax,
                Min(o.Tcoal_skipper(bmax), 1.2, o.Tion))
            # For backward step at gridding only, allow coalescance of
            # visibility points at Facet FoV smearing limit only for
            # BLDep averaging case.
            o.Tcoal_backward = Lambda(bmax,
                Min(o.Tcoal_skipper(bmax) * o.Nfacet/(1+o.using_facet_overlap_frac), o.Tion))
            # scale Skipper time to smaller field of view of facet,
            # rather than full FoV.
        else:
            o.Tcoal_predict = Lambda(bmax, o.Tdump_scaled)
            o.Tcoal_backward = Lambda(bmax, o.Tdump_scaled)

    @staticmethod
    def _apply_geometry_equations(o):

        # ===============================================================================================
        # PDR05 Sec 12.2 - 12.5
        # ===============================================================================================

        bmax = Symbol('bmax')
        o.DeltaW_Earth = Lambda(bmax, bmax ** 2 / (8. * o.R_Earth * o.wl))  # Eq. 19
        # TODO: in the two lines below, PDR05 uses lambda_min, not mean.
        # Eq. 26 : W-deviation for snapshot.
        o.DeltaW_SShot = Lambda(bmax, bmax * o.Omega_E * o.Tsnap / (2. * o.wl))
        o.DeltaW_max = Lambda(bmax, o.Qw * Max(o.DeltaW_SShot(bmax), o.DeltaW_Earth(bmax)))

        # Eq. 25, w-kernel support size **Note difference in cellsize assumption**
        def Ngw(deltaw, fov):
            return 2 * fov * sqrt((deltaw * fov / 2.) ** 2 +
                                  (deltaw**1.5 * fov / (2 * pi * o.epsilon_w)))
        o.Ngw_backward = Lambda(bmax, Ngw(o.DeltaW_max(bmax), o.Theta_fov))
        o.Ngw_predict = Lambda(bmax, Ngw(o.DeltaW_max(bmax), o.Total_fov))

        # TODO: Check split of kernel size for backward and predict steps.
        # squared linear size of combined W and A kernels; used in eqs 23 and 32
        o.Nkernel2_backward = Lambda(bmax, o.Ngw_backward(bmax) ** 2 + o.Naa ** 2)
        o.Nkernel_AW_backward = Lambda(bmax, (o.Ngw_backward(bmax) ** 2 + o.Naa ** 2)**0.5)
        # squared linear size of combined W and A kernels; used in eqs 23 and 32
        o.Nkernel2_predict = Lambda(bmax, o.Ngw_predict(bmax) ** 2 + o.Naa ** 2)
        o.Nkernel_AW_predict = Lambda(bmax, (o.Ngw_predict(bmax) ** 2 + o.Naa ** 2)**0.5)
        if o.on_the_fly:
            o.Qgcf = 1.0

        # Eq. 23 : combined kernel support size and oversampling
        o.Ncvff_backward = Lambda(bmax,
            sqrt(o.Nkernel2_backward(bmax))*o.Qgcf)
        # Eq. 23 : combined kernel support size and oversampling
        o.Ncvff_predict = Lambda(bmax,
            sqrt(o.Nkernel2_predict(bmax))*o.Qgcf)

    @staticmethod
    def _apply_grid_fft_equations(o):

        # ===============================================================================================
        # PDR05 Sec 12.8
        # ===============================================================================================

        # Eq. 31 Visibility rate for backward step, allow coalescing
        # in time and freq prior to gridding
        bmax = Symbol('bmax')
        bcount = Symbol('bcount')
        o.Nvis_backward = Lambda((bcount, bmax),
            bcount * o.Nf_vis_backward(bmax) / o.Tcoal_backward(bmax))
        # Eq. 31 Visibility rate for predict step
        o.Nvis_predict = Lambda((bcount, bmax),
            bcount * o.Nf_vis_predict / o.Tcoal_predict(bmax))
        o.Nvis_predict_no_averaging = o.nbaselines * sum(o.frac_bins) * o.Nf_vis_predict / o.Tdump_scaled
        # The line above uses Tdump_scaled independent of whether
        # BLDTA is used.  This is because BLDTA is only done for
        # gridding, and doesn't affect the amount of data to be
        # buffered

        # Gridding:
        # --------
        o.Rgrid_backward = \
            8. * o.Nmm * o.Nmajor * o.Npp * o.Nbeam * o.Nfacet**2 * \
            Equations._sum_bl_bins(o, bcount, bmax,
                o.Nvis_backward(bcount, bmax) * o.Nkernel2_backward(bmax))
            # Eq 32; FLOPS
            
        # De-gridding in Predict step
        o.Rgrid_predict = \
            8. * o.Nmm * o.Nmajor * o.Npp * o.Nbeam * \
            Equations._sum_bl_bins(o, bcount, bmax,
                o.Nvis_predict(bcount, bmax) * o.Nkernel2_predict(bmax))
            # Eq 32; FLOPS, per half cycle, per polarisation, per beam, per facet - only one facet for predict

        o.Rflop_grid = o.Rgrid_backward + o.Rgrid_predict

        # FFT:
        # ---
        if o.imaging_mode in (ImagingModes.Continuum, ImagingModes.FastImg):
            # make only enough FFT grids to extract necessary spectral info and retain distributability.
            o.Nf_FFT_backward = o.minimum_channels
        elif o.imaging_mode == ImagingModes.Spectral:
            o.Nf_out = o.Nf_max
            o.Nf_FFT_backward = o.Nf_max
        else:
            raise Exception("Unknown Imaging Mode defined : %s" % o.imaging_mode)

        o.Nfacet_x_Npix = o.Nfacet * o.Npix_linear #This is mathematically correct below but potentially misleading (lines 201,203) as the Nln(N,2) is familiar to many users.

        # Eq. 33, per output grid (i.e. frequency)
        # TODO: please check correctness of 2 eqns below.
        # TODO: Note the Nf_out factor is only in the backward step of the final cycle.
        o.Rfft_backward = 5. * o.Nfacet**2 * o.Npix_linear ** 2 * log(o.Npix_linear, 2) / o.Tsnap
        # Eq. 33 per predicted grid (i.e. frequency)
        o.Rfft_predict  = 5. * o.Npix_linear_total_fov** 2 * log(o.Npix_linear_total_fov, 2) / o.Tsnap #Predict step is at full FoV (NfacetXNpix) TODO: PIP.IMG check this
        o.Rfft_intermediate_cycles = (o.Nf_FFT_backward * o.Rfft_backward) + (o.Nf_FFT_predict * o.Rfft_predict)
        # final major cycle, create final data products (at Nf_out channels)
        o.Rfft_final_cycle = (o.Nf_out * o.Rfft_backward) + (o.Nf_FFT_predict * o.Rfft_predict)

        # do Nmajor-1 cycles before doing the final major cycle.
        o.Rflop_fft = o.Npp * o.Nbeam* (((o.Nmajor - 1) * o.Rfft_intermediate_cycles) + o.Rfft_final_cycle)

    @staticmethod
    def _apply_reprojection_equations(o):

        # Re-Projection:
        # -------------
        if o.imaging_mode in (ImagingModes.Continuum, ImagingModes.Spectral):
            o.Rrp = 50. * o.Nfacet**2 * o.Npix_linear ** 2 / o.Tsnap  # Eq. 34
        elif o.imaging_mode == ImagingModes.FastImg:
            o.Rrp = 0  # (Consistent with PDR05 280115)
        else:
            raise Exception("Unknown Imaging Mode : %s" % o.imaging_mode)

        # Reproj intermediate major cycle FFTs (Nmaj -1) times,
        # then do the final ones for the last cycle at the full output spectral resolution.
        o.Rflop_proj = o.Rrp * (o.Nbeam * o.Npp) * ((o.Nmajor - 1) * o.Nf_FFT_backward + o.Nf_out) #TODO: no reporjection for Predict step, only on backward.

    @staticmethod
    def _apply_kernel_equations(o):
        """
        Generate Convolution kernels
        """

        bmax = Symbol('bmax')
        o.dfonF_backward = Lambda(bmax, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_backward(bmax))))
        o.dfonF_predict = Lambda(bmax, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_predict(bmax))))

        # allow uv positional errors up to o.epsilon_f_approx *
        # 1/Qkernel of a cell from frequency smearing.(But not more
        # than Nf_max channels...)
        o.Nf_gcf_backward_nosmear = Lambda(bmax,
            Min(log(o.wl_max / o.wl_min) /
                log(o.dfonF_backward(bmax) + 1.), o.Nf_max)) #TODO: PIP.IMG check please
        o.Nf_gcf_predict_nosmear  = Lambda(bmax,
            Min(log(o.wl_max / o.wl_min) /
                log(o.dfonF_predict(bmax) + 1.), o.Nf_max)) #TODO: PIP.IMG check please

        if o.on_the_fly:
            o.Nf_gcf_backward = o.Nf_vis_backward
            o.Nf_gcf_predict  = Lambda(bmax, o.Nf_vis_predict)
            o.Tkernel_backward = o.Tcoal_backward
            o.Tkernel_predict  = o.Tcoal_predict
        else:
            # For both of the following, maintain distributability;
            # need at least minimum_channels (500) kernels.
            o.Nf_gcf_backward = Lambda(bmax, Max(o.Nf_gcf_backward_nosmear(bmax), o.minimum_channels))
            o.Nf_gcf_predict  = Lambda(bmax, Max(o.Nf_gcf_predict_nosmear(bmax),  o.minimum_channels))
            # TODO: some baseline dependent re-use limits along the
            # same lines as the frequency re-use? PIP.IMG check
            # please.
            o.Tkernel_backward = Lambda(bmax, o.Tion)
            o.Tkernel_predict  = Lambda(bmax, o.Tion)

        # The following two equations correspond to Eq. 35
        bcount = Symbol('bcount')
        o.Rccf_backward = o.Nmajor * o.Npp * o.Nbeam * Equations._sum_bl_bins(o, bcount, bmax,
            bcount * 5. * o.Nf_gcf_backward(bmax) * o.Nfacet**2 *
            o.Ncvff_backward(bmax)**2 * log(o.Ncvff_backward(bmax), 2) *
            o.Nmm / o.Tkernel_backward(bmax))
        o.Rccf_predict  = o.Nmajor * o.Npp * o.Nbeam * Equations._sum_bl_bins(o, bcount, bmax,
            bcount * 5. * o.Nf_gcf_predict(bmax) *
            o.Ncvff_predict(bmax)**2 * log(o.Ncvff_predict(bmax), 2) *
            o.Nmm / o.Tkernel_predict(bmax))
        # TODO we assume Nfacet=1 for predict step, so do we need
        # Nfacet^2 multiplier in here? PIP.IMG check please

        o.Rflop_conv = o.Rccf_backward + o.Rccf_predict

    @staticmethod
    def _apply_phrot_equations(o):
        """Phase rotation (for the faceting)"""

        # Eq. 29. The sign() statement below serves as an "if > 1" statement for this symbolic equation.
        # 25 FLOPS per visiblity. Only do it if we need to facet.
        # dPDR TODO: check line below - is it correct if we don't facet in the predict step? Refer to diagram
        bcount = Symbol('bcount')
        bmax = Symbol('bmax')
        o.Rflop_phrot = \
            sign(o.Nfacet - 1) * 25 * o.Nmajor * o.Npp * o.Nbeam * o.Nfacet ** 2 * \
            Equations._sum_bl_bins(o, bcount, bmax, o.Nvis_backward(bcount, bmax)) # this line was: o.Nvis_predict(bcount, bmax) + o.Nvis_backward(bcount, bmax)

    @staticmethod
    def _apply_flop_equations(o):
        """Calculate overall flop rate"""

        # revised Eq. 30
        o.Rflop = o.Rflop_grid + o.Rflop_fft + o.Rflop_proj + o.Rflop_conv + o.Rflop_phrot

    @staticmethod
    def _apply_io_equations(o):
        """
        Compute the Buffer sizes

        References: Section 12.15 in PDR05
        """

        bcount = Symbol('bcount')
        bmax = Symbol('bmax')
        o.Mw_cache = \
            o.Nbeam * o.Nf_vis_predict * 8.0 * (o.Qgcf ** 3) * \
            Equations._sum_bl_bins(o, bcount, bmax,
                o.Ngw_predict(bmax) ** 3)
            # Eq 48. TODO: re-implement this equation within a better description of where kernels will be stored etc.
        # Note the factor 2 in the line below -- we have a double buffer
        # (allowing storage of a full observation while simultaneously capturing the next)
        # TODO: The o.Nbeam factor in eqn below is not mentioned in PDR05 eq 49. Why? It is in eqn.2 though.
        o.Mbuf_vis = 2 * o.Npp * o.Nvis_predict_no_averaging * o.Nbeam * o.Mvis * o.Tobs  # Eq 49

        # added o.Nfacet dependence; changed Nmajor factor to Nmajor+1 as part of post PDR fixes.
        # TODO: Differs quite substantially from Eq 50, by merit of the Nbeam and Npp, as well as Nfacet ** 2 factors.
        # TODO: PDR05 lacking in this regard and must be updated.
        # TODO: is this correct if we have only got facets for the backward step and use Nfacet=1 for predict step?
        o.Rio = o.Nbeam * o.Npp * (1 + o.Nmajor) * o.Nvis_predict_no_averaging * o.Mvis * o.Nfacet ** 2  # Eq 50
