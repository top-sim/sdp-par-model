"""
This class contains the actual equations that are used to compute the telescopes' performance values and computional
requirements from the supplied basic parameters defined in ParameterDefinitions.

This class contains a method for (symbolically) computing derived parameters using imaging equations described
in PDR05 (version 1.85).
"""

from sympy import log, Min, Max, sqrt, floor, sign, Symbol, Lambda, Add
from numpy import pi, round
import math
from parameter_definitions import Pipelines, Products
from parameter_definitions import ParameterContainer

class Equations:
    def __init__(self):
        pass

    @staticmethod
    def apply_imaging_equations(telescope_parameters, pipeline,
                                bl_dep_time_av, bins, binfracs,
                                on_the_fly=False, scale_predict_by_facet=True,
                                verbose=False):
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
        @param pipeline: The pipeline
        @param bl_dep_time_av: True iff baseline dependent time
            averaging should be used.
        @param on_the_fly: True iff using on-the-fly kernels
        @param scale_predict_by_facet: True iif the predict phase
            scale as the facet FOV (as for backward)
        @param verbose: displays verbose command-line output

        """
        o = telescope_parameters  # Used for shorthand in the equations below
        assert isinstance(o, ParameterContainer)
        assert hasattr(o, "c")  # Checks initialization by proxy of whether the speed of light is defined

        # Store parameters
        o.set_param('pipeline', pipeline)  # e.g. ICAL, DPprepA
        o.bl_dep_time_av = bl_dep_time_av
        o.Bmax_bins = list(bins)
        o.frac_bins = list(binfracs)
        o.on_the_fly = on_the_fly
        o.scale_predict_by_facet = scale_predict_by_facet

        # Check parameters
        if hasattr(o, 'Tobs') and (o.Tobs < 10.0):
            o.Tsnap_min = o.Tobs
            if verbose:
                print 'Warning: Tsnap_min overwritten in equations.py file because observation was shorter than 10s'

        # Derive simple parameters. Note that we ignore some baselines
        # when Bmax is lower than the telescope's maximum.
        #o.Na = 512 * (35.0/o.Ds)**2 #Hack to make station diameter and station number inter-related...comment it out after use
        o.nbaselines = sum(o.frac_bins) * o.Na * (o.Na - 1) / 2.0

        o.wl_max = o.c / o.freq_min  # Maximum Wavelength
        o.wl_min = o.c / o.freq_max  # Minimum Wavelength
        o.wl = 0.5 * (o.wl_max + o.wl_min)  # Representative Wavelength

        # This set of methods must be executed in the defined sequence since
        # some values in sequence. This is ugly and we should fix it one day.
        # Apply general imaging equations. These next 4 methods just set up
        # parameters.
        Equations._apply_image_equations(o)
        Equations._apply_channel_equations(o)
        Equations._apply_coalesce_equations(o)
        Equations._apply_geometry_equations(o)

        # Apply product equations to fill in the Rflop estimates (and others when they arrive).
        Equations._apply_ingest_equations(o)
        Equations._apply_dft_equations(o)
        Equations._apply_flag_equations(o)
        Equations._apply_calibration_equations(o)
        Equations._apply_major_cycle_equations(o)
        Equations._apply_grid_equations(o)
        Equations._apply_fft_equations(o)
        Equations._apply_reprojection_equations(o)
        Equations._apply_spectral_fitting_equations(o)
        Equations._apply_source_find_equations(o)
        Equations._apply_kernel_equations(o)
        Equations._apply_phrot_equations(o)
        Equations._apply_minor_cycle_equations(o)

        # Apply summary equations
        Equations._apply_flop_equations(o)
        Equations._apply_io_equations(o)

        return o

    @staticmethod
    def _sum_bl_bins(o, bcount, b, expr):
        """Helper for dealing with baseline dependence. For a term
        depending on the given symbols, "sum_bl_bins" will build a
        sum term over all baseline bins."""

        # Replace in concrete values for baseline fractions and
        # length. Using Lambda here is a solid 25% faster than
        # subs(). Unfortunately very slow nonetheless...
        results = []
        lam = Lambda((bcount,b), expr)
        nbaselines_full = o.Na * (o.Na - 1) / 2.0
        for (frac_val, bmax_val) in zip(o.frac_bins, o.Bmax_bins):
            results.append(lam(frac_val*nbaselines_full, bmax_val))
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
        # Predict fov and number of pixels depends on whether we facet
        if o.scale_predict_by_facet:
            o.Theta_fov_predict = o.Theta_fov
            o.Nfacet_predict = o.Nfacet
            o.Npix_linear_predict = o.Npix_linear
        else:
            o.Theta_fov_predict = o.Total_fov
            o.Nfacet_predict = 1
            o.Npix_linear_predict = o.Npix_linear_total_fov
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

        b = Symbol('b')
        log_wl_ratio = log(o.wl_max / o.wl_min)

        # The two equations below => combination of Eq 4 and Eq 5 for
        # full and facet FOV at max baseline respectively.  These
        # limit bandwidth smearing to within a fraction
        # (epsilon_f_approx) of a uv cell.
        # Done: PDR05 Eq 5 says o.Nf = log_wl_ratio / (1 + 0.6 * o.Ds / (o.Bmax * o.Q_fov * o.Qbw)).
        # This is fine - substituting in the equation for theta_fov shows it is indeed correct.
        #Use full FoV for de-grid (predict) for high accuracy

        o.Nf_no_smear = \
            log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax * o.Total_fov * o.Qbw)))
        o.Nf_no_smear_backward = Lambda(b,
            log_wl_ratio / log(1 + (3 * o.wl / (2. * b * o.Theta_fov * o.Qbw))))
        o.Nf_no_smear_predict = Lambda(b,
            log_wl_ratio / log(1 + (3 * o.wl / (2. * b * o.Theta_fov_predict * o.Qbw))))

        # The number of visibility channels used in each direction
        # (includes effects of averaging). Bound by minimum parallism
        # and input channel count.
        o.Nf_vis = \
            Min(Max(o.Nf_out, o.Nf_no_smear),o.Nf_max)
        o.Nf_vis_backward = Lambda(b,
            Min(Max(o.Nf_out, o.Nf_no_smear_backward(b)),o.Nf_max))
        o.Nf_vis_predict = Lambda(b,
            Min(Max(o.Nf_out, o.Nf_no_smear_predict(b if o.scale_predict_by_facet else o.Bmax)),
                o.Nf_max))

    @staticmethod
    def _apply_coalesce_equations(o):
        """
        Determines amount of coalescing of visibilities in time.
        """

        # correlator output averaging time scaled for max baseline.
        o.Tdump_scaled = o.Tdump_ref * o.B_dump_ref / o.Bmax
        b = Symbol('b')
        o.combine_time_samples = Lambda(b,
            Max(floor(o.epsilon_f_approx * o.wl /
                      (o.Total_fov * o.Omega_E * b * o.Tdump_scaled)), 1.))
        # coalesce visibilities in time.
        o.Tcoal_skipper = Lambda(b,
            o.Tdump_scaled * o.combine_time_samples(b))

        if o.bl_dep_time_av:
            # For backward step at gridding only, allow coalescance of
            # visibility points at Facet FoV smearing limit only for
            # BLDep averaging case.
            o.Tcoal_backward = Lambda(b,
                Min(o.Tcoal_skipper(b) * o.Nfacet/(1+o.using_facet_overlap_frac), o.Tion))
            if o.scale_predict_by_facet:
                o.Tcoal_predict = Lambda(b, o.Tcoal_backward(b))
            else:
                # Don't let any bl-dependent time averaging be for
                # longer than either 1.2s or Tion. ?Why 1.2s?
                o.Tcoal_predict = Lambda(b, Min(o.Tcoal_skipper(b), 1.2, o.Tion))
        else:
            o.Tcoal_predict = Lambda(b, o.Tdump_scaled)
            o.Tcoal_backward = Lambda(b, o.Tdump_scaled)

        # Eq. 31 Visibility rate for backward step, allow coalescing
        # in time and freq prior to gridding
        bcount = Symbol('bcount')
        o.Nvis_backward = Lambda((bcount, b),
            bcount * o.Nf_vis_backward(b) / o.Tcoal_backward(b))
        # Eq. 31 Visibility rate for predict step
        o.Nvis_predict = Lambda((bcount, b),
            bcount * o.Nf_vis_predict(b) / o.Tcoal_predict(b))
        # Total input visibility rate
        o.Nvis = o.nbaselines * o.Nf_vis / o.Tdump_scaled

        # The line above uses Tdump_scaled independent of whether
        # BLDTA is used.  This is because BLDTA is only done for
        # gridding, and doesn't affect the amount of data to be
        # buffered

    @staticmethod
    def _apply_geometry_equations(o):

        # ===============================================================================================
        # PDR05 Sec 12.2 - 12.5
        # ===============================================================================================

        b = Symbol('b')
        o.DeltaW_Earth = Lambda(b, b ** 2 / (8. * o.R_Earth * o.wl))  # Eq. 19
        # TODO: in the two lines below, PDR05 uses lambda_min, not mean.
        # Eq. 26 : W-deviation for snapshot.
        o.DeltaW_SShot = Lambda(b, b * o.Omega_E * o.Tsnap / (2. * o.wl))
        o.DeltaW_max = Lambda(b, o.Qw * Max(o.DeltaW_SShot(b), o.DeltaW_Earth(b)))

        # Eq. 25, w-kernel support size **Note difference in cellsize assumption**
        def Ngw(deltaw, fov):
            return 2 * fov * sqrt((deltaw * fov / 2.) ** 2 +
                                  (deltaw**1.5 * fov / (2 * pi * o.epsilon_w)))
        o.Ngw_backward = Lambda(b, Ngw(o.DeltaW_max(b), o.Theta_fov))
        o.Ngw_predict = Lambda(b, Ngw(o.DeltaW_max(b), o.Theta_fov_predict))

        # TODO: Check split of kernel size for backward and predict steps.
        # squared linear size of combined W and A kernels; used in eqs 23 and 32
        o.Nkernel2_backward = Lambda(b, o.Ngw_backward(b) ** 2 + o.Naa ** 2)
        o.Nkernel_AW_backward = Lambda(b, (o.Ngw_backward(b) ** 2 + o.Naa ** 2)**0.5)
        # squared linear size of combined W and A kernels; used in eqs 23 and 32
        o.Nkernel2_predict = Lambda(b, o.Ngw_predict(b) ** 2 + o.Naa ** 2)
        o.Nkernel_AW_predict = Lambda(b, (o.Ngw_predict(b) ** 2 + o.Naa ** 2)**0.5)
        if o.on_the_fly:
            o.Qgcf = 1.0

        # Eq. 23 : combined kernel support size and oversampling
        o.Ncvff_backward = Lambda(b,
            sqrt(o.Nkernel2_backward(b))*o.Qgcf)
        # Eq. 23 : combined kernel support size and oversampling
        o.Ncvff_predict = Lambda(b,
            sqrt(o.Nkernel2_predict(b))*o.Qgcf)

    @staticmethod
    def _apply_ingest_equations(o):
        """ Ingest equations """

        # Need autocorrelations as well
        o.Nvis_receive = ((o.nbaselines + o.Na) * o.Nbeam * o.Npp) / o.Tdump_ref

        if o.pipeline == Pipelines.Ingest:
            receiveflop = 2 * o.rma * o.Nf_max * o.Npp * o.Nbeam + 1000 * o.Na * o.minimum_channels * o.Nbeam
            o.set_product(Products.Receive, Rflop=o.Nvis_receive * receiveflop)
            o.set_product(Products.Flag, Rflop=279 * o.Nvis_receive)
            o.set_product(Products.Demix, Rflop=o.cma * o.Nvis_receive * o.Ndemix * (o.NA * (o.NA + 1) / 2.0))
            o.set_product(Products.Average, Rflop=o.cma * o.Nvis_receive)

    @staticmethod
    def _apply_flag_equations(o):
        """ Flagging equations for non-ingest pipelines"""

        if not (o.pipeline == Pipelines.Ingest):
            o.set_product(Products.Flag, Rflop=279 * o.Nvis)

    @staticmethod
    def _apply_grid_equations(o):
        """ Grid """
        if o.pipeline in Pipelines.imaging:
            """ For the ASKAP MSMFS, we grid all data for each taylor term
            with polynominal of delta freq/freq
            """
            b = Symbol('b')
            bcount = Symbol('bcount')
            o.Ntaylor_backward = 1
            o.Ntaylor_predict = 1
            if o.pipeline == Pipelines.DPrepA:
                o.Ntaylor_backward = o.number_taylor_terms
                o.Ntaylor_predict = o.number_taylor_terms
            o.Rgrid_backward_task = Lambda((bcount, b),
                o.cma * o.Nmm * o.Nmajortotal * bcount * o.Nkernel2_backward(b) *
                o.Tsnap / o.Tcoal_backward(b))
            o.Rgrid_backward = \
                o.cma * o.Nmm * o.Nmajortotal * o.Npp * o.Nbeam * o.Ntaylor_backward * o.Nfacet**2 * \
                Equations._sum_bl_bins(o, bcount, b,
                    o.Nvis_backward(bcount, b) * o.Nkernel2_backward(b))
            o.Rgrid_predict_task = Lambda((bcount, b),
                o.cma * o.Nmm * bcount * o.Nkernel2_predict(b) *
                o.Tsnap / o.Tcoal_predict(b))
            o.Rgrid_predict = \
                o.cma * o.Nmm * o.Nmajortotal * o.Npp * o.Nbeam * o.Ntaylor_predict * o.Nfacet_predict**2 * \
                Equations._sum_bl_bins(o, bcount, b,
                    o.Nvis_predict(bcount, b) * o.Nkernel2_predict(b))

            o.Rflop_grid = o.Rgrid_backward + o.Rgrid_predict
            o.set_product(Products.Grid, Rflop=o.Rgrid_backward)
            o.set_product(Products.Degrid, Rflop=o.Rgrid_predict)

    @staticmethod
    def _apply_fft_equations(o):
        """ FFT """
        if o.pipeline in Pipelines.imaging:
            # FFT:
            # ---

            # Eq. 33, per output grid (i.e. frequency)
            # These are real-to-complex for which the prefactor in the FFT is 2.5
            # TODO: Note the Nf_out factor is only in the backward step of the final cycle.
            o.Rfft_backward = 2.5 * o.Nfacet**2 * o.Npix_linear ** 2 * log(o.Npix_linear**2, 2) / o.Tsnap
            # Eq. 33 per predicted grid (i.e. frequency)
            o.Rfft_predict = 2.5 * o.Nfacet_predict**2 * o.Npix_linear_predict**2 * \
                             log(o.Npix_linear_predict**2, 2) / o.Tsnap

            if o.Nf_FFT_backward > 0:
                o.Rflop_fft_bw = o.Npp * o.Nbeam * o.Nmajortotal * o.Nf_FFT_backward * o.Rfft_backward
                o.set_product(Products.FFT, Rflop=o.Rflop_fft_bw)
            if o.Nf_FFT_predict > 0:
                o.Rflop_fft_predict = o.Npp * o.Nbeam * o.Nmajortotal * o.Nf_FFT_predict * o.Rfft_predict
                o.set_product(Products.IFFT, Rflop=o.Rflop_fft_predict)

    @staticmethod
    def _apply_reprojection_equations(o):

        # Re-Projection:
        # -------------
        if o.pipeline in Pipelines.imaging:
        
            # We do 2*o.Nmajortotal*(Tobs/Tsnap) entire image reprojections (i.e. both directions)
            o.Rrp = 2.0  * o.rma * o.Nmajortotal * 50. * o.Nfacet**2 * o.Npix_linear ** 2 / o.Tsnap  # Eq. 34

            o.Nf_proj = o.Nf_FFT_backward
            if o.pipeline == Pipelines.DPrepA_Image:
                o.Nf_proj = o.number_taylor_terms

            o.Rflop_proj = o.Rrp * o.Nbeam * o.Npp * o.Nf_proj

            if o.pipeline != Pipelines.Fast_Img: # (Consistent with PDR05 280115)
                o.set_product(Products.Reprojection, Rflop=o.Rflop_proj)

    @staticmethod
    def _apply_spectral_fitting_equations(o):

        if o.pipeline == Pipelines.DPrepA_Image:
            o.Rflop_fitting = o.rma * o.Nmajortotal * o.Nbeam * o.Npp * o.number_taylor_terms * \
                              (o.Nf_FFT_backward + o.Nf_FFT_predict) * o.Npix_linear_total_fov ** 2 \
                              / o.Tobs
            o.set_product(Products.Image_Spectral_Fitting, Rflop=o.Rflop_fitting)

    @staticmethod
    def _apply_minor_cycle_equations(o):

         if o.pipeline in Pipelines.imaging:
            #
            # Minor cycles
            # -------------
            if o.pipeline in (Pipelines.ICAL, Pipelines.DPrepA, Pipelines.DPrepA_Image):
                Rflop_deconv_common = o.rma * o.Nmajortotal * o.Nbeam * o.Npp * o.Nminor / o.Tobs
                # Search only on I_0
                o.Rflop_identify_component = Rflop_deconv_common * (o.Npix_linear * o.Nfacet)**2 
                # Subtract on all scales and 
                o.Rflop_subtract_image_component = o.Nscales * Rflop_deconv_common * o.Npatch**2 
                o.set_product(Products.Subtract_Image_Component, Rflop=o.Rflop_subtract_image_component)
                o.set_product(Products.Identify_Component, Rflop=o.Rflop_identify_component)
            elif o.pipeline in (Pipelines.DPrepB, Pipelines.DPrepC):
                Rflop_deconv_common = o.rma * o.Nmajortotal * o.Nbeam * o.Npp * o.Nminor / o.Tobs
                # Always search in all frequency space
                o.Rflop_identify_component = o.Nf_out * Rflop_deconv_common * (o.Npix_linear * o.Nfacet)**2 
                # Subtract on all scales and only one frequency
                o.Rflop_subtract_image_component = o.Nscales * Rflop_deconv_common * o.Npatch**2 
                o.set_product(Products.Subtract_Image_Component, Rflop=o.Rflop_subtract_image_component)
                o.set_product(Products.Identify_Component, Rflop=o.Rflop_identify_component)

    @staticmethod
    def _apply_calibration_equations(o):
        # We do one calibration to start with (using the original LSM from the GSM and then we do
        # Nselfcal more.
        Rflop_solve_common = ((o.Nselfcal + 1) * o.Nvis * o.Nbeam * 48 * o.Na * o.Na * o.Nsolve / o.nbaselines /o.Nf_max)
        # ICAL solves for all terms but on different time scales. These should be set for context in the HPSOs.
        if o.pipeline == Pipelines.ICAL:
            o.Rflop_solve = o.Tobs* (1.0 / o.tICAL_G + o.Nf_out / o.tICAL_B + o.NIpatches * o.Na / o.tICAL_I)
            o.set_product(Products.Solve, Rflop=Rflop_solve_common * o.Rflop_solve)

        # RCAL solves for G only
        if o.pipeline == Pipelines.RCAL:
            o.Rflop_solve = Rflop_solve_common * o.Tobs / o.tRCAL_G               
            o.set_product(Products.Solve, Rflop=o.Rflop_solve)
 
    @staticmethod
    def _apply_dft_equations(o):
        if o.pipeline in Pipelines.imaging:
            # If the selfcal loop is embedded, we only need to do this once but since we do 
            # an update of the model every selfcal, we need to do it every selfcal.
            # We assume that these operations counts are correct for FMULT-less
            o.Rflop_dft = (o.Nselfcal + 1) * o.Nvis * o.Npp * o.Nbeam * (64 * o.Na * o.Na * o.Nsource + 242 * o.Na * o.Nsource + 128 * o.Na * o.Na) / o.nbaselines
            o.set_product(Products.DFT, Rflop=o.Rflop_dft)

    @staticmethod
    def _apply_source_find_equations(o):
        """Rough estimate of source finding flops"""
        if o.pipeline == Pipelines.ICAL:
            # We need to fit 6 degrees of freedom to 100 points so we have 600 FMults . Ignore for the moment Theta_beam
            # the solution of these normal equations. This really is a stopgap. We need an estimate for 
            # a non-linear solver.
            o.Rflop_source_find=o.rma * 6 * 100 *o.Nselfcal*o.Nsource_find_iterations*o.rho_gsm*o.Theta_fov**2 / o.Tobs
            o.set_product(Products.Source_Find, Rflop=o.Rflop_source_find)

    @staticmethod
    def _apply_major_cycle_equations(o):

        # Note that we assume this is done for every Selfcal and Major Cycle
        # ---
        if o.pipeline in Pipelines.imaging:
            o.Rflop_subtractvis = o.cma *  o.Nmajortotal * o.Nvis * o.Npp * o.Nbeam
            o.set_product(Products.Subtract_Visibility, Rflop=o.Rflop_subtractvis)

    @staticmethod
    def _apply_kernel_equations(o):
        """
        Generate Convolution kernels
        """
        if o.pipeline in Pipelines.imaging:

            b = Symbol('b')
            o.dfonF_backward = Lambda(b, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_backward(b))))
            o.dfonF_predict = Lambda(b, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_predict(b))))

            # allow uv positional errors up to o.epsilon_f_approx *
            # 1/Qkernel of a cell from frequency smearing.(But not more
            # than Nf_max channels...)
            o.Nf_gcf_backward_nosmear = Lambda(b,
                Min(log(o.wl_max / o.wl_min) /
                    log(o.dfonF_backward(b) + 1.), o.Nf_max)) #TODO: PIP.IMG check please
            o.Nf_gcf_predict_nosmear  = Lambda(b,
                Min(log(o.wl_max / o.wl_min) /
                    log(o.dfonF_predict(b) + 1.), o.Nf_max)) #TODO: PIP.IMG check please

            if o.on_the_fly:
                o.Nf_gcf_backward = o.Nf_vis_backward
                o.Nf_gcf_predict  = o.Nf_vis_predict
                o.Tkernel_backward = o.Tcoal_backward
                o.Tkernel_predict  = o.Tcoal_predict
            else:
                # For both of the following, maintain distributability;
                # need at least minimum_channels (500) kernels.
                o.Nf_gcf_backward = Lambda(b, Max(o.Nf_gcf_backward_nosmear(b), o.minimum_channels))
                o.Nf_gcf_predict  = Lambda(b, Max(o.Nf_gcf_predict_nosmear(b),  o.minimum_channels))
                # TODO: some baseline dependent re-use limits along the
                # same lines as the frequency re-use? PIP.IMG check
                # please.
                o.Tkernel_backward = Lambda(b, o.Tion)
                o.Tkernel_predict  = Lambda(b, o.Tion)

            # The following two equations correspond to Eq. 35
            bcount = Symbol('bcount')
            o.Rccf_backward_task = Lambda(b,
                5. * o.Nmm * o.Ncvff_backward(b)**2 * log(o.Ncvff_backward(b), 2))
            o.Rccf_backward = o.Nmajortotal * o.Npp * o.Nbeam * Equations._sum_bl_bins(o, bcount, b,
               bcount * 5. * o.Nf_gcf_backward(b) * # o.Nfacet**2 *
               o.Ncvff_backward(b)**2 * log(o.Ncvff_backward(b), 2) *
               o.Nmm / o.Tkernel_backward(b))
            o.Rccf_predict_task = Lambda(b,
                5. * o.Nmm * o.Ncvff_predict(b)**2 * log(o.Ncvff_predict(b), 2))
            o.Rccf_predict  = o.Nmajortotal * o.Npp * o.Nbeam * Equations._sum_bl_bins(o, bcount, b,
                bcount * 5. * o.Nf_gcf_predict(b) * # o.Nfacet_predict**2 *
                o.Ncvff_predict(b)**2 * log(o.Ncvff_predict(b), 2) *
                o.Nmm / o.Tkernel_predict(b))

            o.Rflop_conv = (o.Rccf_backward + o.Rccf_predict)
            o.set_product(Products.Gridding_Kernel_Update, Rflop=o.Rflop_conv)

    @staticmethod
    def _apply_phrot_equations(o):
        """Phase rotation (for the faceting)"""

        if o.pipeline in Pipelines.imaging:
            # Eq. 29. The sign() statement below serves as an "if > 1" statement for this symbolic equation.
            # 25 FLOPS per visiblity. Only do it if we need to facet.
            # dPDR TODO: check line below - is it correct if we don't facet in the predict step? Refer to diagram
            bcount = Symbol('bcount')
            b = Symbol('b')
#             o.Rflop_phrot_predict_task = Lambda((bcount,b), \
#                 sign(o.Nfacet - 1) * 25 * o.Nvis_predict(bcount, b) * o.Tsnap / o.Nf_vis_predict(b))
#             o.Rflop_phrot_backward_task = Lambda((bcount, b), \
#                 sign(o.Nfacet - 1) * 25 * o.Nvis_backward(bcount, b) * o.Tsnap / o.Nf_vis_backward(b))
#             o.Rflop_phrot = \
#                 sign(o.Nfacet - 1) * 25 * o.Nmajortotal * o.Npp * o.Nbeam * o.Nfacet ** 2 * \
#                 Equations._sum_bl_bins(o, bcount, b, o.Nvis_backward(bcount, b))
#             if o.scale_predict_by_facet:
#                 o.Rflop_phrot += \
#                     sign(o.Nfacet - 1) * 25 * o.Nmajortotal * o.Npp * o.Nbeam * o.Nfacet ** 2 * \
#                     Equations._sum_bl_bins(o, bcount, b, o.Nvis_predict(bcount, b))

# These equations need to be modified to match the non-task versions
            o.Rflop_phrot_predict_task = Lambda((bcount,b), \
                sign(o.Nfacet - 1) * 25 * o.Nvis * o.Tsnap / o.Nf_vis_predict(b))
            o.Rflop_phrot_backward_task = Lambda((bcount, b), \
                sign(o.Nfacet - 1) * 25 * o.Nvis * o.Tsnap / o.Nf_vis_backward(b))
# Non-task
            o.Rflop_phrot = \
                sign(o.Nfacet - 1) * 25 * o.Nmajortotal * o.Npp * o.Nbeam * o.Nfacet ** 2 * \
                o.Nvis
            if o.scale_predict_by_facet:
                o.Rflop_phrot += \
                    sign(o.Nfacet - 1) * 25 * o.Nmajortotal * o.Npp * o.Nbeam * o.Nfacet ** 2 * \
                    o.Nvis
            o.set_product(Products.PhaseRotation, Rflop=o.Rflop_phrot)

    @staticmethod
    def _apply_flop_equations(o):
        """Calculate overall flop rate"""

        # revised Eq. 30
        o.Rflop = sum(o.get_products('Rflop').values())

        # Calculate interfacet IO rate for faceting: TCC-SDP-151123-1-1 rev 1.1
        o.Rinterfacet = 2 * o.Nmajortotal * min(3.0, 2.0 + 18.0 * o.facet_overlap_frac) * (o.Nfacet * o.Npix_linear)**2 * o.Nf_out * 4  / o.Tobs


    @staticmethod
    def _apply_io_equations(o):
        """
        Compute the Buffer sizes

        References: Section 12.15 in PDR05
        """

        bcount = Symbol('bcount')
        b = Symbol('b')
        o.Mw_cache = \
            o.Ncbytes * o.Nbeam * (o.Qgcf ** 3) * \
            Equations._sum_bl_bins(o, bcount, b,
                o.Nf_vis_predict(b) * o.Ngw_predict(b) ** 3)
            # Eq 48. TODO: re-implement this equation within a better description of where kernels will be stored etc.
        # Note the factor 2 in the line below -- we have a double buffer
        # (allowing storage of a full observation while simultaneously capturing the next)
        # TODO: The o.Nbeam factor in eqn below is not mentioned in PDR05 eq 49. Why? It is in eqn.2 though.
        o.Mbuf_vis = 2 * o.Npp * o.Nvis * o.Nbeam * o.Mvis * o.Tobs  # Eq 49

        # added o.Nfacet dependence; changed Nmajor factor to Nmajor+1 as part of post PDR fixes.
        # TODO: Differs quite substantially from Eq 50, by merit of the Nbeam and Npp, as well as Nfacet ** 2 factors.
        # TODO: PDR05 lacking in this regard and must be updated.
        # This is correct if we have only got facets for the backward step and use Nfacet=1 for predict step: TJC see TCC-SDP-151123-1-1
        # It probably can go much smaller, though: see SDPPROJECT-133
        o.Rio = o.Nbeam * o.Npp * (1 + o.Nmajortotal) * o.Nvis * o.Mvis * o.Nfacet ** 2  # Eq 50
