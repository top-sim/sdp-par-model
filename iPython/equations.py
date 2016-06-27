"""
This class contains the actual equations that are used to compute the telescopes' performance values and computional
requirements from the supplied basic parameters defined in ParameterDefinitions.

This class contains a method for (symbolically) computing derived parameters using imaging equations described
in PDR05 (version 1.85).
"""

from __future__ import print_function

from sympy import log, Min, Max, sqrt, floor, sign, ceiling, Symbol, Lambda, Add, Sum, Mul
from numpy import pi, round
import math
from parameter_definitions import Pipelines, Products
from parameter_definitions import ParameterContainer, BLDep
import warnings

# Check sympy compatibility
import sympy
if sympy.__version__ == "1.0":
    raise Exception("SymPy version 1.0 is broken. Please either upgrade or downgrade your version!")

def blsum(b, expr):
    bcount = Symbol('bcount')
    return BLDep((b, bcount), bcount * expr)

class Equations:
    def __init__(self):
        pass

    @staticmethod
    def apply_imaging_equations(telescope_parameters, pipeline,
                                blcoal, bins, binfracs,
                                on_the_fly=False, scale_predict_by_facet=True,
                                verbose=False, symbolify=''):
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
        @param blcoal: True iff baseline dependent coalescing should be used.
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
        o.blcoal = blcoal
        o.Bmax_bins = list(bins)
        o.frac_bins = list(binfracs)
        o.on_the_fly = on_the_fly
        o.scale_predict_by_facet = scale_predict_by_facet

        # Check parameters
        if hasattr(o, 'Tobs') and (o.Tobs < 10.0):
            o.Tsnap_min = o.Tobs
            if verbose:
                print('Warning: Tsnap_min overwritten in equations.py file because observation was shorter than 10s')

        # Load common equations.
        Equations._apply_common_equations(o)

        # If requested, we replace all parameters so far with symbols,
        # so product equations are purely symbolic.
        if symbolify == 'all':
            o.symbolify()
            o.Bmax_bins = Symbol("B_max")

        # This set of methods must be executed in the defined sequence since
        # some values in sequence. This is ugly and we should fix it one day.
        # Apply general imaging equations. These next 4 methods just set up
        # parameters.
        Equations._apply_image_equations(o)
        Equations._apply_channel_equations(o)
        Equations._apply_coalesce_equations(o)
        Equations._apply_geometry_equations(o)

        # If requested, we replace all parameters so far with symbols,
        # so product equations are purely symbolic.
        if symbolify == 'product':
            o.symbolify()
            o.Bmax_bins = Symbol("B_max")

        # Apply product equations to fill in the Rflop estimates (and others when they arrive).
        Equations._apply_ingest_equations(o)
        Equations._apply_dft_equations(o)
        Equations._apply_flag_equations(o)
        Equations._apply_correct_equations(o)
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
    def _apply_common_equations(o):
        """
        Calculate simple derived values that are going to get used fairly often.
        """

        # Derive simple parameters. Note that we ignore some baselines
        # when Bmax is lower than the telescope's maximum.
        #o.Na = 512 * (35.0/o.Ds)**2 #Hack to make station diameter and station number inter-related...comment it out after use
        o.nbaselines_full = o.Na * (o.Na - 1) / 2.0
        o.nbaselines = sum(o.frac_bins) * o.nbaselines_full
        o.Tdump_scaled = o.Tdump_ref * o.B_dump_ref / o.Bmax

        # Facet overlap is only needed if Nfacet > 1
        o.using_facet_overlap_frac=sign(o.Nfacet - 1)*o.facet_overlap_frac

        # Wavelengths
        o.wl_max = o.c / o.freq_min  # Maximum Wavelength
        o.wl_min = o.c / o.freq_max  # Minimum Wavelength
        o.wl = 0.5 * (o.wl_max + o.wl_min)  # Representative Wavelength

        # TODO: In line below: PDR05 uses *wl_max* instead of wl. Also
        # uses 7.6 instead of 7.66. Is this correct?
        o.Number_imaging_subbands = ceiling(log(o.wl_max/o.wl_min)/log(o.max_subband_freq_ratio))
        o.subband_frequency_ratio = (o.wl_max/o.wl_min)**(1./o.Number_imaging_subbands)

    @staticmethod
    def _apply_image_equations(o):
        """
        Calculate image parameters, such as resolution and size

        References: PDR05 (version 1.85) Sec 9.2
        """

        # max subband wavelength to set image FoV
        o.wl_sb_max = o.wl *sqrt(o.subband_frequency_ratio)
        # min subband wavelength to set pixel size
        o.wl_sb_min = o.wl_sb_max / o.subband_frequency_ratio

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

    @staticmethod
    def _apply_channel_equations(o):
        """
        Determines the number of frequency channels to use in backward &
        predict steps.

        References: PDR05 Sec 9.1
        """

        b = Symbol('b')
        log_wl_ratio = log(o.wl_max / o.wl_min)

        # expansion of sine solves eps = arcsinc(1/amp_f_max).
        o.epsilon_f_approx = sqrt(6 * (1 - (1. / o.amp_f_max)))
        # See notes on https://confluence.ska-sdp.org/display/PIP/Frequency+resolution+and+smearing+effects+in+the+iPython+SDP+Parametric+model
        o.Qbw = 1.47 / o.epsilon_f_approx

        # The two equations below => combination of Eq 4 and Eq 5 for
        # full and facet FOV at max baseline respectively.  These
        # limit bandwidth smearing to within a fraction
        # (epsilon_f_approx) of a uv cell.
        # Done: PDR05 Eq 5 says o.Nf = log_wl_ratio / (1 + 0.6 * o.Ds / (o.Bmax * o.Q_fov * o.Qbw)).
        # This is fine - substituting in the equation for theta_fov shows it is indeed correct.
        #Use full FoV for de-grid (predict) for high accuracy

        o.Nf_no_smear = \
            log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax * o.Total_fov * o.Qbw)))
        o.Nf_no_smear_backward = BLDep(b,
            log_wl_ratio / log(1 + (3 * o.wl / (2. * b * o.Theta_fov * o.Qbw))))
        o.Nf_no_smear_predict = BLDep(b,
            log_wl_ratio / log(1 + (3 * o.wl / (2. * b * o.Theta_fov_predict * o.Qbw))))

        # The number of visibility channels used in each direction
        # (includes effects of averaging). Bound by minimum parallism
        # and input channel count.
        o.Nf_vis = \
            Min(Max(o.Nf_out, o.Nf_no_smear),o.Nf_max)
        o.Nf_vis_backward = BLDep(b,
            Min(Max(o.Nf_out, o.Nf_no_smear_backward(b)),o.Nf_max))
        o.Nf_vis_predict = BLDep(b,
            Min(Max(o.Nf_out, o.Nf_no_smear_predict(b if o.scale_predict_by_facet else o.Bmax)),
                o.Nf_max))

    @staticmethod
    def _apply_coalesce_equations(o):
        """
        Determines amount of coalescing of visibilities in time.
        """

        # correlator output averaging time scaled for max baseline.
        b = Symbol('b')
        o.combine_time_samples = BLDep(b,
            Max(floor(o.epsilon_f_approx * o.wl /
                      (o.Total_fov * o.Omega_E * b * o.Tdump_scaled)), 1.))

        # coalesce visibilities in time.
        o.Tcoal_skipper = BLDep(b,
            o.Tdump_scaled * o.combine_time_samples(b))

        if o.blcoal:
            # For backward step at gridding only, allow coalescance of
            # visibility points at Facet FoV smearing limit only for
            # bl-dep averaging case.
            o.Tcoal_backward = BLDep(b,
                Min(o.Tcoal_skipper(b) * o.Nfacet/(1+o.using_facet_overlap_frac), o.Tion))
            if o.scale_predict_by_facet:
                o.Tcoal_predict = BLDep(b, o.Tcoal_backward(b))
            else:
                # Don't let any bl-dependent time averaging be for
                # longer than either 1.2s or Tion. ?Why 1.2s?
                o.Tcoal_predict = BLDep(b, Min(o.Tcoal_skipper(b), 1.2, o.Tion))
        else:
            o.Tcoal_predict = BLDep(b, o.Tdump_scaled)
            o.Tcoal_backward = BLDep(b, o.Tdump_scaled)

        # Visibility rate on ingest, including autocorrelations
        o.Nvis_ingest = (o.nbaselines + o.Na) * o.Nf_max / o.Tdump_ref
        # Total visibility rate after global coalescing
        o.Nvis = o.nbaselines * o.Nf_vis / o.Tdump_scaled

        # Eq. 31 Visibility rate for backward step, allow coalescing
        # in time and freq prior to gridding
        o.Nvis_backward = blsum(b, o.Nf_vis_backward(b) / o.Tcoal_backward(b))
        # Eq. 31 Visibility rate for predict step
        o.Nvis_predict = blsum(b, o.Nf_vis_predict(b) / o.Tcoal_predict(b))

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
        o.DeltaW_Earth = BLDep(b, b ** 2 / (8. * o.R_Earth * o.wl))  # Eq. 19
        # TODO: in the two lines below, PDR05 uses lambda_min, not mean.
        # Eq. 26 : W-deviation for snapshot.
        o.DeltaW_SShot = BLDep(b, b * o.Omega_E * o.Tsnap / (2. * o.wl))
        o.DeltaW_max = BLDep(b, o.Qw * Max(o.DeltaW_SShot(b), o.DeltaW_Earth(b)))

        # Eq. 25, w-kernel support size **Note difference in cellsize assumption**
        def Ngw(deltaw, fov):
            return 2 * fov * sqrt((deltaw * fov / 2.) ** 2 +
                                  (deltaw**1.5 * fov / (2 * pi * o.epsilon_w)))
        o.Ngw_backward = BLDep(b, Ngw(o.DeltaW_max(b), o.Theta_fov))
        o.Ngw_predict = BLDep(b, Ngw(o.DeltaW_max(b), o.Theta_fov_predict))

        # TODO: Check split of kernel size for backward and predict steps.
        # squared linear size of combined W and A kernels; used in eqs 23 and 32
        o.Nkernel2_backward = BLDep(b, o.Ngw_backward(b) ** 2 + o.Naa ** 2)
        o.Nkernel_AW_backward = BLDep(b, (o.Ngw_backward(b) ** 2 + o.Naa ** 2)**0.5)
        # squared linear size of combined W and A kernels; used in eqs 23 and 32
        o.Nkernel2_predict = BLDep(b, o.Ngw_predict(b) ** 2 + o.Naa ** 2)
        o.Nkernel_AW_predict = BLDep(b, (o.Ngw_predict(b) ** 2 + o.Naa ** 2)**0.5)
        if o.on_the_fly:
            o.Qgcf = 1.0

        # Eq. 23 : combined kernel support size and oversampling
        o.Ncvff_backward = BLDep(b,
            sqrt(o.Nkernel2_backward(b))*o.Qgcf)
        # Eq. 23 : combined kernel support size and oversampling
        o.Ncvff_predict = BLDep(b,
            sqrt(o.Nkernel2_predict(b))*o.Qgcf)

    @staticmethod
    def _apply_ingest_equations(o):
        """ Ingest equations """

        if o.pipeline == Pipelines.Ingest:
            Equations._set_product(
                o, Products.Receive,
                T = o.Tsnap, N = o.Nbeam * o.minimum_channels * o.Npp,
                Rflop = 2 * o.Nvis_ingest / o.minimum_channels +
                        1000 * o.Na / o.Tdump_ref,
                Rout = o.Mvis * o.Nvis_ingest)
            Equations._set_product(
                o, Products.Flag,
                T = o.Tsnap, N = o.Nbeam * o.Npp * o.minimum_channels,
                Rflop = 279 * o.Nvis_ingest / o.minimum_channels,
                Rout = o.Mvis * o.Nvis_ingest / o.minimum_channels)
            # Ndemix is the number of time-frequency products used
            # (typically 1000) so we have to divide out the number of
            # input channels
            Equations._set_product(
                o, Products.Demix,
                T = o.Tsnap, N = o.Nbeam * o.Npp * o.minimum_channels,
                Rflop = 8 * (o.Nvis_ingest * o.Ndemix / o.Nf_max) * (o.NA * (o.NA + 1) / 2.0)
                        / o.minimum_channels,
                Rout = o.Mvis * o.Nvis / o.minimum_channels)
            Equations._set_product(
                o, Products.Average,
                T = o.Tsnap, N = o.Nbeam * o.Npp * o.minimum_channels,
                Rflop = 8 * o.Nvis_ingest / o.minimum_channels,
                Rout = o.Mvis * o.Nvis / o.minimum_channels)

    @staticmethod
    def _apply_flag_equations(o):
        """ Flagging equations for non-ingest pipelines"""

        if not (o.pipeline == Pipelines.Ingest):
            Equations._set_product(
                o, Products.Flag,
                T=o.Tsnap, N=o.Nbeam * o.minimum_channels,
                Rflop=279 * o.Nvis / o.minimum_channels,
                Rout = o.Mvis * o.Nvis / o.minimum_channels)

    @staticmethod
    def _apply_correct_equations(o):
        """ Correction of gains"""

        if not o.pipeline == Pipelines.Ingest:
            Equations._set_product(
                o, Products.Correct,
                T = o.Tsnap, N = o.Nbeam*o.Nmajortotal * o.Npp * o.minimum_channels,
                Rflop = 8 * o.Nmm * o.Nvis * o.NIpatches / o.minimum_channels,
                Rout = o.Mvis * o.Nvis / o.minimum_channels)

    @staticmethod
    def _apply_grid_equations(o):
        """ Grid """

        # For the ASKAP MSMFS, we grid all data for each taylor term
        # with polynominal of delta freq/freq
        b = Symbol('b')
        o.Ntaylor_backward = 1
        o.Ntaylor_predict = 1
        if o.pipeline == Pipelines.DPrepA:
            o.Ntaylor_backward = o.number_taylor_terms
            o.Ntaylor_predict = o.number_taylor_terms

        if not o.pipeline in Pipelines.imaging: return
        Equations._set_product(
            o, Products.Grid, T=o.Tsnap,
            N = BLDep(b, o.Nmajortotal * o.Nbeam * o.Npp * o.Ntaylor_backward *
                         o.Nfacet**2 * o.Nf_vis_backward(b)),
            Rflop = blsum(b, 8 * o.Nmm * o.Nkernel2_backward(b) / o.Tcoal_backward(b)),
            Rout = o.Mcpx * o.Npix_linear * (o.Npix_linear / 2 + 1) / o.Tsnap)
        Equations._set_product(
            o, Products.Degrid, T = o.Tsnap,
            N = BLDep(b, o.Nmajortotal * o.Nbeam * o.Npp * o.Ntaylor_predict *
                         o.Nfacet_predict**2 * o.Nf_vis_predict(b)),
            Rflop = blsum(b, 8 * o.Nmm * o.Nkernel2_predict(b) / o.Tcoal_predict(b)),
            Rout = blsum(b, o.Mvis / o.Tcoal_predict(b)))

    @staticmethod
    def _apply_fft_equations(o):
        """ FFT """

        if not o.pipeline in Pipelines.imaging: return

        b = Symbol("b")

        # Eq. 33, per output grid (i.e. frequency)
        # These are real-to-complex for which the prefactor in the FFT is 2.5
        Equations._set_product(
            o, Products.FFT, T = o.Tsnap,
            N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_FFT_backward * o.Nfacet**2,
            Rflop = 2.5 * o.Npix_linear ** 2 * log(o.Npix_linear**2, 2) / o.Tsnap,
            Rout = o.Mpx * o.Npix_linear**2 / o.Tsnap)

        # Eq. 33 per predicted grid (i.e. frequency)
        Equations._set_product(
            o, Products.IFFT, T = o.Tsnap,
            N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_FFT_predict * o.Nfacet_predict**2,
            Rflop = 2.5 * o.Npix_linear_predict**2 * log(o.Npix_linear_predict**2, 2) / o.Tsnap,
            Rout = o.Mcpx * o.Npix_linear_predict * (o.Npix_linear_predict / 2 + 1) / o.Tsnap)

    @staticmethod
    def _apply_reprojection_equations(o):
        """ Re-Projection """

        o.Nf_proj_predict = o.Nf_FFT_predict
        o.Nf_proj_backward = o.Nf_FFT_backward
        if o.pipeline == Pipelines.DPrepA_Image:
            o.Nf_proj_predict = o.number_taylor_terms
            o.Nf_proj_backward = o.number_taylor_terms

         # (Consistent with PDR05 280115)
        if o.pipeline in Pipelines.imaging and o.pipeline != Pipelines.Fast_Img:
            # We do 2*o.Nmajortotal*(Tobs/Tsnap) entire image reprojections (i.e. both directions)
            Equations._set_product(
                o, Products.Reprojection,
                T = o.Tsnap,
                N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_proj_backward * o.Nfacet**2,
                Rflop = 50. * o.Npix_linear ** 2 / o.Tsnap,
                Rout = o.Mpx * o.Npix_linear**2 / o.Tsnap)
            Equations._set_product(
                o, Products.ReprojectionPredict,
                T = o.Tsnap,
                N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_proj_predict * o.Nfacet**2,
                Rflop = 50. * o.Npix_linear ** 2 / o.Tsnap,
                Rout = o.Mpx * o.Npix_linear**2 / o.Tsnap)

    @staticmethod
    def _apply_spectral_fitting_equations(o):

        if o.pipeline == Pipelines.DPrepA_Image:
            Equations._set_product(
                o, Products.Image_Spectral_Fitting,
                T = o.Tobs,
                N = o.Nmajortotal * o.Nbeam * o.Npp * o.number_taylor_terms,
                Rflop = 2.0 * (o.Nf_FFT_backward + o.Nf_FFT_predict) *
                        o.Npix_linear_total_fov ** 2 / o.Tobs,
                Rout = o.Mpx * o.Npix_linear_total_fov ** 2 / o.Tobs)

    @staticmethod
    def _apply_minor_cycle_equations(o):
        """ Minor Cycles """

        if not o.pipeline in Pipelines.imaging: return

        if o.pipeline in (Pipelines.ICAL, Pipelines.DPrepA, Pipelines.DPrepA_Image):
            # Search only on I_0
            Nf_identify = 1
        elif o.pipeline in (Pipelines.DPrepB, Pipelines.DPrepC):
            # Always search in all frequency space
            Nf_identify = o.Nf_out
        else:
            # No cleaning - e.g. fast imaging
            return

        # Create products
        Equations._set_product(o, Products.Identify_Component,
            T = o.Tobs,
            N = o.Nmajortotal * o.Nbeam,
            Rflop = 2 * o.Npp * o.Nminor * Nf_identify * (o.Npix_linear * o.Nfacet)**2 / o.Tobs,
            Rout = o.Mcpx / o.Tobs)

        # Subtract on all scales and only one frequency
        Equations._set_product(o, Products.Subtract_Image_Component,
            T = o.Tobs,
            N = o.Nmajortotal * o.Nbeam,
            Rflop = 2 * o.Npp * o.Nminor * o.Nscales * o.Npatch**2 / o.Tobs,
            Rout = o.Nscales * o.Npatch**2 / o.Tobs)

    @staticmethod
    def _apply_calibration_equations(o):

        # Number of flops needed per solution interval
        Flop_solver = 12 * o.Npp * o.Nsolve * o.Na**2
        # Number of flops required for averaging one vis. The steps are evaluate the complex phasor for 
        # the model phase (16 flops), multiply Vis by that phasor (16 flops) then average (8 flops)
        Flop_averager = 40 * o.Npp

        # ICAL solves for all terms but on different time scales. These should be set for context in the HPSOs.
        if o.pipeline == Pipelines.ICAL:
            N_Gslots = o.Tobs / o.tICAL_G
            N_Bslots = o.Tobs / o.tICAL_B
            N_Islots = o.Tobs / o.tICAL_I
            Flop_averaging = Flop_averager * o.Nvis * (o.Nf_max * o.tICAL_G + o.tICAL_B + o.Nf_max * o.tICAL_I * o.NIpatches)
            Flop_solving   = Flop_solver * (N_Gslots + o.NB_parameters * N_Bslots + o.NIpatches * N_Islots)
            Equations._set_product(o, Products.Solve,
                T = o.Tsnap,
                # We do one calibration to start with (using the original
                # LSM from the GSM and then we do Nselfcal more.
                N = (o.Nselfcal + 1) * o.Nbeam,
                Rflop = (Flop_solving + Flop_averaging) / o.Tobs,
                Rout = o.Mjones * o.Na * o.Nf_max)

        # RCAL solves for G only
        if o.pipeline == Pipelines.RCAL:
            N_Gslots = o.Tobs / o.tRCAL_G
            # Need to remember to average over all frequencies because a BP may have been applied.
            Flop_averaging = Flop_averager * o.Nvis * o.Nf_max * o.tRCAL_G
            Flop_solving   = Flop_solver * N_Gslots
            Equations._set_product(o, Products.Solve,
                T = o.Tsnap,
                # We need to complete one entire calculation within real time tCal_G
                N = o.Nbeam,
                Rflop = (Flop_solving + Flop_averaging) / o.Tobs,
                Rout = o.Mjones * o.Na * o.Nf_max / o.Tdump_ref)

    @staticmethod
    def _apply_dft_equations(o):
        if o.pipeline in Pipelines.imaging:
            # If the selfcal loop is embedded, we only need to do this
            # once but since we do an update of the model every
            # selfcal, we need to do it every selfcal.
            b = Symbol("b")
            Equations._set_product(o, Products.DFT,
                T = o.Tsnap,
                N = o.Nbeam * o.Nmajortotal * o.Nf_vis,
                Rflop = (64 * o.Na * o.Na * o.Nsource + 242 * o.Na * o.Nsource + 128 * o.Na * o.Na)
                        / o.Tdump_scaled,
                Rout = o.Mvis * o.Nvis / o.Nf_vis)

    @staticmethod
    def _apply_source_find_equations(o):
        """Rough estimate of source finding flops"""
        if o.pipeline == Pipelines.ICAL:
            # We need to fit 6 degrees of freedom to 100 points so we
            # have 600 FMults . Ignore for the moment Theta_beam the
            # solution of these normal equations. This really is a
            # stopgap. We need an estimate for a non-linear solver.
            Equations._set_product(
                o, Products.Source_Find,
                T = o.Tobs,
                N = o.Nmajortotal,
                Rflop = 6 * 100 * o.Nsource_find_iterations * o.Nsource / o.Tobs,
                Rout = 100 * o.Mcpx # guessed
            )

    @staticmethod
    def _apply_major_cycle_equations(o):

        # Note that we assume this is done for every Selfcal and Major Cycle
        if o.pipeline in Pipelines.imaging:
            Equations._set_product(o, Products.Subtract_Visibility,
                T = o.Tsnap,
                N = o.Nmajortotal * o.Npp * o.Nbeam * o.minimum_channels,
                Rflop = 8 * o.Nvis / o.minimum_channels,
                Rout = o.Mvis * o.Nvis / o.minimum_channels)

    @staticmethod
    def _apply_kernel_equations(o):
        """
        Generate Convolution kernels
        """

        b = Symbol('b')
        o.dfonF_backward = BLDep(b, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_backward(b))))
        o.dfonF_predict = BLDep(b, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_predict(b))))

        # allow uv positional errors up to o.epsilon_f_approx *
        # 1/Qkernel of a cell from frequency smearing.(But not more
        # than Nf_max channels...)
        o.Nf_gcf_backward_nosmear = BLDep(b,
            Min(log(o.wl_max / o.wl_min) /
                log(o.dfonF_backward(b) + 1.), o.Nf_max)) #TODO: PIP.IMG check please
        o.Nf_gcf_predict_nosmear  = BLDep(b,
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
            o.Nf_gcf_backward = BLDep(b, Max(o.Nf_gcf_backward_nosmear(b), o.minimum_channels))
            o.Nf_gcf_predict  = BLDep(b, Max(o.Nf_gcf_predict_nosmear(b),  o.minimum_channels))
            # TODO: some baseline dependent re-use limits along the
            # same lines as the frequency re-use? PIP.IMG check
            # please.
            o.Tkernel_backward = BLDep(b, o.Tion)
            o.Tkernel_predict  = BLDep(b, o.Tion)

        if o.pipeline in Pipelines.imaging:

            # The following two equations correspond to Eq. 35
            Equations._set_product(o, Products.Gridding_Kernel_Update,
                T = BLDep(b, o.Tkernel_backward(b)),
                N = BLDep(b, o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_gcf_backward(b)),
                Rflop = blsum(b, 5. * o.Nmm * o.Ncvff_backward(b)**2 * log(o.Ncvff_backward(b), 2) / o.Tkernel_backward(b)),
                Rout = blsum(b, 8 * o.Qgcf**3 * o.Ngw_backward(b)**3 / o.Tkernel_backward(b)))
            Equations._set_product(o, Products.Degridding_Kernel_Update,
                T = BLDep(b,o.Tkernel_predict(b)),
                N = BLDep(b,o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_gcf_predict(b)),
                Rflop = blsum(b, 5. * o.Nmm * o.Ncvff_predict(b)**2 * log(o.Ncvff_predict(b), 2) / o.Tkernel_predict(b)),
                Rout = blsum(b, 8 * o.Qgcf**3 * o.Ngw_predict(b)**3 / o.Tkernel_predict(b)))

    @staticmethod
    def _apply_phrot_equations(o):
        """Phase rotation (for the faceting)"""

        if not o.pipeline in Pipelines.imaging: return

        # Eq. 29. The sign() statement below serves as an "if > 1" statement for this symbolic equation.
        # 25 FLOPS per visiblity. Only do it if we need to facet.
        Nphrot = sign(o.Nfacet - 1)
        b = Symbol("b")

        # Predict phase rotation: Input from facets at predict
        # visibility rate, output at same rate.
        if o.scale_predict_by_facet:
            Equations._set_product(
                o, Products.PhaseRotationPredict,
                T = o.Tsnap,
                N = Nphrot * o.Nmajortotal * o.Npp * o.Nbeam * o.minimum_channels * o.Ntaylor_predict,
                Rflop = blsum(b, 25 * o.Nfacet**2 * o.Nvis / o.nbaselines / o.minimum_channels),
                Rout = blsum(b, o.Mvis * o.Nvis / o.nbaselines / o.minimum_channels))

        # Backward phase rotation: Input at overall visibility
        # rate, output averaged down to backward visibility rate.
        Equations._set_product(
            o, Products.PhaseRotation,
            T = o.Tsnap,
            N = Nphrot * o.Nmajortotal * o.Npp * o.Nbeam * o.Nfacet**2 * o.minimum_channels,
            Rflop = blsum(b, 25 * o.Nvis / o.nbaselines / o.minimum_channels),
            Rout = blsum(b, o.Mvis * o.Nvis_backward(b, 1) / o.minimum_channels))

    @staticmethod
    def _apply_flop_equations(o):
        """Calculate overall flop rate"""

        # revised Eq. 30
        o.Rflop = sum(o.get_products('Rflop').values())

        # Calculate interfacet IO rate for faceting: TCC-SDP-151123-1-1 rev 1.1
        o.Rinterfacet = 2 * o.Nmajortotal * Min(3.0, 2.0 + 18.0 * o.facet_overlap_frac) * (o.Nfacet * o.Npix_linear)**2 * o.Nf_out * 4  / o.Tobs


    @staticmethod
    def _apply_io_equations(o):
        """
        Compute the Buffer sizes

        References: Section 12.15 in PDR05
        """

        # Note: this is the size of the raw binary data that needs to
        # be stored in the visibility buffer, and does not include
        # inevitable overheads. However, this size does include a
        # factor for double-buffering

        b = Symbol('b')
        o.Mw_cache = \
            o.Ncbytes * o.Nbeam * (o.Qgcf ** 3) * \
            Equations._sum_bl_bins(o,
                blsum(b, o.Nf_vis_predict(b) * o.Ngw_predict(b) ** 3))
            # Eq 48. TODO: re-implement this equation within a better description of where kernels will be stored etc.
        # (allowing storage of a full observation while simultaneously capturing the next)
        # TODO: The o.Nbeam factor in eqn below is not mentioned in PDR05 eq 49. Why? It is in eqn.2 though.
        o.Mbuf_vis = o.buffer_factor * o.Npp * o.Nvis * o.Nbeam * o.Mvis * o.Tobs  # Eq 49

        # added o.Nfacet dependence; changed Nmajor factor to Nmajor+1 as part of post PDR fixes.
        # TODO: Differs quite substantially from Eq 50, by merit of the Nbeam and Npp, as well as Nfacet ** 2 factors.
        # TODO: PDR05 lacking in this regard and must be updated.
        # This is correct if we have only got facets for the backward step and use Nfacet=1 for predict step: TJC see TCC-SDP-151123-1-1
        # It probably can go much smaller, though: see SDPPROJECT-133
#        o.Rio = o.Nbeam * o.Npp * (1 + o.Nmajortotal) * o.Nvis * o.Mvis * o.Nfacet ** 2  # Eq 50
        o.Rio = 2.0 * o.Nbeam * o.Npp * (1 + o.Nmajortotal) * o.Nvis * o.Mvis  # Eq 50

    @staticmethod
    def _sum_bl_bins(o, bldep):
        """Helper for dealing with baseline dependence. For a term
        depending on the given symbols, "sum_bl_bins" will build a
        sum term over all baseline bins."""

        # Actually baseline-dependent?
        if isinstance(bldep, BLDep):
            b = bldep.b
            bcount = bldep.bcount
            expr = bldep.term
        else:
            return o.nbaselines * bldep

        # Small bit of ad-hoc formula optimisation: Exploit
        # independent factors. Makes for smaller terms, which is good
        # both for Sympy as well as for output.
        if not b in expr.free_symbols:
            return o.nbaselines * expr.subs(bcount, 1)
        if isinstance(expr, Mul):
            def indep(e): return not (b in e.free_symbols or bcount in e.free_symbols)
            indepFactors = list(filter(indep, expr.as_ordered_factors()))
            if len(indepFactors) > 0:
                def not_indep(e): return not indep(e)
                restFactors = filter(not_indep, expr.as_ordered_factors())
                return Mul(*indepFactors) * Equations._sum_bl_bins(o, BLDep((b, bcount), Mul(*restFactors)))

        # Replace in concrete values for baseline fractions and
        # length. Using Lambda here is a solid 25% faster than
        # subs(). Unfortunately very slow nonetheless...
        results = []

        # Symbolic? Generate actual symbolic sum expression
        if isinstance(o.Bmax_bins, Symbol):
            return Sum(bldep(o.Bmax_bins(b), 1), (b, 1, o.nbaselines))

        # Otherwise generate sum term manually that approximates the
        # full sum using baseline bins
        for (frac_val, bmax_val) in zip(o.frac_bins, o.Bmax_bins):
            results.append(bldep(bmax_val, frac_val*o.nbaselines_full))
        return Add(*results, evaluate=False)

    @staticmethod
    def _set_product(o, product, T=None, N=1, **args):
        """Sets product properties using a task abstraction. Each property is
        expressed as a sum over baselines.

        @param product: Product to set.
        @param T: Observation time covered by this task. Default is the
          entire observation (Tobs). Can be baseline-dependent.
        @param N: Task parallelism / rate multiplier. The number of
           tasks that work on the data in parallel. Can be
           baseline-dependent.
        @param args: Task properties as rates. Will be multiplied by
           N.  If it is baseline-dependent, it will be summed over all
           baselines to yield the final rate.
        """

        # Collect properties
        if T is None: T = o.Tobs
        props = { "N": N, "T": T }
        for k, expr in args.items():

            # Multiply out multiplicator. If either of them is
            # baseline-dependent, this will generate a new
            # baseline-dependent term (see BLDep)
            total = N * expr

            # Baseline-dependent? Generate a sum term, otherwise just say as-is
            if isinstance(total, BLDep):
                props[k] = Equations._sum_bl_bins(o, total)
                props[k+"_task"] = expr
            else:
                props[k] = total

        # Set product
        o.set_product(product, **props)

