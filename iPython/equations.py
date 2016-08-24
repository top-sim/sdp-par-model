"""
This class contains the actual equations that are used to compute the telescopes' performance values and computional
requirements from the supplied basic parameters defined in ParameterDefinitions.

This class contains a method for (symbolically) computing derived parameters using imaging equations described
in PDR05 (version 1.85).
"""

from __future__ import print_function

from sympy import log, Min, Max, sqrt, floor, sign, ceiling, Symbol
from numpy import pi, round
import math
from parameter_definitions import Pipelines, Products
from parameter_definitions import ParameterContainer, BLDep, blsum
import warnings

# Check sympy compatibility
import sympy
if sympy.__version__ == "1.0":
    raise Exception("SymPy version 1.0 is broken. Please either upgrade or downgrade your version!")

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
        if symbolify == 'helper':
            o.symbolify()
            o.Bmax_bins = Symbol("B_max")
        Equations._apply_channel_equations(o, symbolify)
        Equations._apply_coalesce_equations(o, symbolify)
        Equations._apply_geometry_equations(o, symbolify)

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
        o.Nbl_full = o.Na * (o.Na - 1) / 2.0
        o.Nbl = sum(o.frac_bins) * o.Nbl_full

        # Wavelengths
        o.wl_max = o.c / o.freq_min  # Maximum Wavelength
        o.wl_min = o.c / o.freq_max  # Minimum Wavelength
        o.wl = 0.5 * (o.wl_max + o.wl_min)  # Representative Wavelength

        # Calculate number of subbands
        o.Nsubbands = ceiling(log(o.wl_max/o.wl_min)/log(o.max_subband_freq_ratio))
        o.Qsubband = (o.wl_max/o.wl_min)**(1./o.Nsubbands)


    @staticmethod
    def _apply_image_equations(o):
        """
        Calculate image parameters, such as resolution and size

        References:
         * SKA-TEL-SDP-0000040 01D section D - The Resolution and Extent of Images and uv Planes
         * SKA-TEL-SDP-0000040 01D section H.2 - Faceting
        """

        # max subband wavelength to set image FoV
        o.wl_sb_max = o.wl *sqrt(o.Qsubband)
        # min subband wavelength to set pixel size
        o.wl_sb_min = o.wl_sb_max / o.Qsubband

        # Facet overlap is only needed if Nfacet > 1
        o.r_facet = sign(o.Nfacet - 1)*o.r_facet_base

        # Facet Field-of-view (linear) at max sub-band wavelength
        o.Theta_fov = 7.66 * o.wl_sb_max * o.Qfov * (1+o.r_facet) \
                      / (pi * o.Ds * o.Nfacet)
        # Total linear field of view of map (all facets)
        o.Theta_fov_total = 7.66 * o.wl_sb_max * o.Qfov / (pi * o.Ds)
        # Synthesized beam at fiducial wavelength. Called Theta_PSF in PDR05.
        o.Theta_beam = 3 * o.wl_sb_min / (2. * o.Bmax)
        # Pixel size at fiducial wavelength.
        o.Theta_pix = o.Theta_beam / (2. * o.Qpix)

        # Number of pixels on side of facet in subband.
        o.Npix_linear = (o.Theta_fov / o.Theta_pix)
        o.Npix_linear_fov_total = (o.Theta_fov_total / o.Theta_pix)
        # Predict fov and number of pixels depends on whether we facet
        if o.scale_predict_by_facet:
            o.Theta_fov_predict = o.Theta_fov
            o.Nfacet_predict = o.Nfacet
            o.Npix_linear_predict = o.Npix_linear
        else:
            o.Theta_fov_predict = o.Theta_fov_total
            o.Nfacet_predict = 1
            o.Npix_linear_predict = o.Npix_linear_fov_total

        # expansion of sine solves eps = arcsinc(1/amp_f_max).
        o.epsilon_f_approx = sqrt(6 * (1 - (1. / o.amp_f_max)))
        #Correlator dump rate set by smearing limit at field of view needed
        #for ICAL pipeline (Assuming this is always most challenging)
        o.Tdump_no_smear=o.epsilon_f_approx * o.wl \
                    / (o.Omega_E * o.Bmax * 7.66 * o.wl_sb_max * o.Qfov_ICAL / (pi * o.Ds))
        o.Tint_used = Max(o.Tint_min, o.Tdump_no_smear)
        

    @staticmethod
    def _apply_channel_equations(o, symbolify):
        """
        Determines the number of frequency channels to use in backward &
        predict steps.

        References:
         * SKA-TEL-SDP-0000040 01D section B - Covering the Frequency Axis
         * SKA-TEL-SDP-0000040 01D section D - Visibility Averaging and Coalescing
        """

        b = Symbol('b')
        log_wl_ratio = log(o.wl_max / o.wl_min)
  
        # See notes on https://confluence.ska-sdp.org/display/PIP/Frequency+resolution+and+smearing+effects+in+the+iPython+SDP+Parametric+model
        o.Qbw = 1.47 / o.epsilon_f_approx

        if symbolify == 'helper':
            o.epsilon_f_approx = Symbol(o.make_symbol_name('epsilon_f_approx'))
            o.Qbw = Symbol(o.make_symbol_name('Qbw'))

        # Frequency number to avoid smearing for full and facet FOV at
        # max baseline respectively.  These limit bandwidth smearing
        # to within a fraction (epsilon_f_approx) of a uv cell.
        o.Nf_no_smear = \
            log_wl_ratio / log(1 + (3 * o.wl / (2. * o.Bmax * o.Theta_fov_total * o.Qbw)))
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
    def _apply_coalesce_equations(o, symbolify):
        """
        Determines amount of coalescing of visibilities in time.

        References:
         * SKA-TEL-SDP-0000040 01D section A - Covering the Time Axis
         * SKA-TEL-SDP-0000040 01D section D - Visibility Averaging and Coalescing
        """

        # correlator output averaging time scaled for max baseline.
        b = Symbol('b')
        o.combine_time_samples = BLDep(b,
            Max(floor(o.epsilon_f_approx * o.wl /
                      (o.Theta_fov_total * o.Omega_E * b * o.Tint_used)), 1.))

        # coalesce visibilities in time.
        o.Tcoal_skipper = BLDep(b,
            o.Tint_used * o.combine_time_samples(b))
        if symbolify == 'helper':
            o.Tcoal_skipper = Symbol(o.make_symbol_name("Tcoal_skipper"))

        if o.blcoal:
            # For backward step at gridding only, allow coalescance of
            # visibility points at Facet FoV smearing limit only for
            # bl-dep averaging case.
            o.Tcoal_backward = BLDep(b,
                Min(o.Tcoal_skipper(b) * o.Nfacet/(1+o.r_facet), o.Tion))
            if o.scale_predict_by_facet:
                o.Tcoal_predict = BLDep(b, o.Tcoal_backward(b))
            else:
                # Don't let any bl-dependent time averaging be for
                # longer than either 1.2s or Tion. ?Why 1.2s?
                o.Tcoal_predict = BLDep(b, Min(o.Tcoal_skipper(b), 1.2, o.Tion))
        else:
            o.Tcoal_predict = BLDep(b, o.Tint_used)
            o.Tcoal_backward = BLDep(b, o.Tint_used)

        # Visibility rate (visibilities per second) on ingest, including autocorrelations (per beam, per polarisation)
        o.Rvis_ingest = (o.Nbl + o.Na) * o.Nf_max / o.Tint_used


        # Total visibility rate (visibilities per second per beam, per polarisation)
        #after frequency channels have been combined where possible.
        #Autocorrelations included here too
        o.Rvis = (o.Nbl + o.Na) * o.Nf_vis / o.Tint_used

        # Visibility rate for backward step, allow coalescing
        # in time and freq prior to gridding
        o.Rvis_backward = blsum(b, o.Nf_vis_backward(b) / o.Tcoal_backward(b))
        # Visibility rate for predict step
        o.Rvis_predict = blsum(b, o.Nf_vis_predict(b) / o.Tcoal_predict(b))

    @staticmethod
    def _apply_geometry_equations(o, symbolify):
        """
        Telescope geometry in space and time, given curvature and rotation
        of the earth. This determines the maximum w-term that needs to
        be corrected for and hence the size of w-kernels.

        References:
          * SKA-TEL-SDP-0000040 01D section G - Convolution Kernel Sizes
          * SKA-TEL-SDP-0000040 01D section H.1 - Imaging Pipeline Geometry Assumptions
        """

        b = Symbol('b')

        # Contribution of earth curvature to w-term
        o.DeltaW_Earth = BLDep(b, b ** 2 / (8. * o.R_Earth * o.wl))
        # Contribution of earth movement to w-term
        o.DeltaW_SShot = BLDep(b, b * o.Omega_E * o.Tsnap / (2. * o.wl))
        o.DeltaW_max = BLDep(b, o.Qw * Max(o.DeltaW_SShot(b), o.DeltaW_Earth(b)))
        if symbolify == 'helper':
            o.DeltaW_Earth = Symbol(o.make_symbol_name('DeltaW_Earth'))
            o.DeltaW_SShot = Symbol(o.make_symbol_name('DeltaW_SShot'))
            o.DeltaW_max = Symbol(o.make_symbol_name('DeltaW_max'))

        # Eq. 25, w-kernel support size. Note possible difference in
        # cellsize assumption!
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

        # Combined kernel support size and oversampling
        if o.on_the_fly:
            o.Qgcf = 1.0
        o.Ncvff_backward = BLDep(b, sqrt(o.Nkernel2_backward(b))*o.Qgcf)
        o.Ncvff_predict = BLDep(b, sqrt(o.Nkernel2_predict(b))*o.Qgcf)

    @staticmethod
    def _apply_ingest_equations(o):
        """
        Ingest equations

        References: SKA-TEL-SDP-0000040 01D section 3.3 - The Fast and Buffered pre-processing pipelines1
        """

        if o.pipeline == Pipelines.Ingest:
            o.set_product(Products.Receive,
                T = o.Tsnap, N = o.Nbeam * o.Nf_min,
                Rflop = 2 * o.Npp * o.Rvis_ingest / o.Nf_min +
                        1000 * o.Na / o.Tint_used,
                Rout = o.Mvis * o.Npp * o.Rvis_ingest)
            o.set_product(Products.Flag,
                T = o.Tsnap, N = o.Nbeam * o.Nf_min,
                Rflop = 279 * o.Npp * o.Rvis_ingest / o.Nf_min,
                Rout = o.Mvis * o.Npp * o.Rvis_ingest / o.Nf_min)
            # Ndemix is the number of time-frequency products used
            # (typically 1000) so we have to divide out the number of
            # input channels
            o.set_product(Products.Demix,
                T = o.Tsnap, N = o.Nbeam * o.Nf_min,
                Rflop = 8 * o.Npp * (o.Rvis_ingest * o.Ndemix / o.Nf_max) * (o.NA * (o.NA + 1) / 2.0)
                        / o.Nf_min,
                Rout = o.Mvis * o.Npp * o.Rvis / o.Nf_min)
            o.set_product(Products.Average,
                T = o.Tsnap, N = o.Nbeam * o.Nf_min,
                Rflop = 8 * o.Npp * o.Rvis_ingest / o.Nf_min,
                Rout = o.Mvis * o.Npp * o.Rvis / o.Nf_min)

    @staticmethod
    def _apply_flag_equations(o):
        """ Flagging equations for non-ingest pipelines"""

        if not (o.pipeline == Pipelines.Ingest):
            o.set_product(Products.Flag,
                T=o.Tsnap, N=o.Nbeam * o.Nmajortotal * o.Nf_min_gran,
                Rflop=279 * o.Npp * o.Rvis / o.Nf_min_gran,
                Rout = o.Mvis * o.Npp * o.Rvis / o.Nf_min_gran)

    @staticmethod
    def _apply_correct_equations(o):
        """
        Correction of gains

        References: SKA-TEL-SDP-0000040 01D section 3.6.7 - Correct
        """

        if not o.pipeline == Pipelines.Ingest:
            o.set_product(Products.Correct,
                T = o.Tsnap, N = o.Nbeam*o.Nmajortotal * o.Npp * o.Nf_min_gran,
                Rflop = 8 * o.Nmm * o.Rvis * o.NIpatches / o.Nf_min_gran,
                Rout = o.Mvis * o.Rvis / o.Nf_min_gran)

    @staticmethod
    def _apply_grid_equations(o):
        """
        Gridding and degridding of visibilities

        References: SKA-TEL-SDP-0000040 01D section 3.6.11 - Grid and Degrid
        """

        # For the ASKAP MSMFS, we grid all data for each taylor term
        # with polynominal of delta freq/freq
        b = Symbol('b')
        o.Ntt_backward = 1
        o.Ntt_predict = 1
        if o.pipeline == Pipelines.DPrepA:
            o.Ntt_backward = o.Ntt
            o.Ntt_predict = o.Ntt

        if not o.pipeline in Pipelines.imaging: return
        o.set_product(Products.Grid, T=o.Tsnap,
            N = BLDep(b, o.Nmajortotal * o.Nbeam * o.Npp * o.Ntt_backward *
                         o.Nfacet**2 * o.Nf_vis_backward(b)),
            Rflop = blsum(b, 8 * o.Nmm * o.Nkernel2_backward(b) / o.Tcoal_backward(b)),
            Rout = o.Mcpx * o.Npix_linear * (o.Npix_linear / 2 + 1) / o.Tsnap)
        o.set_product(Products.Degrid, T = o.Tsnap,
            N = BLDep(b, o.Nmajortotal * o.Nbeam * o.Npp * o.Ntt_predict *
                         o.Nfacet_predict**2 * o.Nf_vis_predict(b)),
            Rflop = blsum(b, 8 * o.Nmm * o.Nkernel2_predict(b) / o.Tcoal_predict(b)),
            Rout = blsum(b, o.Mvis / o.Tcoal_predict(b)))

    @staticmethod
    def _apply_fft_equations(o):
        """
        Discrete fourier transformation of grids to images (and back)

        References: SKA-TEL-SDP-0000040 01D section 3.6.13 - FFT and iFFT
        """

        if not o.pipeline in Pipelines.imaging: return

        # These are real-to-complex for which the prefactor in the FFT is 2.5
        o.set_product(Products.FFT, T = o.Tsnap,
            N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_FFT_backward * o.Nfacet**2,
            Rflop = 2.5 * o.Npix_linear ** 2 * log(o.Npix_linear**2, 2) / o.Tsnap,
            Rout = o.Mpx * o.Npix_linear**2 / o.Tsnap)
        o.set_product(Products.IFFT, T = o.Tsnap,
            N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_FFT_predict * o.Nfacet_predict**2,
            Rflop = 2.5 * o.Npix_linear_predict**2 * log(o.Npix_linear_predict**2, 2) / o.Tsnap,
            Rout = o.Mcpx * o.Npix_linear_predict * (o.Npix_linear_predict / 2 + 1) / o.Tsnap)

    @staticmethod
    def _apply_reprojection_equations(o):
        """
        Re-projection of skewed images as generated by w snapshots

        References: SKA-TEL-SDP-0000040 01D section 3.6.14 - Reprojection
        """

        o.Nf_proj_predict = o.Nf_FFT_predict
        o.Nf_proj_backward = o.Nf_FFT_backward
        if o.pipeline == Pipelines.DPrepA_Image:
            o.Nf_proj_predict = o.Ntt
            o.Nf_proj_backward = o.Ntt

        if o.pipeline in Pipelines.imaging and o.pipeline != Pipelines.Fast_Img:
            o.set_product(Products.Reprojection,
                T = o.Tsnap,
                N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_proj_backward * o.Nfacet**2,
                Rflop = 50. * o.Npix_linear ** 2 / o.Tsnap,
                Rout = o.Mpx * o.Npix_linear**2 / o.Tsnap)
            o.set_product(Products.ReprojectionPredict,
                T = o.Tsnap,
                N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_proj_predict * o.Nfacet**2,
                Rflop = 50. * o.Npix_linear ** 2 / o.Tsnap,
                Rout = o.Mpx * o.Npix_linear**2 / o.Tsnap)

    @staticmethod
    def _apply_spectral_fitting_equations(o):
        """
        Spectral fitting of the image for CASA-style MSMFS clean.

        References: SKA-TEL-SDP-0000040 01D section 3.6.15 - Image Spectral Fitting
        """

        if o.pipeline == Pipelines.DPrepA_Image:
            o.set_product(Products.Image_Spectral_Fitting,
                T = o.Tobs,
                N = o.Nmajortotal * o.Nbeam * o.Npp * o.Ntt,
                Rflop = 2.0 * (o.Nf_FFT_backward + o.Nf_FFT_predict) *
                        o.Npix_linear_fov_total ** 2 / o.Tobs,
                Rout = o.Mpx * o.Npix_linear_fov_total ** 2 / o.Tobs)

    @staticmethod
    def _apply_minor_cycle_equations(o):
        """
        Minor Cycles implementing deconvolution / cleaning

        References: SKA-TEL-SDP-0000040 01D section 3.6.16 - Subtract Image Component
        """

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
        o.set_product(Products.Identify_Component,
            T = o.Tobs,
            N = o.Nmajortotal * o.Nbeam,
            Rflop = 2 * o.Npp * o.Nminor * Nf_identify * (o.Npix_linear * o.Nfacet)**2 / o.Tobs,
            Rout = o.Mcpx / o.Tobs)

        # Subtract on all scales and only one frequency
        o.set_product(Products.Subtract_Image_Component,
            T = o.Tobs,
            N = o.Nmajortotal * o.Nbeam,
            Rflop = 2 * o.Npp * o.Nminor * o.Nscales * o.Npatch**2 / o.Tobs,
            Rout = o.Nscales * o.Npatch**2 / o.Tobs)

    @staticmethod
    def _apply_calibration_equations(o):
        """
        Self-calibration using predicted visibilities

        References: SKA-TEL-SDP-0000040 01D section 3.6.5 - Solve
        """

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
            Flop_averaging = Flop_averager * o.Rvis * (o.Nf_max * o.tICAL_G + o.tICAL_B + o.Nf_max * o.tICAL_I * o.NIpatches)
            Flop_solving   = Flop_solver * (N_Gslots + o.NB_parameters * N_Bslots + o.NIpatches * N_Islots)
            o.set_product(Products.Solve,
                T = o.tICAL_G,
                # We do one calibration to start with (using the original
                # LSM from the GSM and then we do Nselfcal more.
                N = (o.Nselfcal + 1) * o.Nbeam,
                Rflop = (Flop_solving + Flop_averaging) / o.Tobs,
                Rout = o.Mjones * o.Na * o.Nf_max)

        # RCAL solves for G only
        if o.pipeline == Pipelines.RCAL:
            N_Gslots = o.Tobs / o.tRCAL_G
            # Need to remember to average over all frequencies because a BP may have been applied.
            Flop_averaging = Flop_averager * o.Rvis * o.Nf_max * o.tRCAL_G
            Flop_solving   = Flop_solver * N_Gslots
            o.set_product(Products.Solve,
                T = o.tRCAL_G,
                # We need to complete one entire calculation within real time tCal_G
                N = o.Nbeam,
                Rflop = (Flop_solving + Flop_averaging) / o.Tobs,
                Rout = o.Mjones * o.Na * o.Nf_max / o.Tint_used)

    @staticmethod
    def _apply_dft_equations(o):
        """
        Direct discrete fourier transform as predict alternative to
        Reproject+FFT+Degrid+Phase Rotation.

        References: SKA-TEL-SDP-0000040 01D section 3.6.4 - Predict via Direct Fourier Transform
        """

        if o.pipeline in Pipelines.imaging:
            # If the selfcal loop is embedded, we only need to do this
            # once but since we do an update of the model every
            # selfcal, we need to do it every selfcal.
            b = Symbol("b")
            o.set_product(Products.DFT,
                T = o.Tsnap,
                N = o.Nbeam * o.Nmajortotal * o.Nf_min_gran,
                Rflop = (64 * o.Na * o.Na * o.Nsource + 242 * o.Na * o.Nsource + 128 * o.Na * o.Na)
                        * o.Nf_vis / o.Nf_min_gran / o.Tint_used,
                Rout = o.Npp * o.Mvis * o.Rvis / o.Nf_min_gran)

    @staticmethod
    def _apply_source_find_equations(o):
        """
        Rough estimate of source finding flops.

        References: SKA-TEL-SDP-0000040 01D section 3.6.17 - Source find
        """
        if o.pipeline == Pipelines.ICAL:
            # We need to fit 6 degrees of freedom to 100 points so we
            # have 600 FMults . Ignore for the moment Theta_beam the
            # solution of these normal equations. This really is a
            # stopgap. We need an estimate for a non-linear solver.
            o.set_product(Products.Source_Find,
                T = o.Tobs,
                N = o.Nmajortotal,
                Rflop = 6 * 100 * o.Nsource_find_iterations * o.Nsource / o.Tobs,
                Rout = 100 * o.Mcpx # guessed
            )

    @staticmethod
    def _apply_major_cycle_equations(o):
        """
        Subtract predicted visibilities from last major cycle

        References: SKA-TEL-SDP-0000040 01D section 3.6.6 - Subtract
        """

        # Note that we assume this is done for every Selfcal and Major Cycle
        if o.pipeline in Pipelines.imaging:
            o.set_product(Products.Subtract_Visibility,
                T = o.Tsnap,
                N = o.Nmajortotal * o.Nbeam * o.Nf_min_gran,
                Rflop = 8 * o.Npp * o.Rvis / o.Nf_min_gran,
                Rout = o.Mvis * o.Npp * o.Rvis / o.Nf_min_gran)

    @staticmethod
    def _apply_kernel_equations(o):
        """
        Generate Convolution kernels

        References:
         * SKA-TEL-SDP-0000040 01D section 3.6.12 - Gridding Kernel Update
         * SKA-TEL-SDP-0000040 01D section E - Re-use of Convolution Kernels
        """

        b = Symbol('b')
        o.dfonF_backward = BLDep(b, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_backward(b))))
        o.dfonF_predict = BLDep(b, o.epsilon_f_approx / (o.Qkernel * sqrt(o.Nkernel2_predict(b))))

        # allow uv positional errors up to o.epsilon_f_approx *
        # 1/Qkernel of a cell from frequency smearing.(But not more
        # than Nf_max channels...)
        o.Nf_gcf_backward_nosmear = BLDep(b,
            Min(log(o.wl_max / o.wl_min) /
                log(o.dfonF_backward(b) + 1.), o.Nf_max))
        o.Nf_gcf_predict_nosmear  = BLDep(b,
            Min(log(o.wl_max / o.wl_min) /
                log(o.dfonF_predict(b) + 1.), o.Nf_max))

        if o.on_the_fly:
            o.Nf_gcf_backward = o.Nf_vis_backward
            o.Nf_gcf_predict  = o.Nf_vis_predict
            o.Tkernel_backward = o.Tcoal_backward
            o.Tkernel_predict  = o.Tcoal_predict
        else:
            # For both of the following, maintain distributability;
            # need at least Nf_min kernels.
            o.Nf_gcf_backward = BLDep(b, Max(o.Nf_gcf_backward_nosmear(b), o.Nf_min))
            o.Nf_gcf_predict  = BLDep(b, Max(o.Nf_gcf_predict_nosmear(b),  o.Nf_min))
            o.Tkernel_backward = BLDep(b, o.Tion)
            o.Tkernel_predict  = BLDep(b, o.Tion)

        if o.pipeline in Pipelines.imaging:

            # The following two equations correspond to Eq. 35
            o.set_product(Products.Gridding_Kernel_Update,
                T = BLDep(b, o.Tkernel_backward(b)),
                N = BLDep(b, o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_gcf_backward(b)),
                Rflop = blsum(b, 5. * o.Nmm * o.Ncvff_backward(b)**2 * log(o.Ncvff_backward(b), 2) / o.Tkernel_backward(b)),
                Rout = blsum(b, 8 * o.Qgcf**3 * o.Ngw_backward(b)**3 / o.Tkernel_backward(b)))
            o.set_product(Products.Degridding_Kernel_Update,
                T = BLDep(b,o.Tkernel_predict(b)),
                N = BLDep(b,o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_gcf_predict(b)),
                Rflop = blsum(b, 5. * o.Nmm * o.Ncvff_predict(b)**2 * log(o.Ncvff_predict(b), 2) / o.Tkernel_predict(b)),
                Rout = blsum(b, 8 * o.Qgcf**3 * o.Ngw_predict(b)**3 / o.Tkernel_predict(b)))

    @staticmethod
    def _apply_phrot_equations(o):
        """
        Phase rotation (for the faceting)

        References: SKA-TEL-SDP-0000040 01D section 3.6.9 - Phase Rotation
        """

        if not o.pipeline in Pipelines.imaging: return

        b = Symbol("b")

        # 25 FLOPS per visiblity. Only do it if we need to facet.

        # Predict phase rotation: Input from facets at predict
        # visibility rate, output at same rate.
        if o.scale_predict_by_facet:
            o.set_product(Products.PhaseRotationPredict,
                T = o.Tsnap,
                N = sign(o.Nfacet - 1) * o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_min_gran *
                    o.Ntt_predict * o.Nfacet**2 ,
                Rflop = blsum(b, 25 * o.Nf_vis / o.Nf_min_gran / o.Tint_used),
                Rout = blsum(b, o.Mvis * o.Nf_vis / o.Nf_min_gran / o.Tint_used))

        # Backward phase rotation: Input at overall visibility
        # rate, output averaged down to backward visibility rate.
        o.set_product(Products.PhaseRotation,
            T = o.Tsnap,
            N = sign(o.Nfacet - 1) * o.Nmajortotal * o.Npp * o.Nbeam * o.Nfacet**2 * o.Nf_min_gran,
            Rflop = blsum(b, 25 * o.Rvis / o.Nbl / o.Nf_min_gran),
            Rout = blsum(b, o.Mvis * o.Rvis_backward(b, 1) / o.Nf_min_gran))

    @staticmethod
    def _apply_flop_equations(o):
        """Calculate overall flop rate"""

        # Sum up products
        o.Rflop = sum(o.get_products('Rflop').values())

        # Calculate interfacet IO rate for faceting: TCC-SDP-151123-1-1 rev 1.1
        o.Rinterfacet = \
            2 * o.Nmajortotal * Min(3.0, 2.0 + 18.0 * o.r_facet_base) * \
            (o.Nfacet * o.Npix_linear)**2 * o.Nf_out * 4  / o.Tobs

    @staticmethod
    def _apply_io_equations(o):
        """
        Compute the Buffer sizes

        References: SKA-TEL-SDP-0000040 01D section H.3 - Convolution Kernel Cache Size
        """

        b = Symbol('b')

        # Visibility buffer size
        #
        # This is the size of the raw binary data that needs to
        # be stored in the visibility buffer, and does not include
        # inevitable overheads. However, this size does include a
        # factor for double-buffering
        #
        # TODO: The o.Nbeam factor in eqn below is not mentioned in PDR05 eq 49. Why? It is in eqn.2 though.
        o.Mbuf_vis = o.buffer_factor * o.Npp * o.Rvis * o.Nbeam * o.Mvis * o.Tobs

        # Visibility read rate
        #
        # According to SDPPROJECT-133 (JIRA) assume that we only need
        # to read all visibilities twice per major cycle and beam.
        o.Rio = 2.0 * o.Nbeam * o.Npp * (1 + o.Nmajortotal) * o.Rvis * o.Mvis
        # o.Rio = o.Nbeam * o.Npp * (1 + o.Nmajortotal) * o.Rvis * o.Mvis * o.Nfacet ** 2

        # Convolution kernel cache size
        #
        # TODO: re-implement this equation within a better description
        # of where kernels will be stored etc.  (allowing storage of a
        # full observation while simultaneously capturing the next)
        o.Mw_cache = (o.Ngw_predict(o.Bmax) ** 3) * (o.Qgcf ** 3) * o.Ncbytes * o.Nbeam * o.Nf_vis_predict(o.Bmax)
