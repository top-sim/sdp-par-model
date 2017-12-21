"""
This module contains the actual equations that are used to compute
the telescopes' performance values and computional requirements from
the supplied basic parameters defined in ParameterDefinitions.
"""

from __future__ import print_function  # Makes Python-3 style print() function available in Python 2.x

import math
import warnings

from numpy import pi, round
import sympy
from sympy import log, Min, Max, sqrt, floor, sign, ceiling, Symbol, sin

from .definitions import Pipelines, Products
from .container import ParameterContainer, BLDep, blsum

# Check sympy compatibility
if sympy.__version__ == "1.0":

    # There seems to be a bug that leads sympy 1.0.0 into an infinite
    # loop when using non-natural logarithms. Thanks to Juande
    # Santander-Vela for discovering the workaround.
    def log(v, b=None):
        if b is None:
            return sympy.log(v)
        else:
            return sympy.log(v) / math.log(b)

def apply_imaging_equations(telescope_parameters, pipeline, bins, binfracs, verbose, symbolify=''):
    """
    (Symbolically) computes a set of derived parameters using imaging
    equations described in PDR05 (version 1.85).

    The derived parameters are added to the supplied
    `telescope_parameters` object (locally referred to as `o`).  Where
    parameters are only described symbolically (using sympy) they
    can be numerically evaluated at a later stage, when unknown
    symbolic variables are suitably substituted.

    :param telescope_parameters: :class:`.container.ParameterContainer` object
        containing the telescope parameters. Will be modified in-place by
        appending / overwriting the relevant fields
    :param pipeline: The pipeline
    :param verbose: displays verbose command-line output
    """

    o = telescope_parameters  # Used for shorthand in the equations below
    assert isinstance(o, ParameterContainer)
    assert hasattr(o, "c")  # Checks initialization by proxy of whether the speed of light is defined

    # Store parameters
    o.set_param('pipeline', pipeline)  # e.g. ICAL, DPprepA

    # Check parameters
    if hasattr(o, 'Tobs') and (o.Tobs < 10.0):
        o.Tsnap_min = o.Tobs
        if verbose:
            print('Warning: Tsnap_min overwritten in equations.py file because observation was shorter than 10s')

    # Load common equations.
    _apply_common_equations(o, bins, binfracs)

    # If requested, we replace all parameters so far with symbols,
    # so product equations are purely symbolic.
    if symbolify == 'all':
        o.symbolify()

    # This set of methods must be executed in the defined sequence since
    # some values in sequence. This is ugly and we should fix it one day.
    # Apply general imaging equations. These next 4 methods just set up
    # parameters.
    _apply_image_equations(o)
    if symbolify == 'helper':
        o.symbolify()
    _apply_channel_equations(o, symbolify)
    _apply_coalesce_equations(o, symbolify)
    _apply_geometry_equations(o, symbolify)
    _apply_kernel_equations(o)

    # If requested, we replace all parameters so far with symbols,
    # so product equations are purely symbolic.
    if symbolify == 'product':
        o.symbolify()

    # Apply product equations to fill in the Rflop estimates (and others when they arrive).
    _apply_ingest_equations(o)
    _apply_dft_equations(o)
    _apply_flag_equations(o)
    _apply_correct_equations(o)
    _apply_calibration_equations(o)
    _apply_major_cycle_equations(o)
    _apply_grid_equations(o)
    _apply_fft_equations(o)
    _apply_reprojection_equations(o)
    _apply_spectral_fitting_equations(o)
    _apply_source_find_equations(o)
    _apply_kernel_product_equations(o)
    _apply_phrot_equations(o)
    _apply_minor_cycle_equations(o)

    # Apply summary equations
    _apply_io_equations(o)
    _apply_flop_equations(o)

    return o


def _apply_common_equations(o, bins, binfracs):
    """
    Calculate simple derived values that are going to get used fairly often.
    """

    # Derive simple parameters. Note that we ignore some baselines
    # when Bmax is lower than the telescope's maximum.
    #o.Na = 512 * (35.0/o.Ds)**2 #Hack to make station diameter and station number inter-related...comment it out after use
    o.Nbl_full = o.Na * (o.Na - 1) / 2.0
    o.Nbl = sum(binfracs) * o.Nbl_full

    # Build baseline bins
    o.bl_bins = list([
        { 'b': bmax,
          'bfrac': frac,
          'bcount': frac * o.Nbl_full
        } for bmax, frac in zip(bins, binfracs)])

    # Wavelengths
    o.wl_max = o.c / o.freq_min  # Maximum Wavelength
    o.wl_min = o.c / o.freq_max  # Minimum Wavelength
    o.wl = 0.5 * (o.wl_max + o.wl_min)  # Representative Wavelength

    # Calculate number of subbands
    o.Nsubbands = ceiling(log(o.wl_max/o.wl_min)/log(o.max_subband_freq_ratio))
    o.Qsubband = (o.wl_max/o.wl_min)**(1./o.Nsubbands)



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

    # grid width in wavelengths
    o.Lambda_grid = 1 / o.Theta_pix
    o.Lambda_bl = 2 * o.Bmax / o.wl_sb_min

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
    o.Nf_vis = BLDep(b,
        Min(Max(o.Nf_out, o.Nf_no_smear),o.Nf_max))
    o.Nf_vis_backward = BLDep(b,
        Min(Max(o.Nf_out, o.Nf_no_smear_backward(b)),o.Nf_max))
    o.Nf_vis_predict = BLDep(b,
        Min(Max(o.Nf_out, o.Nf_no_smear_predict(b if o.scale_predict_by_facet else o.Bmax)),
            o.Nf_max))

    # Determine frequency buckets we need to do FFT for
    o.Nf_FFT_backward = o.Nf_out
    o.Nf_FFT_predict = Min(o.Nf_max, o.Ntt * o.Nf_out)
    if o.pipeline == Pipelines.DPrepA:
        o.Nf_FFT_backward = o.Ntt * o.Nf_out
        o.Nf_FFT_predict = o.Ntt * o.Nf_out
    if o.pipeline in [Pipelines.DPrepB, Pipelines.DPrepC, Pipelines.Fast_Img]:
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = Min(o.Ntt * o.Nf_min, o.Nf_out)

    # Determine frequency buckets we need to reproject
    o.Nf_proj_predict = o.Nf_FFT_predict
    o.Nf_proj_backward = o.Nf_FFT_backward
    if o.pipeline == Pipelines.DPrepA_Image:
        o.Nf_proj_predict = o.Ntt
        o.Nf_proj_backward = o.Ntt

def _apply_coalesce_equations(o, symbolify):
    """
    Determines amount of coalescing of visibilities in time.

    References:
     * SKA-TEL-SDP-0000040 01D section A - Covering the Time Axis
     * SKA-TEL-SDP-0000040 01D section D - Visibility Averaging and Coalescing
    """

    # Time averaging scaled for max baseline. Note that we have
    # two averaging steps to consider - one in ingest and then one
    # after phase rotation. We assume that "global" averaging must
    # retain enough information to prevent smearing for a
    # hypothetical imaging process using the full field of view,
    # but reduce that requirement to the actual facet field of
    # view later. We also assume that we can chose these two
    # averaging degrees independently without any ill effects.
    b = Symbol('b')
    combine_samples = lambda theta: BLDep(b,
        Max(floor(o.epsilon_f_approx * o.wl /
                  (theta * o.Omega_E * b * o.Tint_used)), 1.))
    o.combine_time_samples = combine_samples(o.Theta_fov_total)
    o.combine_time_samples_facet = combine_samples(o.Theta_fov)

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
            Min(o.Tint_used * o.combine_time_samples_facet(b), o.Tion))
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

    # Total visibility rate (visibilities per second per beam, per
    # polarisation) after frequency channels have been combined where
    # possible. We focus on imaging pipelines here, so we remove
    # auto-correlations. If they are required by science pipelines, we
    # assume that they are tracked separately as a data product not
    # covered by the parametric model.
    if o.global_blcoal:
        o.Rvis = blsum(b, o.Nf_vis(b) / Min(o.Tcoal_skipper(b), 1.2, o.Tion))
    else:
        o.Rvis = blsum(b, o.Nf_vis(b) / o.Tint_used)

    # Visibility rate for backward step, allow coalescing
    # in time and freq prior to gridding
    o.Rvis_backward = blsum(b, o.Nf_vis_backward(b) / o.Tcoal_backward(b))
    # Visibility rate for predict step
    o.Rvis_predict = blsum(b, o.Nf_vis_predict(b) / o.Tcoal_predict(b))


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
    o.DeltaW_SShot = BLDep(b, b * sin(o.Omega_E * o.Tsnap) / (2. * o.wl))
    o.DeltaW_max = BLDep(b, o.Qw * Max(o.DeltaW_SShot(b), o.DeltaW_Earth(b)))
    if symbolify == 'helper':
        o.DeltaW_Earth = Symbol(o.make_symbol_name('DeltaW_Earth'))
        o.DeltaW_SShot = Symbol(o.make_symbol_name('DeltaW_SShot'))
        o.DeltaW_max = Symbol(o.make_symbol_name('DeltaW_max'))
    o.DeltaW_wproj = BLDep(b, Min(o.DeltaW_stack, o.DeltaW_max(b)))

    # Eq. 25, w-kernel support size. Note possible difference in
    # cellsize assumption!
    def Ngw(deltaw, fov):
        return 2 * fov * sqrt((deltaw * fov / 2.) ** 2 +
                              (deltaw**1.5 * fov / (2 * pi * o.epsilon_w)))
    o.Ngw_backward = BLDep(b, Ngw(o.DeltaW_wproj(b), o.Theta_fov))
    o.Ngw_predict = BLDep(b, Ngw(o.DeltaW_wproj(b), o.Theta_fov_predict))


    # TODO: Check split of kernel size for backward and predict steps.
    # squared linear size of combined W and A kernels; used in eqs 23 and 32
    o.Nkernel_AW_backward = BLDep(b, (o.Ngw_backward(b) ** 2 + o.Naa ** 2)**0.5)
    o.Nkernel2_backward = BLDep(b, o.Nkernel_AW_backward(b)**2)
    # squared linear size of combined W and A kernels; used in eqs 23 and 32
    o.Nkernel_AW_predict = BLDep(b, (o.Ngw_predict(b) ** 2 + o.Naa ** 2)**0.5)
    o.Nkernel2_predict = BLDep(b, o.Nkernel_AW_predict(b)**2)


def _apply_ingest_equations(o):
    """
    Ingest equations

    References: SKA-TEL-SDP-0000040 01D section 3.3 - The Fast and Buffered pre-processing pipelines
    """

    if o.pipeline == Pipelines.Ingest:
        o.set_product(Products.Receive,
            T = o.Tsnap, N = o.Nbeam * o.Nf_min,
            Rflop = 4 * o.Npp * o.Rvis_ingest / o.Nf_min +
                    1000 * o.Na / o.Tint_used,
            Rout = o.Mvis * o.Npp * o.Rvis_ingest / o.Nf_min)
        o.set_product(Products.Flag,
            T = o.Tsnap, N = o.Nbeam * o.Nf_min,
            Rflop = 278 * o.Npp * o.Rvis_ingest / o.Nf_min,
            Rout = o.Mvis * o.Npp * o.Rvis_ingest / o.Nf_min)
        o.set_product(Products.Demix,
            T = o.Tsnap, N = o.Nbeam * o.Nf_min,
            Rflop = o.Rvis_ingest *
                      (154 * o.NAteam + 84 +
                       (o.NAteam**2 + o.NAteam * (33 + 24 * o.Nsolve) + 64)
                       / (o.Tion / o.Tint_used) / (o.Nf_max / o.Nf_min)),
            Rout = o.Mvis * o.Npp * o.Rvis_ingest / o.Nf_min)
        o.set_product(Products.Average,
            T = o.Tsnap, N = o.Nbeam * o.Nf_min,
            # Slight overestimation, as Rvis_ingest includes autocorrelations
            Rflop = blsum(Symbol('b'), 8 * o.Npp * o.Rvis_ingest / o.Nf_min / o.Nbl),
            Rout = o.Mvis * o.Npp * o.Rvis / o.Nf_min)


def _apply_flag_equations(o):
    """ Flagging equations for non-ingest pipelines"""

    if not (o.pipeline == Pipelines.Ingest):
        o.set_product(Products.Flag,
            T=o.Tsnap, N=o.Nbeam * o.Nmajortotal * o.Nf_min_gran,
            Rflop=279 * o.Npp * o.Rvis / o.Nf_min_gran,
            Rout = o.Mvis * o.Npp * o.Rvis / o.Nf_min_gran)


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

    o.set_product(Products.Visibility_Weighting, T=o.Tsnap,
        N = BLDep(b, o.Nmajortotal * o.Nbeam * o.Npp *
                     o.Nfacet**2 * o.Nf_min_gran),
        Rflop = blsum(b, 2 * 8 * o.Nf_vis_backward(b) / o.Nf_min_gran / o.Tcoal_backward(b)),
        Rout = blsum(b, o.Mvis / o.Tcoal_backward(b)))

    # Assume 8 flops per default. For image-domain gridding we
    # need 6 flops additionally.
    Rflop_per_vis = 8
    if not isinstance(o.image_gridding, Symbol) and o.image_gridding > 0:
        Rflop_per_vis = 8 + 6

    o.set_product(Products.Grid, T=o.Tsnap,
        N = BLDep(b, o.Nmajortotal * o.Nbeam * o.Npp * o.Ntt_backward *
                     o.Nfacet**2 * o.Nf_vis_backward(b)),
        Rflop = blsum(b, Rflop_per_vis * o.Nmm * o.Nkernel2_backward(b) / o.Tcoal_backward(b)),
        Rout = o.Mcpx * o.Npix_linear * (o.Npix_linear / 2 + 1) / o.Tsnap)
    o.set_product(Products.Degrid, T = o.Tsnap,
        N = BLDep(b, o.Nmajortotal * o.Nbeam * o.Npp * o.Ntt_predict *
                     o.Nfacet_predict**2 * o.Nf_vis_predict(b)),
        Rflop = blsum(b, Rflop_per_vis * o.Nmm * o.Nkernel2_predict(b) / o.Tcoal_predict(b)),
        Rout = blsum(b, o.Mvis / o.Tcoal_predict(b)))


def _apply_fft_equations(o):
    """
    Discrete fourier transformation of grids to images (and back)

    References: SKA-TEL-SDP-0000040 01D section 3.6.13 - FFT and iFFT
    """

    if not o.pipeline in Pipelines.imaging: return

    # Determine number of w-stacks we need
    o.Nwstack = Max(1, o.DeltaW_max(o.Bmax) / o.DeltaW_stack)
    o.Nwstack_predict = Max(1, o.DeltaW_max(o.Bmax) / o.DeltaW_stack)

    # These are real-to-complex for which the prefactor in the FFT is 2.5
    o.set_product(Products.FFT, T = o.Tsnap,
        N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_FFT_backward * o.Nfacet**2,
        Rflop = 2.5 * o.Nwstack * o.Npix_linear ** 2 * log(o.Npix_linear**2, 2) / o.Tsnap,
        Rout = o.Mpx * o.Npix_linear**2 / o.Tsnap)
    o.set_product(Products.IFFT, T = o.Tsnap,
        N = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_FFT_predict * o.Nfacet_predict**2,
        Rflop = 2.5 * o.Nwstack_predict * o.Npix_linear_predict**2 * log(o.Npix_linear_predict**2, 2) / o.Tsnap,
        Rout = o.Mcpx * o.Npix_linear_predict * (o.Npix_linear_predict / 2 + 1) / o.Tsnap)


def _apply_reprojection_equations(o):
    """
    Re-projection of skewed images as generated by w snapshots

    References: SKA-TEL-SDP-0000040 01D section 3.6.14 - Reprojection
    """

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


def _apply_minor_cycle_equations(o):
    """
    Minor Cycles implementing deconvolution / cleaning

    References:
      * SKA-TEL-SDP-0000040 01D section 3.6.16 - Subtract Image Component
      * TCC-SDP-151123-2 - Recommendations
    """

    o.Nf_identify = 0
    if not o.pipeline in Pipelines.imaging: return

    if o.pipeline in (Pipelines.ICAL, Pipelines.DPrepA, Pipelines.DPrepA_Image):
        # Search for every sub-band independently
        o.Nf_identify = o.Nsubbands
    elif o.pipeline in (Pipelines.DPrepB, Pipelines.DPrepC):
        # Always search in all frequency space
        o.Nf_identify = o.Nf_out
    else:
        # No cleaning - e.g. fast imaging
        return

    # Identification is assumed to only use I_0, but this still
    # requires multiplication of one row of the Hessian matrix. We
    # assume that this is done indepedently per facet (with or
    # without consolidation), therefore we use Npix*Nfacet
    # including overlap
    o.set_product(Products.Identify_Component,
        T = o.Tobs,
        N = o.Nmajortotal * o.Nbeam * o.Nf_identify * o.Nfacet**2,
        Rflop = 2 * (o.Npix_linear**2  + o.Nminor * o.Npatch**2) * o.Ntt * o.Nscales/ o.Tobs,
        Rout = o.Nminor * o.Mcpx / o.Tobs)

    # Subtract on all scales, polarisations and taylor terms
    o.set_product(Products.Subtract_Image_Component,
        T = o.Tobs,
        N = o.Nmajortotal * o.Nbeam * o.Nf_identify * o.Nfacet**2,
        Rflop = 2 * o.Npp * o.Nminor * o.Ntt * o.Nscales * o.Npatch**2 / o.Nfacet**2 / o.Tobs,
        Rout = o.Mpx * o.Ntt * o.Npix_linear**2 / o.Tobs)

    # Working memory requirements according to TCC-SDP-151123-2
    o.M_MSMFS = o.Mpx * o.Nf_identify * (o.Ntt * (o.Nscales + 1) + o.Nscales) * (o.Npix_linear * o.Nfacet)**2


def _apply_calibration_equations(o):
    """
    Self-calibration using predicted visibilities

    References: SKA-TEL-SDP-0000040 01D section 3.6.5 - Solve
    """

    # Number of flops needed per solution interval
    o.NFlop_solver = 48 * o.Npp * o.Nsolve * o.Na**2
    # Number of flops required for averaging one vis. The steps are evaluate the complex phasor for 
    # the model phase (16 flops), multiply Vis by that phasor (16 flops) then average (8 flops)
    o.NFlop_averager = 40 * o.Npp

    # Collect calibration windows to solve
    G_cal = B_cal = I_cal = dict(Ndircal=0, Nfcal=1, Tcal=1)
    if o.pipeline == Pipelines.ICAL:
        G_cal = dict(Ndircal=1, Nfcal=o.Nsubbands, Tcal=o.tICAL_G)
        B_cal = dict(Ndircal=1, Nfcal=o.NB_parameters, Tcal=o.tICAL_B)
        I_cal = dict(Ndircal=o.NIpatches, Nfcal=o.Nsubbands, Tcal=o.tICAL_I)
    elif o.pipeline == Pipelines.RCAL:
        G_cal = dict(Ndircal=1, Nfcal=o.Nf_out, Tcal=o.tRCAL_G)

    # Calculate number of calibration problems total / per snapshot&subband
    def calcNcal(Nf, t, Ndircal, Nfcal, Tcal):
        return Ndircal * Max(1, Nfcal / Nf) * Max(1, t / Tcal)
    o.Ncal_G_obs = calcNcal(1, o.Tobs, **G_cal)
    o.Ncal_B_obs = calcNcal(1, o.Tobs, **B_cal)
    o.Ncal_I_obs = calcNcal(1, o.Tobs, **I_cal)
    o.Ncal_G_solve = calcNcal(o.Nsubbands, o.Tsolve, **G_cal)
    o.Ncal_B_solve = calcNcal(o.Nsubbands, o.Tsolve, **B_cal)
    o.Ncal_I_solve = calcNcal(o.Nsubbands, o.Tsolve, **I_cal)

    # Global calibration solution size
    o.Mcal_out = o.Mjones * o.Na * (o.Ncal_G_obs + o.Ncal_B_obs + o.Ncal_I_obs)

    # Calibration problem & solutions sizes (per solve interval)
    o.Mcal_solve_in = 2 * o.Mvis * o.Npp * o.Nbl * (o.Ncal_G_solve + o.Ncal_B_solve + o.Ncal_I_solve)
    o.Mcal_solve_out = o.Mjones * o.Na * (o.Ncal_G_solve + o.Ncal_B_solve + o.Ncal_I_solve)

    # How many directions do we need to solve in total? Assume we have
    # to average for each separately.
    N_solve = G_cal['Ndircal'] + B_cal['Ndircal'] + I_cal['Ndircal']
    if N_solve > 0:

        # Averaging needs to be done for each calibration method
        Rflop_averaging = o.NFlop_averager * N_solve * o.Rvis.eval_sum(o.bl_bins)
        Rflop_solving   = o.NFlop_solver * (o.Ncal_G_solve + o.Ncal_B_solve + o.Ncal_I_solve) / o.Tsolve
        o.set_product(Products.Solve,
            T = o.Tsolve,
            # We do one calibration to start with (using the original
            # LSM from the GSM and then we do Nselfcal more.
            N = (o.Nselfcal + 1) * o.Nsubbands * o.Nbeam,
            Rflop = Rflop_solving + Rflop_averaging / o.Nsubbands,
            Rout = o.Mcal_solve_out / o.Tsolve)

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
            Rflop = blsum(b,
                    (32 * o.Na**2 * o.Nsource + (10 + 224 + 32) * o.Na * o.Nsource + 128 * o.Na * o.Na)
                          * o.Rvis(b) / o.Nf_min_gran / o.Nbl),
            Rout = blsum(b, o.Npp * o.Mvis * o.Rvis(b) / o.Nf_min_gran))


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


def _apply_major_cycle_equations(o):
    """
    Subtract predicted visibilities from last major cycle

    References: SKA-TEL-SDP-0000040 01D section 3.6.6 - Subtract
    """

    # Note that we assume this is done for every Selfcal and Major Cycle
    if o.pipeline in Pipelines.imaging:
        b = Symbol('b')
        o.set_product(Products.Subtract_Visibility,
            T = o.Tsnap,
            N = o.Nmajortotal * o.Nbeam * o.Nf_min_gran,
            Rflop = blsum(b, 2 * o.Npp * o.Rvis(b) / o.Nf_min_gran),
            Rout = blsum(b, o.Mvis * o.Npp * o.Rvis(b) / o.Nf_min_gran))


def _apply_kernel_equations(o):
    """
    Generate parameters for Convolution kernels

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

    # Number of times kernel convolution needs to be repeated to
    # generate all oversampling values used in gridding.
    o.Ngcf_used_backward = BLDep(b, 1)
    o.Ngcf_used_predict = BLDep(b, 1)

    if o.on_the_fly:
        o.Nf_gcf_backward = o.Nf_vis_backward
        o.Nf_gcf_predict  = o.Nf_vis_predict
        o.Tkernel_backward = o.Tcoal_backward
        o.Tkernel_predict  = o.Tcoal_predict
    elif not isinstance(o.image_gridding, Symbol) and o.image_gridding > 0:
        # For image-domain gridding, we multiply kernels that get
        # applied closely together by the visibilities in image
        # space, apply phase ramps to account for their shift
        # relative to each other, FFT the result once, then add to
        # the grid. So w need larger kernels, gridding is more
        # expensive, but we need less "kernels" (which are now
        # "sub-grids"). Note that sub-grid convolution now happens
        # *after* gridding!
        o.Nf_gcf_backward = BLDep(b, o.Theta_fov * b / o.wl_sb_min / o.image_gridding
                                  * (1 - 1 / o.Qsubband))
        o.Nf_gcf_predict = BLDep(b, o.Theta_fov_predict * b / o.wl_sb_min / o.image_gridding
                                 * (1 - 1 / o.Qsubband))
        o.Tkernel_backward = BLDep(b, Max(o.Tcoal_backward(b), Min(o.Tsnap,
            24 * 3600 * o.image_gridding / 2 / pi / o.Theta_fov / b * o.wl_sb_min)))
        o.Tkernel_predict = BLDep(b, Max(o.Tcoal_backward(b), Min(o.Tsnap,
            24 * 3600 * o.image_gridding / 2 / pi / o.Theta_fov_predict / b * o.wl_sb_min)))
    else:
        # For both of the following, maintain distributability;
        # need at least Nf_min kernels.
        o.Nf_gcf_backward = BLDep(b, Max(o.Nf_gcf_backward_nosmear(b), o.Nf_min))
        o.Nf_gcf_predict  = BLDep(b, Max(o.Nf_gcf_predict_nosmear(b),  o.Nf_min))
        o.Tkernel_backward = BLDep(b, o.Tion)
        o.Tkernel_predict  = BLDep(b, o.Tion)

        # Determine number of visibilities per kernel, approximate
        # number of oversampling values used.
        o.Nvis_gcf_backward = BLDep(b,
            o.Nf_vis_backward(b) / o.Nf_gcf_backward(b) * o.Tkernel_backward(b) / o.Tcoal_backward(b))
        o.Nvis_gcf_predict = BLDep(b,
            o.Nf_vis_predict(b) / o.Nf_gcf_predict(b) * o.Tkernel_predict(b) / o.Tcoal_predict(b))
        o.Ngcf_used_backward = BLDep(b, o.Qgcf**2 * (1 - (1 - 1 / o.Qgcf**2)**o.Nvis_gcf_backward(b)))
        o.Ngcf_used_predict = BLDep(b, o.Qgcf**2 * (1 - (1 - 1 / o.Qgcf**2)**o.Nvis_gcf_predict(b)))


def _apply_kernel_product_equations(o):
    """
    Generate parameters for Convolution kernels

    References:
     * SKA-TEL-SDP-0000040 01D section E - Re-use of Convolution Kernels
    """

    b = Symbol('b')

    # Baselines to cover with kernels.
    if o.NAProducts == 'all' or o.on_the_fly or not isinstance(o.image_gridding, Symbol) and o.image_gridding > 0:
        bins = o.bl_bins

    else:
        # If we do not need a separate A^A-kernel per baseline,
        # just make the appropriate number of A-kernels at a
        # resolution appropriate for the longest baseline, and
        # assume that all other baselines can use it
        bins = [ { 'b': o.Bmax, 'bcount': o.NAProducts } ]

    if o.pipeline in Pipelines.imaging:

        # The following two equations correspond to Eq. 35
        o.set_product(Products.Gridding_Kernel_Update,
            T = BLDep(b, o.Tkernel_backward(b)),
            N = BLDep(b, o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_gcf_backward(b) * o.Nfacet**2),
            bins = bins,
            Rflop = blsum(b, 5. * o.Nmm * o.Ngcf_used_backward(b) * o.Nkernel_AW_backward(b)**2 * log(o.Nkernel_AW_backward(b)**2, 2) / o.Tkernel_backward(b)),
            Rout = blsum(b, 8 * o.Qgcf**3 * o.Ngw_backward(b)**3 / o.Tkernel_backward(b)))
        o.set_product(Products.Degridding_Kernel_Update,
            T = BLDep(b,o.Tkernel_predict(b)),
            N = BLDep(b,o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_gcf_predict(b) * o.Nfacet_predict**2),
            bins = bins,
            Rflop = blsum(b, 5. * o.Nmm * o.Ngcf_used_predict(b) * o.Nkernel_AW_predict(b)**2 * log(o.Nkernel_AW_predict(b)**2, 2) / o.Tkernel_predict(b)),
            Rout = blsum(b, 8 * o.Qgcf**3 * o.Ngw_predict(b)**3 / o.Tkernel_predict(b)))


def _apply_phrot_equations(o):
    """
    Phase rotation (for the faceting)

    References: SKA-TEL-SDP-0000040 01D section 3.6.9 - Phase Rotation
    """

    if not o.pipeline in Pipelines.imaging: return

    b = Symbol("b")

    # 28 FLOPS per visiblity. Only do it if we need to facet.

    # Predict phase rotation: Input from facets at predict
    # visibility rate, output at same rate.
    if o.scale_predict_by_facet:
        o.set_product(Products.PhaseRotationPredict,
            T = o.Tsnap,
            N = sign(o.Nfacet - 1) * o.Nmajortotal * o.Npp * o.Nbeam * o.Nf_min_gran *
                o.Ntt_predict * o.Nfacet**2 ,
            Rflop = blsum(b, 28 * o.Nf_vis(b) / o.Nf_min_gran / o.Tint_used),
            Rout = blsum(b, o.Mvis * o.Nf_vis(b) / o.Nf_min_gran / o.Tint_used))

    # Backward phase rotation: Input at overall visibility
    # rate, output averaged down to backward visibility rate.
    o.set_product(Products.PhaseRotation,
        T = o.Tsnap,
        N = sign(o.Nfacet - 1) * o.Nmajortotal * o.Npp * o.Nbeam * o.Nfacet**2 * o.Nf_min_gran,
        Rflop = blsum(b, 28 * o.Nf_vis(b) / o.Nf_min_gran / o.Tint_used),
        Rout = blsum(b, o.Mvis * o.Rvis_backward(b) / o.Nf_min_gran))


def _apply_flop_equations(o):
    """Calculate overall flop rate"""

    # Sum up products
    o.Rflop = sum(o.get_products('Rflop').values())

    # Calculate interfacet IO rate for faceting: TCC-SDP-151123-1-1 rev 1.1
    o.Rinterfacet = \
        2 * o.Nmajortotal * Min(3.0, 2.0 + 18.0 * o.r_facet_base) * \
        (o.Nfacet * o.Npix_linear)**2 * o.Nf_out * 4  / o.Tobs


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
    o.Mbuf_vis = o.buffer_factor * o.Npp * o.Rvis.eval_sum(o.bl_bins) * o.Nbeam * o.Mvis * o.Tobs

    # Visibility read rate
    #
    # According to SDPPROJECT-133 (JIRA) assume that we only need
    # to read all visibilities twice per major cycle and beam.
    o.Rio = 2.0 * o.Nbeam * o.Npp * (1 + o.Nmajortotal) * o.Rvis.eval_sum(o.bl_bins) * o.Mvis
    # o.Rio = o.Nbeam * o.Npp * (1 + o.Nmajortotal) * o.Rvis * o.Mvis * o.Nfacet ** 2

    # Facet visibility rate
    #
    # Visibilities that go into a single facet
    o.Rfacet_vis = o.Nbeam * o.Npp * o.Nmajortotal * o.Rvis_backward.eval_sum(o.bl_bins) * o.Mvis

    # Snapshot Size
    #
    # The amount of visibility data considered together for a snapshot
    # and FFT frequency band
    o.Msnap = o.Mvis * o.Rvis * o.Npp * o.Tsnap / o.Nf_FFT_backward
    o.Msnap_predict = o.Mvis * o.Rvis_predict * o.Npp * o.Tsnap / o.Nf_FFT_predict

    # Image sizes
    #
    # First for just a single image (single polarisation, frequency
    # and beam), then the cube of all results put together
    o.Mfacet = o.Mpx * o.Npix_linear**2
    o.Mfacet_cube = o.Nbeam * o.Mfacet * o.Npp * o.Nf_out
    o.Mimage = o.Mpx * o.Npix_linear_fov_total**2
    o.Mimage_cube = o.Nbeam * o.Mimage * o.Npp * o.Nf_out

    # Image write rate
    #
    # Basically the output rate of the FFTs
    o.Rimage = o.Nmajortotal * o.Nbeam * o.Npp * o.Nf_FFT_backward * o.Mpx * o.Npix_linear**2 * o.Nfacet**2 / o.Tsnap

    # Convolution kernel cache size
    #
    # TODO: re-implement this equation within a better description
    # of where kernels will be stored etc.  (allowing storage of a
    # full observation while simultaneously capturing the next)
    o.Mw_cache = (o.Ngw_predict(o.Bmax) ** 3) * (o.Qgcf ** 3) * o.Mcpx * o.Nbeam * \
                 o.Nsubbands * o.Nfacet ** 2
