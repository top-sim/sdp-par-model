"""
Enumerates and defines the parameters of the telescopes, bands,
pipelines, etc. Several methods are supplied by which values can be
found by lookup as well (e.g. finding the telescope that is associated
with a given mode)

Parameters defined here include telecope parameters as well as
physical constants. Generally, the output of the functions are
ParameterContainer objects (usually referred to by the variable o in
the methods below) that has parameters as fields.
"""

from __future__ import print_function

from sympy import symbols
import numpy as np

from .container import ParameterContainer

class Constants:
    """
    A new class that takes over the roles of sympy.physics.units and astropy.const, because it is simpler this way
    """
    kilo = 1000
    mega = 1000000
    giga = 1000000000
    tera = 1000000000000
    peta = 1000000000000000

    degree = np.pi / 180
    arcminute = np.pi / 180 / 60
    arcsecond = np.pi / 180 / 60 / 60


class Telescopes:
    """
    Enumerate the possible telescopes to choose from (used in
    :meth:`apply_telescope_parameters`)
    """
    # The originally planned (pre-rebaselining) SKA1 telescopes. TODO: remove?
    SKA1_Low_old = 'SKA1_Low_old'
    SKA1_Mid_old = 'SKA1_Mid_old'
    SKA1_Sur_old = 'SKA1_Survey_old'
    # The rebaselined SKA1 telescopes
    SKA1_Low = 'SKA1_Low'
    SKA1_Mid = 'SKA1_Mid'
    # Proposed SKA2 telescopes
    SKA2_Low = 'SKA2_Low'
    SKA2_Mid = 'SKA2_Mid'

    # Currently supported telescopes (will show up in notebooks)
    available_teles = (SKA1_Low, SKA1_Mid)

class Bands:
    """
    Enumerate all possible bands (used in :meth:`apply_band_parameters`)
    """
    # SKA1 Bands
    Low = 'Low'
    Mid1 = 'Mid1'
    Mid2 = 'Mid2'
    Mid3 = 'Mid3'
    Mid4 = 'Mid4'
    Mid5A = 'Mid5A'
    Mid5B = 'Mid5B'
    Mid5C = 'Mid5C'
    # SKA1 Survey bands - Now obsolete?
    Sur1 = 'Sur1'
    Sur2A = 'Sur2A'
    Sur2B = 'Sur2B'
    Sur3A = 'Sur3A'
    Sur3B = 'Sur3B'
    # SKA2 Bands
    SKA2Low = 'LOWSKA2'
    SKA2Mid = 'MIDSKA2'

    # group the bands defined above into logically coherent sets
    low_bands = {Low}
    mid_bands = {Mid1, Mid2, Mid3, Mid4, Mid5A, Mid5B, Mid5C}
    survey_bands = {Sur1, Sur2A, Sur2B, Sur3A, Sur3B}  # Now obsolete?
    low_bands_ska2 = {SKA2Low}
    mid_bands_ska2 = {SKA2Mid}

    available_bands = (Low,
                       Mid1, Mid2, Mid5A, Mid5B, Mid5C,
                       Sur1)

class Products:
    """
    Enumerate the SDP Products used in pipelines
    """
    Alert = 'Alert'
    Average = 'Average'
    Calibration_Source_Finding = 'Calibration Source Finding'
    Correct = 'Correct'
    Degrid = 'Degrid'
    DFT = 'DFT'
    Demix = 'Demix'
    FFT = 'FFT'
    Flag = 'Flag'
    Grid = 'Grid'
    Gridding_Kernel_Update = 'Gridding Kernel Update'
    Degridding_Kernel_Update = 'Degridding Kernel Update'
    Identify_Component = 'Identify Component'
    Extract_LSM = 'Extract_LSM'
    IFFT = 'IFFT'
    Image_Spectral_Averaging = 'Image Spectral Averaging'
    Image_Spectral_Fitting = 'Image Spectral Fitting'
    Notify_GSM = 'Update GSM'
    PhaseRotation = 'Phase Rotation'
    PhaseRotationPredict = 'Phase Rotation Predict'
    QA = 'QA'
    Receive = 'Receive'
    Reprojection = 'Reprojection'
    ReprojectionPredict = 'Reprojection Predict'
    Select = 'Select'
    Solve = 'Solve'
    Source_Find = 'Source Find'
    Subtract_Visibility = 'Subtract Visibility'
    Subtract_Image_Component = 'Subtract Image Component'
    Update_LSM = 'Update LSM'
    Visibility_Weighting = 'Visibility Weighting'

class Pipelines:
    """
    Enumerate the SDP pipelines. These must map onto the Products. The HPSOs invoke these.
    """
    Ingest = 'Ingest' # Ingest pipeline
    ICAL = 'ICAL'     # ICAL (the big one):produce calibration solutions
    RCAL = 'RCAL'     # Produce calibration solutions in real time
    DPrepA = 'DPrepA' # Produce continuum taylor term images in Stokes I
    DPrepA_Image = 'DPrepA_Image' # Produce continuum taylor term images in Stokes I as CASA does in images
    DPrepB = 'DPrepB' # Produce coarse continuum image cubes in I,Q,U,V (with Nf_out channels)
    DPrepC = 'DPrepC' # Produce fine spectral resolution image cubes un I,Q,U,V (with Nf_out channels)
    DPrepD = 'DPrepD' # Produce calibrated, averaged (In time and freq) visibility data
    Fast_Img = 'Fast_Img' # Produce continuum subtracted residual image every 1s or so

    input = [Ingest]
    imaging = [RCAL, ICAL, DPrepA, DPrepA_Image, DPrepB, DPrepC, Fast_Img]
    output = [DPrepA, DPrepA_Image, DPrepB, DPrepC, Fast_Img]
    all = [Ingest, ICAL, RCAL, DPrepA, DPrepA_Image, DPrepB, DPrepC, DPrepD, Fast_Img]
    pure_pipelines = [Ingest, ICAL, RCAL, DPrepA, DPrepA_Image, DPrepB, DPrepC, DPrepD, Fast_Img]

    # Pipelines that are currently supported (will show up in notebooks)
    available_pipelines = all

class HPSOs:
    """
    Enumerate the High Priority Science Objectives (used in :meth:`apply_hpso_parameters`)
    """
    hpso_max_Low_c = 'max_LOW_continuum'
    hpso_max_Low_s = 'max_LOW_spectral'
    hpso_max_Mid_c = 'max_MID_continuum'
    hpso_max_Mid_s = 'max_MID_spectral'
    hpso_max_band5_Mid_c = 'max_Band5_MID_continuum'
    hpso_max_band5_Mid_s = 'max_Band5_MID_spectral'
    hpso01ICAL = '01ICAL'
    hpso01DPrepA = '01DPrepA'
    hpso01DPrepB = '01DPrepB'
    hpso01DPrepC = '01DPrepC'
    hpso02AICAL = '02AICAL'
    hpso02ADprepA = '02ADPrepA'
    hpso02ADPrepB = '02ADPrepB'
    hpso02ADPrepC = '02ADPrepC'
    hpso02BICAL = '02BICAL'
    hpso02BDPrepA = '02BDPrepA'
    hpso02BDPrepB = '02BDPrepB'
    hpso02BDPrepC = '02BDPrepC'
    hpso13ICAL = '13ICAL'
    hpso13DPrepA = '13DPrepA'
    hpso13DPrepB = '13DPrepB'
    hpso13DPrepC = '13DPrepC'
    hpso14ICAL = '14ICAL'
    hpso14DPrepA = '14DPrepA'
    hpso14DPrepB = '14DPrepB'
    hpso14DPrepC = '14DPrepC'
    hpso15ICAL = '15ICAL'
    hpso15DPrepA = '15DPrepA'
    hpso15DPrepB = '15DPrepB'
    hpso15DPrepC = '15DPrepC'
    hpso22ICAL = '22ICAL'
    hpso22DprepA = '22DPrepA'
    hpso22DprepB = '22DPrepB'
    hpso27ICAL = '27ICAL'
    hpso27DPrepA = '27DPrepA'
    hpso27DPrepB = '27DPrepB'
    hpso32ICAL = '32ICAL'
    hpso32DPrepB = '32DPrepB'
    hpso37aICAL = '37aICAL'
    hpso37aDprepA = '37aDPrepA'
    hpso37aDprepB = '37aDPrepB'
    hpso37bICAL = '37bICAL'
    hpso37bDprepA = '37bDPrepA'
    hpso37bDprepB = '37bDPrepB'
    hpso37cICAL = '37cICAL'
    hpso37cDPrepA = '37cDPrepA'
    hpso37cDPrepB = '37cDPrepB'
    hpso38aICAL = '38aICAL'
    hpso38aDPrepA = '38aDPrepA'
    hpso38aDPrepB = '38aDPrepB'
    hpso38bICAL = '38bICAL'
    hpso38bDPrepA = '38bDPrepA'
    hpso38bDPrepB = '38bDPrepB'

    # group the HPSOs according to which telescope they refer to
    #    hpsos_using_SKA1Low = {hpso01, hpso02A, hpso02B}
    #hpsos_using_SKA1Mid = {hpso19, hpso22, hpso37a, hpso37b, hpso38a,
#hpso38b, hpso14c, hpso14s, hpso14sfull}
    #hpsos_originally_for_SKA1Sur = {hpso13, hpso15, hpso27c, hpso27s, hpso33, hpso35, hpso37c, hpso13c, hpso13s, hpso15c, hpso15s}
    # Because we are no longer building Survey, assume that the HPSOs intended for Survey will run on Mid?
#hpsos_using_SKA1Mid = hpsos_using_SKA1Mid | hpsos_originally_for_SKA1Sur

    # HPSOs that are currently supported (will show up in notebooks).
    # The High Priority Science Objective list below includes the
    # HPSOs that were originally intended for The Survey
    # telescope. These have since been reassigned to Mid.
    available_hpsos = [hpso_max_Low_c, hpso_max_Low_s, hpso_max_Mid_c, hpso_max_Mid_s,
                       hpso_max_band5_Mid_c, hpso_max_band5_Mid_s,
                       hpso01ICAL, hpso01DPrepA, hpso01DPrepB, hpso01DPrepC,
                       hpso02AICAL, hpso02ADprepA, hpso02ADPrepB, hpso02ADPrepC,
                       hpso02BICAL, hpso02BDPrepA, hpso02BDPrepB, hpso02BDPrepC,
                       hpso13ICAL, hpso13DPrepA, hpso13DPrepB, hpso13DPrepC,
                       hpso14ICAL, hpso14DPrepA, hpso14DPrepB, hpso14DPrepC,
                       hpso15ICAL, hpso15DPrepA, hpso15DPrepB, hpso15DPrepC,
                       hpso22ICAL, hpso22DprepA, hpso22DprepB,
                       hpso27ICAL, hpso27DPrepA, hpso27DPrepB,
                       hpso32ICAL, hpso32DPrepB,
                       hpso37aICAL,hpso37aDprepA,hpso37aDprepB,
                       hpso37bICAL, hpso37bDprepA, hpso37bDprepB,
                       hpso37cICAL, hpso37cDPrepA, hpso37cDPrepB,
                       hpso38aICAL, hpso38aDPrepA, hpso38aDPrepB,
                       hpso38bICAL, hpso38bDPrepA, hpso38bDPrepB]


def define_symbolic_variables(o):
    """
    This method defines the *symbolic* variables that we will use during computations
    and that need to be kept symbolic during evaluation of formulae. One reason to do this would be to allow
    the output formula to be optimized by varying this variable (such as with Tsnap and Nfacet)
    @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    @rtype : ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.Tsnap = symbols("Tsnap", positive=True)  # Snapshot timescale implemented
    o.Nfacet = symbols("Nfacet", integer=True, positive=True)  # Number of facets

    return o


def define_symbolic_variables_design_equations(o):
    """
    This method defines the *symbolic* variables that we will use during computations
    and that may need to be kept symbolic during evaluation. One reason to do this would be to allow
    the output formula to be optimized by varying these variables
    @param o: A supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    @rtype : ParameterContainer
    """
    assert isinstance(o, ParameterContainer)

    o.NfOut = symbols("N_f\,out", integer=True, positive=True)
    o.Nmajor = symbols("N_major", integer=True, positive=True)
    o.Nfacet = symbols("N_facet", integer=True, positive=True)
    o.Ncu = symbols("N_cu", integer=True, positive=True)  # number of compute units.
    o.RcuFLOP = symbols("R_cu\,FLOP", positive=True)  # peak FLOP capability of the compute unit
    o.RcuInter = symbols("R_cu\,inter", positive=True)  # maximum bandwidth of interconnect per Compute Unit
    o.RcuIo = symbols("R_cu\,io", positive=True)  # maximum I/O bandwidth of each compute unit to buffer
    o.Rfft = symbols("R_FFT", positive=True)
    o.Rinterfacet = symbols("R_interfacet", positive=True)
    o.Rrp = symbols("R_RP", positive=True)
    o.MuvGrid = symbols("M_uv\,grid", positive=True)
    o.McuWork = symbols("M_cu\,work", positive=True)  # Size of main working memory of the compute unit
    o.McuPool = symbols("M_cu\,pool", positive=True)  # Size of slower (swap) working memory of the compute unit
    o.McuBuf = symbols("M_cu\,buf", positive=True)  # Size of buffer (or share of data-island local buffer)

    o.RspecFLOP = symbols("R^spec_FLOP", positive=True)
    o.RcontFLOP = symbols("R^cont_FLOP", positive=True)
    o.RfastFLOP = symbols("R^fast_FLOP", positive=True)

    o.RspecIo = symbols("R^spec_io", positive=True)
    o.RcontIo = symbols("R^cont_io", positive=True)

    o.Fci = symbols("F_ci", positive=True)
    o.MspecBufVis = symbols("M^spec_buf\,vis", positive=True)
    o.McontBufVis = symbols("M^cont_buf\,vis", positive=True)
    return o


def apply_global_parameters(o):
    """
    Applies the global parameters to the parameter container object o.
    @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    @rtype : ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.c = 299792458  # The speed of light, in m/s (from sympy.physics.units.c)
    o.Omega_E = 7.292115e-5  # Rotation relative to the fixed stars in radians/second
    o.R_Earth = 6378136  # Radius if the Earth in meters (equal to astropy.const.R_earth.value)
    o.epsilon_w = 0.01  # Amplitude level of w-kernels to include
    #o.Mvis = 10.0  # Memory size of a single visibility datum in bytes. Set at 10 on 26 Jan 2016 (Ferdl Graser, CSP ICD)
    o.Mjones = 64.0  # Memory size of a Jones matrix (taken from Ronald's calibration calculations)
    o.Naa = 10  # Support Size of the A Kernel, in (linear) Pixels. Changed to 10, after PDR submission
    o.Nmm = 4  # Mueller matrix Factor: 1 is for diagonal terms only, 4 includes off-diagonal terms too.
    o.Npp = 4  # Number of polarization products
    o.Nw = 2  # Bytes per value
    o.Mpx = 8.0  # Memory size of an image pixel in bytes
    o.Mcpx = 16.0  # Memory size of a complex grid pixel in bytes
    o.Ndemix = 1000 # Number of time-frequency samples used in demixing
    o.NA = 10 # Number of A-team sources used in demixing
    # o.Qbw = 4.3 #changed from 1 to give 0.34 uv cells as the bw smearing limit. Should be investigated and linked to depend on amp_f_max, or grid_cell_error
    o.Qfcv = 1.0  #changed to 1 to disable but retain ability to see affect in parameter sweep.
    o.Qgcf = 8.0
    o.Qkernel = 10.0  #  epsilon_f/ o.Qkernel is the fraction of a uv cell we allow frequence smearing at edge of convoluion kernel to - i.e error on u,v, position one kernel-radius from gridding point.
    # o.grid_cell_error = 0.34 #found from tump time as given by SKAO at largest FoV (continuum).
    o.Qw = 1.0
    o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
    o.Qfov = 1.0 # Define this in case not defined below
    o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
    o.Tion = 10.0  #This was previously set to 60s (for PDR) May wish to use much smaller value.
    o.Tsnap_min = 0.1 #1.0 logically, this shoudl be set to Tdump, but odd behaviour happens for fast imaging. TODO
    o.Nf_min = 40  #minimum number of channels to still enable distributed computing, and to reconstruct 5 Taylor terms
    o.Fast_Img_channels = 40  #minimum number of channels to still enable distributed computing, and to calculate spectral images
    o.Nf_min_gran = 800 # minimum number of channels in predict output to prevent overly large output sizes
    o.Ntt = 5 # Number of Taylor terms to compute
    o.NB_parameters = 500 # Number of terms in B parametrization
    o.r_facet_base = 0.2 #fraction of overlap (linear) in adjacent facets.
    o.max_subband_freq_ratio = 1.35 #maximum frequency ratio supported within each subband. 1.35 comes from Jeff Wagg SKAO ("30% fractional bandwidth in subbands").
    o.buffer_factor = 2  # The factor by which the buffer will be oversized. Factor 2 = "double buffering".
    o.Mvis = 12  # Memory size of a single visibility datum in bytes. See below. Estimated value may change (again).
    o.Qfov_ICAL = 2.7 #Put the Qfov factor for the ICAL pipeline in here. It is used to calculate the correlator dump rate for instances where the maximum baseline used for an experiment is smaller than the maximum possible for the array. In that case, we might be able to request a longer correlator integration time in the correlator.

    # From CSP we are ingesting 10 bytes per visibility (single
    # polarization) built up as follows:
    #
    #   1 byte for time centroid
    # + 8 bytes for complex visibility
    # + 1 byte for flagging fraction, as per CSP ICD v.1.0.
    #     (i.e. 8 bytes for a complex value)
    # + 2 extra bytes for us to reconstruct timestamps etc.
    # + 2 bytes added by "ingest"
    # -----
    #  12 bytes per datum.
    #
    # Somewhere in the "Receive Visibilities" Function of the
    # Ingest Pipeline the 2 bytes additional information are
    # dropped (or used).  However, now we add:
    #
    # +4 byte for Weights (1 float)
    # +1 byte for Flags (1 bit per sample minimum).
    #
    # So the 12 bytes is the 8 for the complex value + 4 for the
    # Weight, which makes 12.  We should also add at least 1 bit
    # for Flags + some small amount of overhead for other meta
    # data.

    o.Nsource_find_iterations = 10 # Number of iterations in source finding
    o.Nsource = 1000 # Number of point sources modelled TODO: Should be set per HPSO
    o.Nminor = 1000 # Average number of minor cycles per major cycle
    o.Nsolve = 10 # Number of Stefcal iterations
    o.Nscales = 10 # Number of scales in MS-MFS - updated to match requirement: SDP_REQ-676 (L2)
    o.Npatch = 4097 # Number of pixels in clean patch

    # To be overridden by the pipelines
    o.Nf_FFT_backward = 0
    o.Nf_FFT_predict = 0
    o.Nmajor = 2
    o.Nselfcal = 3
    o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) 
    o.NAProducts = 'all'  # Number of A^A terms to be modelled
    o.tRCAL_G = 180.0 # Real time solution interval for Antenna gains
    o.tICAL_G = 1.0 # Solution interval for Antenna gains
    o.tICAL_B = 3600.0  # Solution interval for Bandpass
    o.tICAL_I = 1.0 # Solution interval for Ionosphere
    o.NIpatches = 1 # Number of ionospheric patches to solve

    # Pipeline variants
    o.on_the_fly = False
    o.blcoal = True
    o.scale_predict_by_facet = True
    o.image_gridding = 0  # Pixels to pad to allow image-plane gridding

    return o


def apply_telescope_parameters(o, telescope):
    """
    Applies the parameters that apply to the supplied telescope to the parameter container object o
    @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    @param telescope:
    @rtype : ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.set_param('telescope', telescope)

    if telescope == Telescopes.SKA1_Low:
        o.Bmax = 65000  # Actually constructed max baseline in *m*
        o.Ds = 35  # station "diameter" in metres
        o.Na = 512  # number of antennas
        o.Nbeam = 1  # number of beams
        o.Nf_max = 65536  # maximum number of channels
        o.B_dump_ref = 65000  # m
        o.Tint_min = 0.9  # Minimum correlator integration time (dump time) in *sec* - in reference design
        #o.baseline_bins = np.array((4900, 7100, 10400, 15100, 22100, 32200, 47000, 65000))  # m
        #o.baseline_bin_distribution = np.array(
            #(52.42399198, 7.91161595, 5.91534571, 9.15027832, 7.39594812, 10.56871804, 6.09159108, 0.54251081))#OLD 753 ECP Abs length only
            #(49.79516935, 7.2018153,  6.30406311, 9.87679703, 7.89016813, 11.59539474, 6.67869761, 0.65789474))#LOW ECP 160015 Abs length only
            #These layouts have been foreshortened with elevations of 50,60,70 degrees, and a range of azimuthal angles.
            #(56.96094258,   8.22266894,   7.55474842,  11.56658646,   8.05191328, 5.67275575,   1.85699165,   0.11339293)) #LOW ECP 160015
            #(60.31972106,   7.7165451,    6.92064107,  10.73309147,   7.3828517, 5.17047626,   1.6627458,    0.09392754)) #4a array
            #(60.22065697,   7.72434788,   7.05826222,  10.72169655,   7.36441056, 5.15572168   1.66138048   0.09352366)) #OLD 753 ECP
            
        
        o.baseline_bins = np.array((o.Bmax/16., o.Bmax/8., o.Bmax/4., o.Bmax/2., o.Bmax)) #neater baseline binning!
        o.baseline_bin_distribution = np.array((49.3626883, 13.32914111, 13.65062318, 17.10107961, 6.5564678))#LOW ECP 160015 Abs length only
            
            
        #o.amp_f_max = 1.08  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
        # o.NAProducts = o.nr_baselines # We must model the ionosphere for each station
        o.NAProducts = 'all' # We must model the ionosphere for each station
        o.tRCAL_G = 180.0
        o.tICAL_G = 10.0 # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 1.0 # Solution interval for Ionosphere
        o.NIpatches = 30 # Number of ionospheric patches to solve

    elif telescope == Telescopes.SKA1_Low_old:
        o.Bmax = 100000  # Actually constructed max baseline in *m*
        o.Ds = 35  # station "diameter" in metres
        o.Na = 1024  # number of antennas
        o.Nbeam = 1  # number of beams
        o.Nf_max = 256000  # maximum number of channels
        o.Tint_min = 0.6  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.B_dump_ref = 100000  # m
        o.baseline_bins = np.array((4900, 7100, 10400, 15100, 22100, 32200, 47000, 68500, 100000))  # m
        o.baseline_bin_distribution = np.array((49.361, 7.187, 7.819, 5.758, 10.503, 9.213, 8.053, 1.985, 0.121))
        o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
        o.NAProducts = 'all' # We must model the ionosphere for each station
        o.tRCAL_G = 180.0
        o.tICAL_G = 10.0 # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 1.0 # Solution interval for Ionosphere
        o.NIpatches = 30 # Number of ionospheric patches to solve

    elif telescope == Telescopes.SKA1_Mid:
        o.Bmax = 150000  # Actually constructed max baseline in *m*
        o.Ds = 13.5  # station "diameter" in metres, assume 13.5 as this matches the MeerKat antennas
        o.Na = 133 + 64  # number of antennas (expressed as the sum between new and Meerkat antennas)
        o.Nbeam = 1  # number of beams
        o.Nf_max = 65536  # maximum number of channels
        o.Tint_min = 0.14  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.B_dump_ref = 150000  # m
        # Rosie's conservative, ultra simple numbers (see Absolute_Baseline_length_distribution.ipynb)
        o.baseline_bins = np.array((5000.,7500.,10000.,15000.,25000.,35000.,55000.,75000.,90000.,110000.,130000.,150000)) #"sensible" baseline bins
        o.baseline_bin_distribution = np.array(( 6.14890420e+01,   5.06191389e+00 ,  2.83923113e+00 ,  5.08781928e+00, 7.13952645e+00,   3.75628206e+00,   5.73545412e+00,   5.48158127e+00, 1.73566136e+00,   1.51805606e+00,   1.08802653e-01 ,  4.66297083e-02))#July2-15 post-rebaselining, from Rebaselined_15July2015_SKA-SA.wgs84.197x4.txt % of baselines within each baseline bin
        #o.baseline_bins = np.array((150000,)) #single bin
        #o.baseline_bin_distribution = np.array((100,))#single bin, handy for debugging tests
        #o.NAProducts = 3 # Most antennas can be modelled as the same. [deactivated for now]
        o.tRCAL_G = 180.0
        o.tICAL_G = 10.0 # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 1.0 # Solution interval for Ionosphere
        o.NIpatches = 1 # Number of ionospheric patches to solve
        #o.Tion = 3600

    elif telescope == Telescopes.SKA1_Mid_old:
        o.Bmax = 200000  # Actually constructed max baseline, in *m*
        o.Ds = 13.5  # station "diameter" in meters, 13.5 for Meerkat antennas
        o.Na = 190 + 64  # number of antennas
        o.Nbeam = 1  # number of beams
        o.Nf_max = 256000  # maximum number of channels
        o.B_dump_ref = 200000  # m
        o.Tint_min = 0.08  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.baseline_bins = np.array((4400, 6700, 10300, 15700, 24000, 36700, 56000, 85600, 130800, 200000))  # m
        o.baseline_bin_distribution = np.array(
            (57.453, 5.235, 5.562, 5.68, 6.076, 5.835, 6.353, 5.896, 1.846, 0.064))
        o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
        #o.NAProducts = 3 # Most antennas can be modelled as the same. [deactivated for now]
        o.tRCAL_G = 180.0
        o.tICAL_G = 10.0 # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 1.0 # Solution interval for Ionosphere
        o.NIpatches = 1 # Number of ionospheric patches to solve

    elif telescope == Telescopes.SKA1_Sur_old:
        o.Bmax = 50000  # Actually constructed max baseline, in *m*
        o.Ds = 15  # station "diameter" in meters
        o.Na = 96  # number of antennas
        o.Nbeam = 36  # number of beams
        o.Nf_max = 256000  # maximum number of channels
        o.B_dump_ref = 50000  # m
        o.Tint_min = 0.3  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.baseline_bins = np.array((3800, 5500, 8000, 11500, 16600, 24000, 34600, 50000))  # m
        o.baseline_bin_distribution = np.array((48.39, 9.31, 9.413, 9.946, 10.052, 10.738, 1.958, 0.193))
        o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
        #o.NAProducts = 1 # Each antenna can be modelled as the same. [deactivated for now]
        o.tRCAL_G = 180.0
        o.tICAL_G = 10.0 # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 1.0 # Solution interval for Ionosphere
        o.NIpatches = 1 # Number of ionospheric patches to solve

    elif telescope == Telescopes.SKA2_Low:
        o.Bmax = 180000  # Actually constructed max baseline, in *m*
        o.Ds = 180  # station "diameter" in meters
        o.Na = 155  # number of antennas
        o.Nbeam = 200  # number of beams
        o.B_dump_ref = 180000  # m
        o.Nf_max = 256000  # maximum number of channels
        o.Tint_min = 0.6  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.B_dump_ref = 100000  # m
        o.baseline_bins = np.array((4400, 6700, 10300, 15700, 24000, 36700, 56000, 85600, 130800, 180000))  # m
        o.baseline_bin_distribution = np.array(
            (57.453, 5.235, 5.563, 5.68, 6.076, 5.835, 6.352, 5.896, 1.846, 0.064))
        o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
        o.NAProducts = 'all' # We must model the ionosphere for each station
        o.tRCAL_G = 180.0
        o.tICAL_G = 10.0 # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 1.0 # Solution interval for Ionosphere
        o.NIpatches = 1 # Number of ionospheric patches to solve

    elif telescope == Telescopes.SKA2_Mid:
        o.Bmax = 1800000  # Actually constructed max baseline, in *m*
        o.Ds = 15  # station "diameter" in meters
        o.Na = 155  # number of antennas
        o.Nbeam = 200  # number of beams
        o.Nf_max = 256000  # maximum number of channels
        o.B_dump_ref = 1800000  # m
        o.Tint_min = 0.008  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.baseline_bins = np.array((44000, 67000, 103000, 157000, 240000, 367000, 560000, 856000, 1308000, 1800000))
        o.baseline_bin_distribution = np.array(
            (57.453, 5.235, 5.563, 5.68, 6.076, 5.835, 6.352, 5.896, 1.846, 0.064))
        o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
        #o.NAProducts = 3 # Most antennas can be modelled as the same. [deactivated for now]
        o.tRCAL_G = 180.0
        o.tICAL_G = 10.0 # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 1.0 # Solution interval for Ionosphere
        o.NIpatches = 1 # Number of ionospheric patches to solve

    else:
        raise Exception('Unknown Telescope!')

    o.telescope = telescope
    return o


def get_telescope_from_hpso(hpso):
    """
    Returns the telescope that is associated with the provided HPSO. Not really necessary any more, as the HPSO
    definitions now contain the relevant telescope.
    @param hpso:
    @return: the telescope corresponding to this HPSO
    @raise Exception:
    """
    if hpso in HPSOs.hpsos_using_SKA1Low:
        telescope = Telescopes.SKA1_Low
    elif hpso in HPSOs.hpsos_using_SKA1Mid:
        telescope = Telescopes.SKA1_Mid
    elif hpso in HPSOs.hpsos_originally_for_SKA1Sur:
        telescope = Telescopes.SKA1_Sur_old
    else:
        raise Exception('HPSO not associated with a telescope')

    return telescope


def apply_band_parameters(o, band):
    """
    Applies the parameters that apply to the band to the parameter container object o
    @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    @param band:
    @rtype : ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.band = band
    if band == Bands.Low:
        o.telescope = Telescopes.SKA1_Low
        o.freq_min = 50e6  # in Hz
        o.freq_max = 350e6  # in Hz
    elif band == Bands.Mid1:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 350e6
        o.freq_max = 1.05e9
    elif band == Bands.Mid2:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 949.367e6
        o.freq_max = 1.7647e9
    elif band == Bands.Mid3:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 1.65e9
        o.freq_max = 3.05e9
    elif band == Bands.Mid4:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 2.80e9
        o.freq_max = 5.18e9
    elif band == Bands.Mid5A:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 4.60e9
        o.freq_max = 7.10e9
    elif band == Bands.Mid5B:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 11.3e9
        o.freq_max = 13.8e9
    elif band == Bands.Mid5C:
        print("Band = Mid5C: using 2x2.5GHz subbands from 4.6-9.6GHz for band 5")
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 4.6e9
        o.freq_max = 9.6e9
    elif band == Bands.Sur1:
        o.telescope = Telescopes.SKA1_Sur_old
        o.freq_min = 350e6
        o.freq_max = 850e6
    elif band == Bands.Sur2A:
        o.telescope = Telescopes.SKA1_Sur_old
        o.freq_min = 650e6
        o.freq_max = 1.35e9
    elif band == Bands.Sur2B:
        o.telescope = Telescopes.SKA1_Sur_old
        o.freq_min = 1.17e9
        o.freq_max = 1.67e9
    elif band == Bands.Sur3A:
        o.telescope = Telescopes.SKA1_Sur_old
        o.freq_min = 1.5e9
        o.freq_max = 2.0e9
    elif band == Bands.Sur3B:
        o.telescope = Telescopes.SKA1_Sur_old
        o.freq_min = 3.5e9
        o.freq_max = 4.0e9
    elif band == Bands.SKA2Low:
        o.telescope = Telescopes.SKA2_Low
        o.freq_min = 70e6
        o.freq_max = 350e6
    elif band == Bands.SKA2Mid:
        o.telescope = Telescopes.SKA2_Mid
        o.freq_min = 450e6
        o.freq_max = 1.05e9
    else:
        raise Exception('Unknown Band!')

    return o


def define_pipeline_products(o, pipeline, named_pipeline_products=[]):
    o.pipeline = pipeline
    o.products = {}
    for product in named_pipeline_products:
        o.products[product] = {'Rflop':0, 'Rio':0.0, 'Rinteract':0.0, 'MW_cache':0}
    return o


def apply_pipeline_parameters(o, pipeline):
    """
    Applies the parameters that apply to the pipeline to the parameter container object o
    @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    @param pipeline: Type of pipeline
    @raise Exception:
    @rtype : ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    define_pipeline_products(o, pipeline)

    if pipeline == Pipelines.Ingest:
        o.Nf_out = o.Nf_max
        o.Nselfcal = 0
        o.Nmajor = 0
        o.Nmajortotal = 0
        o.Npp = 4 # We get everything
        o.Tobs = 1. * 3600.0  # in seconds
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.08
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.ICAL:
        o.Qfov = o.Qfov_ICAL  # Field of view factor
        o.Nselfcal = 3
        o.Nmajor = 2
        o.Nminor = 10000
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = o.Ntt * o.Nf_out
        o.Npp = 4 # We get everything
        o.Tobs = 1. * 3600.0  # in seconds
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.08
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.RCAL:
        o.Qfov = 2.7  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 0
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = o.Ntt * o.Nf_out
        o.Npp = 4 # We get everything
        o.Tobs = 1. * 3600.0  # in seconds
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.08
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.DPrepA:
        o.Qfov = 1.0  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Nf_FFT_backward = o.Ntt * o.Nf_out
        o.Nf_FFT_predict = o.Ntt * o.Nf_out
        o.Npp = 2 # We only want Stokes I, V
        o.Tobs = 1. * 3600.0  # in seconds
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.08
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.DPrepA_Image:
        o.Qfov = 1.0  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = o.Ntt * o.Nf_out
        o.Npp = 2 # We only want Stokes I, V
        o.Tobs = 1. * 3600.0  # in seconds
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.08
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.DPrepB:
        o.Qfov = 1.0  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Npp = 4 # We want Stokes I, Q, U, V
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = o.Nf_out
        o.Tobs = 1. * 3600.0  # in seconds
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.08
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.034


    elif pipeline == Pipelines.DPrepC:
        o.Qfov = 1.0  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1 
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = o.Nf_max  # The same as the maximum number of channels
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = o.Ntt * o.Nf_min
        o.Tobs = 1. * 3600
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.02
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.01
        else:
            raise Exception("amp_f_max not defined for Spectral mode for the telescope %s" % o.telescope)

    elif pipeline == Pipelines.DPrepD:
        o.Qfov = 1.0  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1 
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = o.Nf_max  # The same as the maximum number of channels
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = o.Ntt * o.Nf_min
        o.Tobs = 1. * 3600
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.02
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.01

    elif pipeline == Pipelines.Fast_Img:
        #TODO: check whether Naa (A kernel size) should be smaller (or zero) for fast imaging
        #TODO: update this to cope with multiple timescales for output
        o.Qfov = 0.9  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1 
        o.Qpix = 1.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Fast_Img_channels, o.Nf_max)  # Initially this value was computed, but now capped to 500.
        o.Nf_FFT_backward = o.Nf_out
        o.Nf_FFT_predict = o.Nf_out
        o.Npp = 2 # We only want Stokes I, V
        o.Tobs = 1.0
        o.Tsnap_min = o.Tobs
        if o.telescope == Telescopes.SKA1_Low:
            o.amp_f_max = 1.02
        elif o.telescope == Telescopes.SKA1_Mid:
            o.amp_f_max = 1.02

        o.Nmm = 1 # Off diagonal terms probably not needed?

    else:
        raise Exception('Unknown pipeline: %s' % str(pipeline))

    return o



def apply_hpso_parameters(o, hpso):
    """
    Applies the parameters that apply to the supplied HPSO to the parameter container object o. Each Telescope
    serves only one specialised pipeline and one hpso
    @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    @param hpso:
    @rtype : ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.band = 'HPSO ' + str(hpso)
    o.hpso = hpso
    if  hpso == HPSOs.hpso_max_Low_c: #"Maximal" case for LOW
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepA
        o.freq_min = 50e6
        o.freq_max = 350e6
        o.Nbeam = 1  # only 1 beam here
        o.Nf_out = 500  #
        o.Tobs = 1.0 * 3600.0
        o.Nf_max = 65536
        o.Bmax = 65000  # m
        o.Texp = 6 * 3600.0  # sec
        o.Tpoint = 6 * 3600.0  # sec
    elif  hpso == HPSOs.hpso_max_Low_s: #"Maximal" case for LOW
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepC
        o.freq_min = 50e6
        o.freq_max = 350e6
        o.Nbeam = 1  # only 1 beam here
        o.Nf_out = 65536  #
        o.Tobs = 1.0 * 3600.0
        o.Nf_max = 65536
        o.Bmax = 65000  # m
        o.Texp = 6 * 3600.0  # sec
        o.Tpoint = 6 * 3600.0  # sec
    elif hpso == HPSOs.hpso_max_Mid_c:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA
        o.freq_min = 350e6
        o.freq_max = 1.05e9
        o.Nbeam = 1
        o.Nf_out = 500
        o.Tobs = 1.0 * 3600.0
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 6 * 3600.0  # sec
        o.Tpoint = 6 * 3600.0  # sec
    elif hpso == HPSOs.hpso_max_Mid_s:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepC
        o.freq_min = 350e6
        o.freq_max = 1.05e9
        o.Nbeam = 1
        o.Nf_out = 65536
        o.Tobs = 1.0 * 3600.0
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 6 * 3600.0  # sec
        o.Tpoint = 6 * 3600.0  # sec
    elif hpso == HPSOs.hpso_max_band5_Mid_c:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA
        o.freq_min = 8.5e9
        o.freq_max = 13.5e9
        o.Nbeam = 1
        o.Nf_out = 500
        o.Tobs = 1.0 * 3600.0
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 6 * 3600.0  # sec
        o.Tpoint = 6 * 3600.0  # sec
    elif hpso == HPSOs.hpso_max_band5_Mid_s:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepC
        o.freq_min = 8.5e9
        o.freq_max = 13.5e9
        o.Nbeam = 1
        o.Nf_out = 65536
        o.Tobs = 1.0 * 3600.0
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 6 * 3600.0  # sec
        o.Tpoint = 6 * 3600.0  # sec
    elif hpso == HPSOs.hpso01ICAL:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Qfov = 2.7
    elif hpso == HPSOs.hpso01DPrepA:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
    elif hpso == HPSOs.hpso01DPrepB:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Nf_out = 500  #
        o.Tobs = 6 * 3600.0
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso01DPrepC:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepC
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Nf_out = 1500  # 1500 channels in output
        o.Tobs = 6 * 3600.0
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso02AICAL:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max    = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 100 * 3600.0  # sec
        o.Qfov = 1.8
    elif hpso == HPSOs.hpso02ADprepA:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max    = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 100 * 3600.0  # sec
    elif hpso == HPSOs.hpso02ADPrepB:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max    = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Nf_out = 500  # 500 channel pseudo continuum output
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 100 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso02ADPrepC:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepC
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max    = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Nf_out = 1500  # 1500 channels in output
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 100 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso02BICAL:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
        o.Qfov = 1.8
    elif hpso == HPSOs.hpso02BDPrepA:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
    elif hpso == HPSOs.hpso02BDPrepB:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Nf_out = 500  # 500 channel pseudo continuum
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso02BDPrepC:
        o.set_param('telescope', Telescopes.SKA1_Low)
        o.pipeline = Pipelines.DPrepC
        o.freq_min = 50e6
        o.freq_max = 200e6
        o.Nbeam = 2  # using 2 beams as per HPSO request...
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536/o.Nbeam #only half the number of channels when Nbeam is doubled
        o.Nf_out = 1500  # 1500 channels in output - test to see if this is cheaper than 500cont+1500spec
        o.Bmax = 65000  # m
        o.Texp = 2500 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso13ICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.ICAL
        o.comment = 'HI, limited BW'
        o.freq_min = 790e6
        o.freq_max = 950e6
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 41000  #41k comes from assuming 3.9kHz width over 790-950MHz
        o.Bmax = 40000  # m
        o.Texp = 5000 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Qfov = 2.7
    elif hpso == HPSOs.hpso13DPrepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.DPrepA_Image
        o.comment = 'HI, limited BW'
        o.freq_min = 790e6
        o.freq_max = 950e6
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 41000  #41k comes from assuming 3.9kHz width over 790-950MHz
        #o.Nf_out = 500
        o.Bmax = 40000  # m
        o.Texp = 5000 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
    elif hpso == HPSOs.hpso13DPrepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.DPrepB
        o.comment = 'HI, limited BW'
        o.freq_min = 790e6
        o.freq_max = 950e6
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 500 #500 channel pseudo continuum
        o.Bmax = 40000  # m
        o.Texp = 5000 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Npp=2
    elif hpso == HPSOs.hpso13DPrepC:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.DPrepC
        o.comment = 'HI, limited BW'
        o.freq_min = 790e6
        o.freq_max = 950e6
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 3200
        o.Bmax = 40000  # m
        o.Texp = 5000 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Npp=2
    elif hpso == HPSOs.hpso14ICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.comment = 'HI'
        o.freq_min = 1.2e9
        o.freq_max = 1.5e9 #Increase freq range to give >1.2 ratio for continuum
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536  #
        o.Nf_out = 500
        o.Bmax = 25000  # m (25km set by experiment)
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
        o.Qfov = 1.8
    elif hpso == HPSOs.hpso14DPrepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA_Image
        o.comment = 'HI'
        o.freq_min = 1.2e9
        o.freq_max = 1.5e9 #Increase freq range to give >1.2 ratio for continuum
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536  #
        o.Bmax = 25000  # m (25km set by experiment)
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
    elif hpso == HPSOs.hpso14DPrepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.comment = 'HI'
        o.freq_min = 1.2e9
        o.freq_max = 1.5e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 100  # 300 channel pseudo continuum (small BW)
        o.Bmax = 25000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
        o.Npp=2
    elif hpso == HPSOs.hpso14DPrepC:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepC
        o.comment = 'HI'
        o.freq_min = 1.3e9
        o.freq_max = 1.4e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 5000  # Only 5,000 spectral line channels.
        o.Bmax = 25000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 10 * 3600.0  # sec
        o.Npp=2
    elif hpso == HPSOs.hpso15ICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.ICAL
        o.comment = 'HI, limited spatial resolution'
        o.freq_min = 1.30e9 # was 1.415e9 #change this to give larger frac BW for continuum accuracy
        o.freq_max = 1.56e9 # was 1.425e9 #increased to give 20% frac BW in continuum
        o.Tobs = 4.4 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 500
        o.Bmax = 15000 #13000  # m (Experiment needs 13, use 15 (round up to nearest 5km) for ICAL as part of Science roll out work)
        o.Texp = 12600 * 3600.0  # sec
        o.Tpoint = 4.4 * 3600.0  # sec
        o.Qfov = 1.8
    elif hpso == HPSOs.hpso15DPrepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.DPrepA_Image
        o.comment = 'HI, limited spatial resolution'
        o.freq_min = 1.30e9 # was 1.415e9 #change this to give larger frac BW for continuum accuracy
        o.freq_max = 1.56e9 # was 1.425e9 #increased to give 20% frac BW in continuum
        o.Tobs = 4.4 * 3600.0  # sec
        o.Nf_max = 65536
        #o.Nf_out = 500
        o.Bmax = 15000  # m (matched for experiment)
        o.Texp = 12600 * 3600.0  # sec
        o.Tpoint = 4.4 * 3600.0  # sec
    elif hpso == HPSOs.hpso15DPrepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.DPrepB
        o.comment = 'HI, limited spatial resolution'
        o.freq_min = 1.30e9 # was 1.415e9 #change this to give larger frac BW for continuum accuracy
        o.freq_max = 1.56e9 # was 1.425e9 #increased to give 20% frac BW in continuum
        o.Tobs = 4.4 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 300 # 300 channels pseudo continuum
        o.Bmax = 15000  # m (matched for experiment)
        o.Texp = 12600 * 3600.0  # sec
        o.Tpoint = 4.4 * 3600.0  # sec
        o.Npp=2
    elif hpso == HPSOs.hpso15DPrepC:
        o.set_param('telescope', Telescopes.SKA1_Mid)  #WAS SURVEY: UPDATED
        o.pipeline = Pipelines.DPrepC
        o.comment = 'HI, limited spatial resolution'
        o.freq_min = 1.415e9
        o.freq_max = 1.425e9
        o.Tobs = 4.4 * 3600.0  # sec
        o.Nf_max = 2500  # Only 2,500 spectral line channels.
        o.Bmax = 15000  # m (for experinemt)
        o.Texp = 12600 * 3600.0  # sec
        o.Tpoint = 4.4 * 3600.0  # sec
        o.Npp=2
    elif hpso == HPSOs.hpso22ICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.comment = 'Cradle of life'
        o.freq_min = 10e9
        o.freq_max = 12e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 6000 * 3600.0  # sec
        o.Tpoint = 600 * 3600.0  # sec
        o.Qfov = 2.7
    elif hpso == HPSOs.hpso22DprepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA
        o.comment = 'Cradle of life'
        o.freq_min = 10e9
        o.freq_max = 12e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 6000 * 3600.0  # sec
        o.Tpoint = 600 * 3600.0  # sec
    elif hpso == HPSOs.hpso22DprepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.comment = 'Cradle of life'
        o.freq_min = 10e9
        o.freq_max = 12e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 500  # 500 channel continuum observation - band 5.
        o.Bmax = 150000  # m
        o.Texp = 6000 * 3600.0  # sec
        o.Tpoint = 600 * 3600.0  # sec
        o.Tpoint = 600 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso27ICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 1.0e9
        o.freq_max = 1.5e9
        o.Tobs = 0.123 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 50000  # m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 0.123 * 3600.0  # sec
        o.Qfov = 1.0
    elif hpso == HPSOs.hpso27DPrepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 1.0e9
        o.freq_max = 1.5e9
        o.Tobs = 0.123 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 50000  # m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 0.123 * 3600.0  # sec
    elif hpso == HPSOs.hpso27DPrepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 1.0e9
        o.freq_max = 1.5e9
        o.Tobs = 0.123 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 500 # continuum experiment with 500 output channels
        o.Bmax = 50000  # m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 0.123 * 3600.0  # sec
        o.Npp = 4
    
    
    elif hpso == HPSOs.hpso32ICAL: #defintions for interferometry support for SUC 32 are work in progress...
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 350e6
        o.freq_max = 1050e6
        o.Tobs = 2.2 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 500  # 700 channels required in output continuum cubes
        o.Bmax = 20000  # m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 2.2 * 3600.0  # sec
        o.Qfov = 1.0
    elif hpso == HPSOs.hpso32DPrepB: #defintions for interferometry support for SUC 32 are work in progress...
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 350e6
        o.freq_max = 1050e6
        o.Tobs = 2.2 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 700  # 700 channels required in output continuum cubes
        o.Bmax = 20000  # m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 2.2 * 3600.0  # sec
        o.Npp=2
    
    
    elif hpso == HPSOs.hpso37aICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 500  # 700 channels required in output continuum cubes
        o.Bmax = 150000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 95 * 3600.0  # sec
        o.Qfov = 1.8
    elif hpso == HPSOs.hpso37aDprepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 95 * 3600.0  # sec
    elif hpso == HPSOs.hpso37aDprepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 700  # 700 channels required in output continuum cubes
        o.Bmax = 150000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 95 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso37bICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 2000 * 3600.0  # sec
        o.Qfov = 2.7
    elif hpso == HPSOs.hpso37bDprepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 2000 * 3600.0  # sec
    elif hpso == HPSOs.hpso37bDprepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 700  # 700 channels required in output continuum cubes
        o.Bmax = 150000  # m
        o.Texp = 2000 * 3600.0  # sec
        o.Tpoint = 2000 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso37cICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 95000 # m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 95 * 3600.0  # sec
        o.Qfov = 1.8
    elif hpso == HPSOs.hpso37cDPrepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 95000 # m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 95 * 3600.0  # sec
    elif hpso == HPSOs.hpso37cDPrepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 1e9
        o.freq_max = 1.7e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 700 # 700 channels in output cube
        o.Bmax = 95000 ## m
        o.Texp = 10000 * 3600.0  # sec
        o.Tpoint = 95 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso38aICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 7e9
        o.freq_max = 11e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 1000 * 3600.0  # sec
        o.Tpoint = 16.4 * 3600.0  # sec
        o.Qfov = 1.8
    elif hpso == HPSOs.hpso38aDPrepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 7e9
        o.freq_max = 11e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 1000 * 3600.0  # sec
        o.Tpoint = 16.4 * 3600.0  # sec
    elif hpso == HPSOs.hpso38aDPrepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 7e9
        o.freq_max = 11e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 1000 #
        o.Bmax = 150000  # m
        o.Texp = 1000 * 3600.0  # sec
        o.Tpoint = 16.4 * 3600.0  # sec
        o.Npp = 4
    elif hpso == HPSOs.hpso38bICAL:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.ICAL
        o.freq_min = 7e9
        o.freq_max = 11e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 1000 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Qfov = 2.7
    elif hpso == HPSOs.hpso38bDPrepA:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepA_Image
        o.freq_min = 7e9
        o.freq_max = 11e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Bmax = 150000  # m
        o.Texp = 1000 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
    elif hpso == HPSOs.hpso38bDPrepB:
        o.set_param('telescope', Telescopes.SKA1_Mid)
        o.pipeline = Pipelines.DPrepB
        o.freq_min = 7e9
        o.freq_max = 11e9
        o.Tobs = 6 * 3600.0  # sec
        o.Nf_max = 65536
        o.Nf_out = 1000 #
        o.Bmax = 150000  # m
        o.Texp = 1000 * 3600.0  # sec
        o.Tpoint = 1000 * 3600.0  # sec
        o.Npp = 4
    else:
        raise Exception('Unknown HPSO %s!' % hpso)

    return o
