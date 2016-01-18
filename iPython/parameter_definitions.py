"""
This Python file contains several classes that enumerates and defines the parameters of the telescopes, bands, modes,
etc. Several methods are supplied by which values can be found by lookup as well
(e.g. finding the telescope that is associated with a given mode)
"""

import numpy as np
from sympy import symbols

class ParameterContainer:
    """
    The ParameterContainer class is used throughout the Python implementation to store parameters and pass them
    around. It is basically an empty class to which fields can be added, read from or overwritten
    """
    def __init__(self):
        pass

    def set_param(self, param_name, value, prevent_overwrite=True):
        """
        Provides a method for setting a parameter. By default first checks that the value has not already been defined.
        Useful for preventing situations where values may inadvertently be overwritten.
        @param param_name: The name of the parameter/field that needs to be assigned - provided as text
        @param value: the value. Need not be text.
        @param prevent_overwrite: Disallows this value to be overwritten once defined. Default = True.
        @return: Nothing
        """
        assert isinstance(param_name, str)
        try:
            if prevent_overwrite and hasattr(self, param_name):
                if eval('self.%s == value' % param_name):
                    print 'Inefficiency Warning: reassigning already-defined parameter "%s" with an identical value.'
                else:
                    assert eval('self.%s == None' % param_name)
        except AssertionError:
            raise AssertionError("The parameter %s has already been defined and may not be overwritten." % param_name)

        exec('self.%s = value' % param_name)  # Write the value

class Constants:
    """
    A new class that takes over the roles of sympy.physics.units and astropy.const, because it is simpler this way
    """
    kilo = 1000
    mega = 1000000
    giga = 1000000000
    tera = 1000000000000
    peta = 1000000000000000


class Telescopes:
    """
    Enumerate the possible telescopes to choose from (used in the ParameterDefinitions class)
    """
    # The originally planned SKA1 telescopes
    SKA1_Low_old = 'SKA1_Low_old'
    SKA1_Mid_old = 'SKA1_Mid_old'
    SKA1_Sur_old = 'SKA1_Survey_old'
    # The rebaselined SKA1 telescopes
    SKA1_Low = 'SKA1_Low_rebaselined'
    SKA1_Mid = 'SKA1_Mid_rebaselined'
    # Proposed SKA2 telescopes
    SKA2_Low = 'SKA2_Low'
    SKA2_Mid = 'SKA2_Mid'


class Bands:
    """
    Enumerate all possible bands (used in the ParameterDefinitions class)
    """
    Low = 'Low'
    Mid1 = 'Mid1'
    Mid2 = 'Mid2'
    Mid3 = 'Mid3'
    Mid4 = 'Mid4'
    Mid5A = 'Mid5A'
    Mid5B = 'Mid5B'
    Sur1 = 'Sur1'
    Sur2A = 'Sur2A'
    Sur2B = 'Sur2B'
    Sur3A = 'Sur3A'
    Sur3B = 'Sur3B'
    SKA2Low = 'LOWSKA2'
    SKA2Mid = 'MIDSKA2'

    # group the bands defined above into logically coherent sets
    low_bands = {Low}
    mid_bands = {Mid1, Mid2, Mid3, Mid4, Mid5A, Mid5B}
    survey_bands = {Sur1, Sur2A, Sur2B, Sur3A, Sur3B}
    low_bands_ska2 = {SKA2Low}
    mid_bands_ska2 = {SKA2Mid}


class ImagingModes:
    """
    Enumerate the possible imaging modes (used in the ParameterDefinitions class)
    """
    Continuum = 'Continuum'
    Spectral = 'Spectral'  # Spectral only. Spectral mode will usually follow Continuum mode. See ContAntSpectral.
    FastImg = 'Fast Imaging'
    ContAndSpectral = 'Sequential (Cont+Spec)'  # Continuum and Spectral modes run sequentially, as in some HPSOs
    All = 'All, Summed (Cont+Spec+FastImg)'
    pure_modes = (Continuum, Spectral, FastImg)

class HPSOs:
    """
    Enumerate the High Priority Science Objectives (used in the ParameterDefinitions class)
    """
    hpso01 = '01'
    hpso02A = '02A'
    hpso02B = '02B'
    hpso03A = '03A'
    hpso03B = '03B'
    hpso04A = '04A'
    hpso04B = '04B'
    hpso05A = '05A'
    hpso05B = '05B'
    hpso13 = '13'
    hpso13c = '13c'  # Continuum component of HPSO 13
    hpso13s = '13s'  # Spectral component of HPSO 13
    hpso14 = '14'
    hpso14c = '14c'  # Continuum component of HPSO 14
    hpso14s = '14s'  # Spectral  component of HPSO 14
    hpso14sfull = '14sfull'
    hpso15 = '15'
    hpso15c = '15c'  # Continuum component of HPSO 15
    hpso15s = '15s'  # Spectral  component of HPSO 15
    hpso19 = '19'
    hpso22 = '22'
    hpso27 = '27'
    hpso33 = '33'
    hpso35 = '35'
    hpso37a = '37a'
    hpso37b = '37b'
    hpso37c = '37c'
    hpso38a = '38a'
    hpso38b = '38b'

    # group the HPSOs according to which telescope they refer to
    hpsos_using_SKA1Low = {hpso01, hpso02A, hpso02B, hpso03A, hpso03B}
    hpsos_using_SKA1Mid = {hpso04A, hpso04B, hpso05A, hpso05B, hpso14, hpso19, hpso22, hpso37a, hpso37b, hpso38a,
                           hpso38b, hpso14c, hpso14s, hpso14sfull}
    hpsos_originally_for_SKA1Sur = {hpso13, hpso15, hpso27, hpso33, hpso35, hpso37c, hpso13c, hpso13s, hpso15c, hpso15s}
    # Because we are no longer building Survey, assume that the HPSOs intended for Survey will run on Mid?
    hpsos_using_SKA1Mid = hpsos_using_SKA1Mid | hpsos_originally_for_SKA1Sur

class ParameterDefinitions:
    """
    This class contains several methods for defining parameters. These include Telecope parameters, as well as
    physical constants. Generally, the output of the methods of this class is a ParameterContainer object
    (usually referred to by the variable o in the methods below) that has parameters as fields.
    """

    @staticmethod
    def define_symbolic_variables(o):
        """
        This method defines the *symbolic* variables that we will use during computations
        and that need to be kept symbolic during evaluation of formulae. One reason to do this would be to allow
        the output formula to be optimized by varying this variable (such as with Tsnap and Nfacet)
        @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
        @rtype : ParameterContainer
        """
        assert isinstance(o, ParameterContainer)
        o.Tsnap = symbols("T_snap", positive=True)  # Snapshot timescale implemented
        o.Nfacet = symbols("N_facet", integer=True, positive=True)  # Number of facets

        # The following two parameters are used for baseline-dependent calculations
        o.Bmax_bin = symbols("Bmax\,bin", positive=True)  # The maximum baseline corresponding to a given bin
        o.binfrac = symbols("f_bin", positive=True)  # Fraction of total baselines in a given bin - value in (0,1)
        return o

    @staticmethod
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
        o.Mvis = 12.0  # Memory size of a single visibility datum in bytes. Back to 12; likely to change in future
        o.Naa = 10  # Changed to 10, after PDR submission
        o.Nmm = 4  # Mueller matrix Factor: 1 is for diagonal terms only, 4 includes off-diagonal terms too.
        o.Npp = 4  # Number of polarization products
        o.Nw = 2  # Bytes per value
        # o.Qbw = 4.3 #changed from 1 to give 0.34 uv cells as the bw smearing limit. Should be investigated and linked to depend on amp_f_max, or grid_cell_error
        o.Qfcv = 1.0  #changed to 1 to disable but retain ability to see affect in parameter sweep.
        o.Qgcf = 8.0
        o.Qkernel = 10.0  #  epsilon_f/ o.Qkernel is the fraction of a uv cell we allow frequence smearing at edge of convoluion kernel to - i.e error on u,v, position one kernel-radius from gridding point.
        #o.grid_cell_error = 0.34 #found from tump time as given by SKAO at largest FoV (continuum).
        o.Qw = 1.0
        o.Tion = 10.0  #This was previously set to 60s (for PDR) May wish to use much smaller value.
        o.Tsnap_min = 1.0
        o.minimum_channels = 500  #minimum number of channels to still enable distributed computing, and to reconstruct Taylor terms
        o.facet_overlap_frac = 0.2 #fraction of overlap (linear) in adjacent facets.
        o.max_subband_freq_ratio = 1.35 #maximum frequency ratio supported within each subband. 1.35 comes from Jeff Wagg SKAO ("30% fractional bandwidth in subbands").
        return o

    @staticmethod
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
            o.Bmax = 80000  # Actually constructed max baseline in *m*
            o.Ds = 35  # station "diameter" in metres
            o.Na = 512  # number of antennas
            o.Nbeam = 1  # number of beams
            o.Nf_max = 65536  # maximum number of channels
            o.B_dump_ref = 80000  # m
            o.Tdump_ref = 0.9  # Correlator dump time in reference design in *seconds*
            o.baseline_bins = np.array((4900, 7100, 10400, 15100, 22100, 32200, 47000, 80000))  # m
            o.nr_baselines = 10180233
            o.baseline_bin_distribution = np.array(
                (52.42399198, 7.91161595, 5.91534571, 9.15027832, 7.39594812, 10.56871804, 6.09159108, 0.54251081))
        #            o.amp_f_max = 1.08  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.

        elif telescope == Telescopes.SKA1_Low_old:
            o.Bmax = 100000  # Actually constructed max baseline in *m*
            o.Ds = 35  # station "diameter" in metres
            o.Na = 1024  # number of antennas
            o.Nbeam = 1  # number of beams
            o.Nf_max = 256000  # maximum number of channels
            o.Tdump_ref = 0.6  # Correlator dump time in reference design in *sec*
            o.B_dump_ref = 100000  # m
            o.baseline_bins = np.array((4900, 7100, 10400, 15100, 22100, 32200, 47000, 68500, 100000))  # m
            o.nr_baselines = 10192608
            o.baseline_bin_distribution = np.array((49.361, 7.187, 7.819, 5.758, 10.503, 9.213, 8.053, 1.985, 0.121))
            o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.

        elif telescope == Telescopes.SKA1_Mid:
            o.Bmax = 150000  # Actually constructed max baseline in *m*
            o.Ds = 15  # station "diameter" in metres
            o.Na = 133 + 64  # number of antennas (expressed as the sum between new and Meerkat antennas)
            o.Nbeam = 1  # number of beams
            o.Nf_max = 65536  # maximum number of channels
            o.Tdump_ref = 0.14  # Correlator dump time in reference design in *sec*
            o.B_dump_ref = 150000  # m
            o.nr_baselines = 1165860
            # Rosie's conservative, ultra simple numbers (see Absolute_Baseline_length_distribution.ipynb)
            o.baseline_bins = np.array((5000.,7500.,10000.,15000.,25000.,35000.,55000.,75000.,90000.,110000.,130000.,150000)) #"sensible" baseline bins
            o.baseline_bin_distribution = np.array(( 6.14890420e+01,   5.06191389e+00 ,  2.83923113e+00 ,  5.08781928e+00, 7.13952645e+00,   3.75628206e+00,   5.73545412e+00,   5.48158127e+00, 1.73566136e+00,   1.51805606e+00,   1.08802653e-01 ,  4.66297083e-02))#July2-15 post-rebaselining, from Rebaselined_15July2015_SKA-SA.wgs84.197x4.txt % of baselines within each baseline bin
            #o.baseline_bins = np.array((150000,)) #single bin
            #o.baseline_bin_distribution = np.array((100,))#single bin, handy for debugging tests


        elif telescope == Telescopes.SKA1_Mid_old:
            o.Bmax = 200000  # Actually constructed max baseline, in *m*
            o.Ds = 15  # station "diameter" in meters
            o.Na = 190 + 64  # number of antennas
            o.Nbeam = 1  # number of beams
            o.Nf_max = 256000  # maximum number of channels
            o.B_dump_ref = 200000  # m
            o.Tdump_ref = 0.08  # Correlator dump time in reference design
            o.baseline_bins = np.array((4400, 6700, 10300, 15700, 24000, 36700, 56000, 85600, 130800, 200000))  # m
            o.nr_baselines = 1165860
            o.baseline_bin_distribution = np.array(
                (57.453, 5.235, 5.562, 5.68, 6.076, 5.835, 6.353, 5.896, 1.846, 0.064))
            o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.

        elif telescope == Telescopes.SKA1_Sur_old:
            o.Bmax = 50000  # Actually constructed max baseline, in *m*
            o.Ds = 15  # station "diameter" in meters
            o.Na = 96  # number of antennas
            o.Nbeam = 36  # number of beams
            o.Nf_max = 256000  # maximum number of channels
            o.B_dump_ref = 50000  # m
            o.Tdump_ref = 0.3  # Correlator dump time in reference design
            o.baseline_bins = np.array((3800, 5500, 8000, 11500, 16600, 24000, 34600, 50000))  # m
            o.nr_baselines = 167616
            o.baseline_bin_distribution = np.array((48.39, 9.31, 9.413, 9.946, 10.052, 10.738, 1.958, 0.193))
            o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.

        elif telescope == Telescopes.SKA2_Low:
            o.Bmax = 180000  # Actually constructed max baseline, in *m*
            o.Ds = 180  # station "diameter" in meters
            o.Na = 155  # number of antennas
            o.Nbeam = 200  # number of beams
            o.B_dump_ref = 180000  # m
            o.Nf_max = 256000  # maximum number of channels
            o.Tdump_ref = 0.6  # Correlator dump time in reference design
            o.B_dump_ref = 100000  # m
            o.baseline_bins = np.array((4400, 6700, 10300, 15700, 24000, 36700, 56000, 85600, 130800, 180000))  # m
            o.nr_baselines = 1165860
            o.baseline_bin_distribution = np.array(
                (57.453, 5.235, 5.563, 5.68, 6.076, 5.835, 6.352, 5.896, 1.846, 0.064))
            o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.

        elif telescope == Telescopes.SKA2_Mid:
            o.Bmax = 1800000  # Actually constructed max baseline, in *m*
            o.Ds = 15  # station "diameter" in meters
            o.Na = 155  # number of antennas
            o.Nbeam = 200  # number of beams
            o.Nf_max = 256000  # maximum number of channels
            o.B_dump_ref = 1800000  # m
            o.Tdump_ref = 0.008  # Correlator dump time in reference design
            o.baseline_bins = np.array((44000, 67000, 103000, 157000, 240000, 367000, 560000, 856000, 1308000, 1800000))
            o.nr_baselines = 1165860
            o.baseline_bin_distribution = np.array(
                (57.453, 5.235, 5.563, 5.68, 6.076, 5.835, 6.352, 5.896, 1.846, 0.064))
            o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.

        else:
            raise Exception('Unknown Telescope!')

        return o

    @staticmethod
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

    @staticmethod
    def apply_band_parameters(o, band):
        """
        Applies the parameters that apply to the band to the parameter container object o
        @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
        @param band:
        @rtype : ParameterContainer
        """
        assert isinstance(o, ParameterContainer)
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

    @staticmethod
    def apply_imaging_mode_parameters(o, mode):
        """
        Applies the parameters that apply to the imaging mode to the parameter container object o
        @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
        @param mode:
        @raise Exception:
        @rtype : ParameterContainer
        """
        assert isinstance(o, ParameterContainer)

        if mode == ImagingModes.Continuum:
            o.Qfov = 1.8  # Field of view factor
            o.Nmajor = 10  # Number of major CLEAN cycles to be done
            o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
            o.Nf_out = min(500, o.Nf_max)
            o.Tobs = 6 * 3600  # in seconds
            if o.telescope == Telescopes.SKA1_Low:
                o.amp_f_max = 1.08
            elif o.telescope == Telescopes.SKA1_Mid:
                o.amp_f_max = 1.034


        elif mode == ImagingModes.Spectral:
            o.Qfov = 1.0  # Field of view factor
            o.Nmajor = 1.5  # Number of major CLEAN cycles to be done: updated to 1.5 as post-PDR fix.
            o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
            o.Nf_out = o.Nf_max  # The same as the maximum number of channels
            o.Tobs = 6 * 3600
            if o.telescope == Telescopes.SKA1_Low:
                o.amp_f_max = 1.02
            elif o.telescope == Telescopes.SKA1_Mid:
                o.amp_f_max = 1.01

        elif mode == ImagingModes.FastImg:
            o.Qfov = 0.9  # Field of view factor
            o.Nmajor = 1  # Number of major CLEAN cycles to be done
            o.Qpix = 1.5  # Quality factor of synthesised beam oversampling
            o.Nf_out = min(500, o.Nf_max)  # Initially this value was computed, but now capped to 500.
            o.Tobs = 0.9  # Used to be equal to Tdump but after talking to Rosie set this to 1.2 sec
            if o.telescope == Telescopes.SKA1_Low:
                o.amp_f_max = 1.02
            elif o.telescope == Telescopes.SKA1_Mid:
                o.amp_f_max = 1.02

        elif mode == ImagingModes.ContAndSpectral:
            raise Exception("'apply_imaging_mode_parameters' needs to compute Continuum and Spectral modes separately")

        else:
            raise Exception('Unknown mode: %s!' % str(mode))

        return o

    @staticmethod
    def apply_hpso_parameters(o, hpso):
        """
        Applies the parameters that apply to the supplied HPSO to the parameter container object o
        @param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
        @param hpso:
        @rtype : ParameterContainer
        """
        assert isinstance(o, ParameterContainer)
        if hpso == HPSOs.hpso01:
            o.telescope = Telescopes.SKA1_Low
            o.mode = ImagingModes.Continuum
            o.freq_min = 50e6
            o.freq_max = 200e6
            o.Nbeam = 2  # using 2 beams as per HPSO request...
            o.Nf_out = 1500  # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Tobs = 6 * 3600
            o.Nf_max = 1500
            o.Bmax = 80000  # m
            o.Texp = 2500 * 3600  # sec
            o.Tpoint = 1000 * 3600  # sec
        elif hpso == HPSOs.hpso02A:
            o.telescope = Telescopes.SKA1_Low
            o.mode = ImagingModes.Continuum
            o.freq_min = 50e6
            o.freq_max = 200e6
            o.Nbeam = 2  # using 2 beams as per HPSO request...
            o.Tobs = 6 * 3600  # sec
            o.Nf_max    = 256000
            o.Nf_out = 1500  # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax = 80000  # m
            o.Texp = 2500 * 3600  # sec
            o.Tpoint = 100 * 3600  # sec
        elif hpso == HPSOs.hpso02B:
            o.telescope = Telescopes.SKA1_Low
            o.mode = ImagingModes.Continuum
            o.freq_min = 50e6
            o.freq_max = 200e6
            o.Nbeam = 2  # using 2 beams as per HPSO request...
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 256000
            o.Nf_out = 1500  # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax = 80000  # m
            o.Texp = 2500 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso03A:
            o.telescope = Telescopes.SKA1_Low
            o.mode = ImagingModes.FastImg
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate'
            o.freq_min = 150e6
            o.freq_max = 350e6
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Bmax = 100000  # m
            o.Texp = 12800 * 3600  # sec
            o.Tpoint = 0.17 * 3600  # sec
            o.Nmajor = 10
        elif hpso == HPSOs.hpso03B:
            o.telescope = Telescopes.SKA1_Low
            o.mode = ImagingModes.FastImg
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate'
            o.freq_min = 150e6
            o.freq_max = 350e6
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Bmax = 100000  # m
            o.Texp = 4300 * 3600  # sec
            o.Tpoint = 0.17 * 3600  # sec
            o.Nmajor = 10
        elif hpso == HPSOs.hpso04A:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.FastImg
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min = 650e6
            o.freq_max = 950e6
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Bmax = 10000  # m
            o.Texp = 800 * 3600  # sec  # in *sec*
            o.Tpoint = 10 / 60.0 * 3600  # sec
            o.Nmajor = 10
        elif hpso == HPSOs.hpso04B:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.FastImg
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min = 1.25e9
            o.freq_max = 1.55e9
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Bmax = 10000  # m
            o.Texp = 800 * 3600  # sec
            o.Tpoint = 10 / 60.0 * 3600  # sec
            o.Nmajor = 10
        elif hpso == HPSOs.hpso05A:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.FastImg
            o.comment = 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.'
            o.freq_min = 0.95e9
            o.freq_max = 1.76e9
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Bmax = 15000  # m
            o.Texp = 1600 * 3600  # sec
            o.Tpoint = (10 / 60.0) * 3600  # sec
            o.Nmajor = 10
        elif hpso == HPSOs.hpso05B:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.FastImg
            o.comment = 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.'
            o.freq_min = 1.65e9
            o.freq_max = 3.05e9
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Bmax = 15000  # m
            o.Texp = 1600 * 3600  # sec
            o.Tpoint = (10 / 60.0) * 3600  # sec
            o.Nmajor = 10
        elif hpso == HPSOs.hpso13:
            o.telescope = Telescopes.SKA1_Mid #WAS SURVEY: UPDATED
            o.mode = ImagingModes.ContAndSpectral
            o.comment = 'HI, limited BW'
            o.freq_min = 790e6
            o.freq_max = 950e6
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 3200  # Assume 500 in continuum as well - defualt.
            o.Bmax = 40000  # m
            o.Texp = 5000 * 3600  # sec
            o.Tpoint = 1000 * 3600  # sec
        elif hpso == HPSOs.hpso13c:
            o.telescope = Telescopes.SKA1_Mid #WAS SURVEY: UPDATED
            o.mode = ImagingModes.Continuum
            o.comment = 'HI, limited BW'
            o.freq_min = 790e6
            o.freq_max = 950e6
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 3200  # Assume 500 in continuum as well - defualt.
            o.Bmax = 40000  # m
            o.Texp = 5000 * 3600  # sec
            o.Tpoint = 1000 * 3600  # sec
        elif hpso == HPSOs.hpso13s:
            o.telescope = Telescopes.SKA1_Mid #WAS SURVEY: UPDATED
            o.mode = ImagingModes.Spectral
            o.comment = 'HI, limited BW'
            o.freq_min = 790e6
            o.freq_max = 950e6
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 3200  # Assume 500 in continuum as well - defualt.
            o.Bmax = 40000  # m
            o.Texp = 5000 * 3600  # sec
            o.Tpoint = 1000 * 3600  # sec
        elif hpso == HPSOs.hpso14:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.ContAndSpectral
            o.comment = 'HI'
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 5000  # Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 150000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso14c:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.comment = 'HI'
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 5000  # Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 150000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso14s:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Spectral
            o.comment = 'HI'
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 5000  # Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 150000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso15:
            o.telescope = Telescopes.SKA1_Mid #WAS SURVEY: UPDATED
            o.mode = ImagingModes.ContAndSpectral
            o.comment = 'HI, limited spatial resolution'
            o.freq_min = 1.415e9
            o.freq_max = 1.425e9
            o.Tobs = 4.4 * 3600  # sec
            o.Nf_max = 2500  # Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 13000  # m
            o.Texp = 12600 * 3600  # sec
            o.Tpoint = 4.4 * 3600  # sec
        elif hpso == HPSOs.hpso15c:
            o.telescope = Telescopes.SKA1_Mid #WAS SURVEY: UPDATED
            o.mode = ImagingModes.Continuum
            o.comment = 'HI, limited spatial resolution'
            o.freq_min = 1.415e9 #change this to give larger frac BW for continuum accuracy
            o.freq_max = 1.425e9
            o.Tobs = 4.4 * 3600  # sec
            o.Nf_max = 2500  # Only 2,500 spectral line channels. Assume 500 - default - for continuum as well. Probably want continuum across whole band, or at least 35% frac BW
            o.Bmax = 13000  # m
            o.Texp = 12600 * 3600  # sec
            o.Tpoint = 4.4 * 3600  # sec
        elif hpso == HPSOs.hpso15s:
            o.telescope = Telescopes.SKA1_Mid #WAS SURVEY: UPDATED
            o.mode = ImagingModes.Spectral
            o.comment = 'HI, limited spatial resolution'
            o.freq_min = 1.415e9
            o.freq_max = 1.425e9
            o.Tobs = 4.4 * 3600  # sec
            o.Nf_max = 2500  # Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 13000  # m
            o.Texp = 12600 * 3600  # sec
            o.Tpoint = 4.4 * 3600  # sec
        elif hpso == HPSOs.hpso19:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.FastImg
            o.comment = 'Transients. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min = 650e6
            o.freq_max = 950e6
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Bmax = 10000  # m
            o.Texp = 10000 * 3600  # sec
            o.Tpoint = (10 / 60.0) * 3600  # sec
            o.Nmajor = 10
        elif hpso == HPSOs.hpso22:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.comment = 'Cradle of life'
            o.freq_min = 10e9
            o.freq_max = 12e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 4000
            o.Nf_out = 4000  # 4000 channel continuum observation - band 5.
            o.Bmax = 200000  # m
            o.Texp = 6000 * 3600  # sec
            o.Tpoint = 600 * 3600  # sec
        elif hpso == HPSOs.hpso27:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.freq_min = 1.0e9
            o.freq_max = 1.5e9
            o.Tobs = 0.123 * 3600  # sec
            o.Nf_max = 256000
            o.Nf_out = 500  # continuum experiment with 500 output channels
            o.Bmax = 50000  # m
            o.Texp = 10000 * 3600  # sec
            o.Tpoint = 0.123 * 3600  # sec
        elif hpso == HPSOs.hpso33:
            o.telescope = Telescopes.SKA1_Mid #WAS SURVEY: UPDATED
            o.mode = ImagingModes.Continuum
            o.freq_min = 1.0e9
            o.freq_max = 1.5e9
            o.Tobs = 0.123 * 3600  # sec
            o.Nf_out = 500  # continuum experiment with 500 output channels
            o.Bmax = 50000  # m
            o.Texp = 10000 * 3600  # sec
            o.Tpoint = 0.123 * 3600  # sec
        elif hpso == HPSOs.hpso37a:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.freq_min = 1e9
            o.freq_max = 1.7e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 700  # 700 channels required in output continuum cubes
            o.Bmax = 150000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 95 * 3600  # sec
        elif hpso == HPSOs.hpso37b:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.freq_min = 1e9
            o.freq_max = 1.7e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 700  # 700 channels required in output continuum cubes
            o.Bmax = 200000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 2000 * 3600  # sec
        elif hpso == HPSOs.hpso37c:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.freq_min = 1e9
            o.freq_max = 1.5e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 500  # 500 channels in output cube
            o.Bmax = 93000  # m
            o.Texp = 10000 * 3600  # sec
            o.Tpoint = 95 * 3600  # sec
        elif hpso == HPSOs.hpso38a:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.freq_min = 7e9
            o.freq_max = 11e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 1000
            o.Bmax = 150000  # m
            o.Texp = 1000 * 3600  # sec
            o.Tpoint = 16.4 * 3600  # sec
        elif hpso == HPSOs.hpso38b:
            o.telescope = Telescopes.SKA1_Mid
            o.mode = ImagingModes.Continuum
            o.freq_min = 7e9
            o.freq_max = 11e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 1000
            o.Bmax = 150000  # m
            o.Texp = 1000 * 3600  # sec
            o.Tpoint = 1000 * 3600  # sec
        else:
            raise Exception('Unknown HPSO!')

        return o