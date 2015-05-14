"""
This Python file contains several classes that enumerates and defines the parameters of the telescopes, bands, modes,
etc. Several methods are supplied by which values can be found by lookup as well
(e.g. finding the telescope that is associated with a given mode)
"""

from sympy import symbols
import numpy as np

# A new class that takes over the roles of sympy.physics.units and astropy.const, because it is simpler this way
class Constants:
    kilo = 1000
    mega = 1000000
    giga = 1000000000
    tera = 1000000000000
    peta = 1000000000000000

# Enumerate the possible telescopes to choose from (used in the ParameterDefinitions class)
class Telescopes:
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

    # Supply string representations that are nice to read for humans
    telescopes_pretty_print = {SKA1_Low_old: 'SKA1-Low',
                               SKA1_Mid_old: 'SKA1-Mid (Band 1)',
                               SKA1_Sur_old: 'SKA1-Survey (Band 1)'
                               }

# Enumerate all possible bands (used in the ParameterDefinitions class)
class Bands:
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

    low_bands = {Low}
    mid_bands = {Mid1, Mid2, Mid3, Mid4, Mid5A, Mid5B}
    survey_bands = {Sur1, Sur2A, Sur2B, Sur3A, Sur3B}
    low_ska2_bands = {SKA2Low}
    mid_ska2_bands = {SKA2Mid}


# Enumerate the possible imaging modes (used in the ParameterDefinitions class)
class ImagingModes:
    Continuum = 'Continuum'
    Spectral = 'Spectral'
    SlowTrans = 'SlowTrans'
    ContAndSpectral = 'CS'  # A special case for some of the HPSOs where continuum and spectral are done sequentially
    CSS = 'Summed (Cont+Spec+Slow)'

    # Supply string representations that are nice to read for humans
    modes_pretty_print = {Continuum: 'Continuum',
                          Spectral: 'Spectral',
                          SlowTrans: 'SlowTrans',
                          CSS: 'All Modes'
                          }

# Enumerate the High Priority Science Objectives (used in the ParameterDefinitions class)
class HPSOs:
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
    hpso13c = '13c'
    hpso13s = '13s'
    hpso14 = '14'
    hpso14c = '14c'
    hpso14s = '14s'
    hpso14sfull = '14sfull'
    hpso15 = '15'
    hpso15c = '15c'
    hpso15s = '15s'
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

    hpsos_using_SKA1Low = {hpso01, hpso02A, hpso02B, hpso03A, hpso03B}
    hpsos_using_SKA1Mid = {hpso04A, hpso04B, hpso05A, hpso05B, hpso14, hpso19, hpso22, hpso37a, hpso37b, hpso38a,
                           hpso38b, hpso14c, hpso14s, hpso14sfull}
    hpsos_using_SKA1Sur = {hpso13, hpso15, hpso27, hpso33, hpso35, hpso37c, hpso13c, hpso13s, hpso15c, hpso15s}


class ParameterDefinitions:
    @staticmethod
    def define_symbolic_variables(o):
        """
        This method defines the *symbolic* variables that we will use during computations
        and that need to be kept symbolic during evaluation of formulae. One reason to do this would be to allow
        the output formula to be optimized by varying this variable (such as with Tsnap and Nfacet)
        @param o:
        """
        o.Tsnap = symbols("T_snap", positive=True)  # Snapshot timescale implemented
        o.Bmax_bin = symbols("Bmax\,bin", positive=True)
        o.binfrac = symbols("f_bin",
                            positive=True)  # The fraction of baselines that fall in a given bin (used for baseline-dependent calculations)
        o.Nfacet = symbols("N_facet", integer=True, positive=True)

    @staticmethod
    def apply_global_parameters(o):
        """
        Applies the global parameters to the parameter container object o
        @param o:
        """
        o.c = 299792458          # The speed of light, in m/s (from sympy.physics.units.c)
        o.Omega_E = 7.292115e-5  # Rotation relative to the fixed stars in radians/second
        o.R_Earth = 6378136      # Radius if the Earth in meters (equal to astropy.const.R_earth.value)
        o.epsilon_w = 0.01
        o.Mvis = 12.0  # back to 12. Likely to change in future
        o.Naa = 10     # Changed to 10, after PDR submission
        o.Nmm = 4
        o.Npp = 4
        o.Nw = 2  # Bytes per value
        o.Qbw = 1.0
        o.Qfcv = 10.0
        o.Qgcf = 8.0
        o.Qw = 1.0
        o.Tion = 60.0
        o.Tsnap_min = 1.0
        o.amp_f_max = 1.01  # Added by Rosie Bolton

    @staticmethod
    def apply_telescope_parameters(o, telescope):
        """
        Applies the parameters that apply to the supplied telescope to the parameter container object o
        @param o:
        @param telescope:
        """
        if telescope == Telescopes.SKA1_Low:
            o.Bmax = 80000  # Actually constructed max baseline in *m*
            o.Ds = 35    # station "diameter" in meters
            o.Na = 512  # number of antennas
            o.Nbeam = 1  # number of beams
            o.Nf_max = 65536  # maximum number of channels
            o.B_dump_ref = 80000  # m
            o.Tdump_ref = 0.6  # Correlator dump time in reference design in *seconds*
            o.baseline_bins = np.array((4900, 7100, 10400, 15100, 22100, 32200, 47000, 80000))  # m
            o.nr_baselines = 10180233
            o.baseline_bin_distribution=np.array((52.42399198, 7.91161595, 5.91534571, 9.15027832, 7.39594812, 10.56871804, 6.09159108, 0.54251081))
        elif telescope == Telescopes.SKA1_Low_old:
            o.Bmax = 100000  # Actually constructed max baseline in *m*
            o.Ds = 35  # station "diameter" in meters
            o.Na = 1024  # number of antennas
            o.Nbeam = 1  # number of beams
            o.Nf_max = 256000  # maximum number of channels
            o.Tdump_ref = 0.6  # Correlator dump time in reference design in *sec*
            o.B_dump_ref = 100000  # m
            o.baseline_bins = np.array((4900, 7100, 10400, 15100, 22100, 32200, 47000, 68500, 100000))  # m
            o.nr_baselines = 10192608
            o.baseline_bin_distribution = np.array((49.361, 7.187, 7.819, 5.758, 10.503, 9.213, 8.053, 1.985, 0.121))
        elif telescope == Telescopes.SKA1_Mid:
            o.Bmax = 150000  # Actually constructed max baseline in *m*
            o.Ds = 15  # station "diameter" in meters
            o.Na = 133 + 64  # number of antennas (expressed as the sum between new and Meerkat antennas)
            o.Nbeam = 1  # number of beams
            o.Nf_max = 65536  # maximum number of channels
            o.Tdump_ref = 0.08   # Correlator dump time in reference design in *sec*
            o.B_dump_ref = 200000  # m
            o.baseline_bins = np.array((4400, 6700, 10300, 15700, 24000, 36700, 56000, 85600, 130800, 150000))  # m
            o.nr_baselines = 1165860
            o.baseline_bin_distribution = np.array((57.453, 5.235, 5.562, 5.68, 6.076, 5.835, 6.353, 5.896, 1.846, 0.064)) #Original distribution
            # o.baseline_bin_distribution = np.array((56.78620346,   5.25152534,   5.6811107,    5.72469182,   6.21031005, 5.64375545,   6.21653592,   6.00485618 ,  2.42186527,   0.05914581))
            # Rosie's conservative, ultra simple numbers (see Absolute_Baseline_length_distribution.ipynb)
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
        elif telescope == Telescopes.SKA1_Sur_old:
            o.Bmax = 50000 # Actually constructed max baseline, in *m*
            o.Ds = 15  # station "diameter" in meters
            o.Na = 96  # number of antennas
            o.Nbeam = 36  # number of beams
            o.Nf_max = 256000  # maximum number of channels
            o.B_dump_ref = 50000  # m
            o.Tdump_ref = 0.3  # Correlator dump time in reference design
            o.baseline_bins = np.array((3800, 5500, 8000, 11500, 16600, 24000, 34600, 50000))  # m
            o.nr_baselines = 167616
            o.baseline_bin_distribution = np.array((48.39, 9.31, 9.413, 9.946, 10.052, 10.738, 1.958, 0.193))
        elif telescope == Telescopes.SKA2_Low:
            o.Bmax = 180000 # Actually constructed max baseline, in *m*
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
        elif telescope == Telescopes.SKA2_Mid:
            o.Bmax = 1800000 # Actually constructed max baseline, in *m*
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
        else:
            raise Exception('Unknown Telescope!')

    @staticmethod
    def get_telescope_from_hpso(hpso):
        """
        Returns the telescope that is associated with the provided HPSO
        @param hpso:
        @return: the telescope corresponding to this HPSO
        @raise Exception:
        """
        telescope = None
        if hpso in HPSOs.hpsos_using_SKA1Low:
            telescope = Telescopes.SKA1_Low
        elif hpso in HPSOs.hpsos_using_SKA1Mid:
            telescope = Telescopes.SKA1_Mid
        elif hpso in HPSOs.hpsos_using_SKA1Sur:
            telescope = Telescopes.SKA1_Sur_old
        else:
            raise Exception('HPSO not associated with a telescope')

        return telescope

    @staticmethod
    def apply_band_parameters(o, band):
        """
        Applies the parameters that apply to the band to the parameter container object o
        @param o:
        @param band:
        """
        if band == Bands.Low:
            o.telescope = Telescopes.SKA1_Low
            o.freq_min = 50e6   # in Hz
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

    @staticmethod
    def apply_imaging_mode_parameters(o, mode):
        """
        Applies the parameters that apply to the imaging mode to the parameter container object o
        @param o:
        @param mode:
        @raise Exception:
        """
        if mode == ImagingModes.Continuum:
            o.Qfov = 1.8  # Field of view factor
            o.Nmajor = 10  # Number of major CLEAN cycles to be done
            o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
            o.Nf_out = min(500, o.Nf_max)
            o.Tobs = 6 * 3600  # in seconds

        elif mode == ImagingModes.Spectral:
            o.Qfov = 1.0  # Field of view factor
            o.Nmajor = 1.5  # Number of major CLEAN cycles to be done: updated to 1.5 as post-PDR fix.
            o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
            o.Nf_out = o.Nf_max  # The same as the maximum number of channels
            o.Tobs = 6 * 3600

        elif mode == ImagingModes.SlowTrans:
            o.Qfov = 0.9  # Field of view factor
            o.Nmajor = 1  # Number of major CLEAN cycles to be done
            o.Qpix = 1.5  # Quality factor of synthesised beam oversampling
            o.Nf_out = min(500, o.Nf_max)  # Initially this value was computed, but now capped to 500.
            o.Tobs = 1.0  # Used to be equal to Tdump but after talking to Rosie set this to 1.2 sec
        else:
            raise Exception('Unknown mode!')

    @staticmethod
    def apply_hpso_parameters(o, hpso):
        """
        Applies the parameters that apply to the supplied HPSO to the parameter container object o
        @param o:
        @param hpso:
        """
        if hpso == HPSOs.hpso01:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 50e6
            o.freq_max = 200e6
            o.Nbeam = 2  # using 2 beams as per HPSO request...
            o.Nf_out = 1500  # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Tobs = 6 * 3600
            o.Nf_max = 1500
            o.Bmax = 100000  # m
            o.Texp = 2500 * 3600  # sec
            o.Tpoint = 1000 * 3600  # sec
        elif hpso == HPSOs.hpso02A:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 50e6
            o.freq_max = 200e6
            o.Nbeam = 2  # using 2 beams as per HPSO request...
            o.Tobs = 6 * 3600  # sec
            # o.Nf_max    = 256000
            o.Nf_out = 1500  # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax = 100000  # m
            o.Texp = 2500 * 3600  # sec
            o.Tpoint = 100 * 3600  # sec
        elif hpso == HPSOs.hpso02B:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 50e6
            o.freq_max = 200e6
            o.Nbeam = 2  # using 2 beams as per HPSO request...
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 256000
            o.Nf_out = 1500  # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax = 100000  # m
            o.Texp = 2500 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso03A:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode = ImagingModes.SlowTrans
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
            o.telescope = Telescopes.SKA1_Low_old
            o.mode = ImagingModes.SlowTrans
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
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.SlowTrans
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
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.SlowTrans
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
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.SlowTrans
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
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.SlowTrans
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
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.ContAndSpectral
            o.comment = 'HI, limited BW'
            o.freq_min = 790e6
            o.freq_max = 950e6
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 3200  # Assume 500 in continuum as well - defualt.
            o.Bmax = 40000  # m
            o.Texp = 5000 * 3600  # sec
            o.Tpoint = 2500 * 3600  # sec
        elif hpso == HPSOs.hpso13c:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.Continuum
            o.comment = 'HI, limited BW'
            o.freq_min = 790e6
            o.freq_max = 950e6
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 3200  # Assume 500 in continuum as well - defualt.
            o.Bmax = 40000  # m
            o.Texp = 5000 * 3600  # sec
            o.Tpoint = 2500 * 3600  # sec
        elif hpso == HPSOs.hpso13s:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.Spectral
            o.comment = 'HI, limited BW'
            o.freq_min = 790e6
            o.freq_max = 950e6
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 3200  # Assume 500 in continuum as well - defualt.
            o.Bmax = 40000  # m
            o.Texp = 5000 * 3600  # sec
            o.Tpoint = 2500 * 3600  # sec
        elif hpso == HPSOs.hpso14:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.ContAndSpectral
            o.comment = 'HI'
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 5000  # Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 200000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso14c:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.Continuum
            o.comment = 'HI'
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 5000  # Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 200000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso14s:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.Spectral
            o.comment = 'HI'
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 5000  # Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 200000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso14sfull:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.Spectral
            o.comment = 'HI'
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 50000  # Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 200000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso15:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.ContAndSpectral
            o.comment = 'HI, limited spatial resolution'
            o.freq_min = 1.415e9
            o.freq_max = 1.425e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 2500  # Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 13000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso15c:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.Continuum
            o.comment = 'HI, limited spatial resolution'
            o.freq_min = 1.415e9
            o.freq_max = 1.425e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 2500  # Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 13000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso15s:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.Spectral
            o.comment = 'HI, limited spatial resolution'
            o.freq_min = 1.415e9
            o.freq_max = 1.425e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 2500  # Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax = 13000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso19:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.SlowTrans
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
            o.telescope = Telescopes.SKA1_Mid_old
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
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 1.0e9
            o.freq_max = 1.5e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_max = 256000
            o.Nf_out = 500  # continuum experiment with 500 output channels
            o.Bmax = 50000  # m
            o.Texp = 17500 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso33:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 1.0e9
            o.freq_max = 1.5e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 500  # continuum experiment with 500 output channels
            o.Bmax = 50000  # m
            o.Texp = 17500 * 3600  # sec
            o.Tpoint = 10 * 3600  # sec
        elif hpso == HPSOs.hpso35:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.SlowTrans
            o.comment = 'Autocorrelation'
            o.freq_min = 650e6
            o.freq_max = 1.15e9
            o.Tobs = 0 * 3600  # sec
            o.Nf_max = 256000
            o.Nf_out = 500
            o.Bmax = 10000  # m
            o.Texp = 5500 * 3600  # sec
            o.Tpoint = 3.3 * 3600  # sec
        elif hpso == HPSOs.hpso37a:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 1e9
            o.freq_max = 1.7e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 700  # 700 channels required in output continuum cubes
            o.Bmax = 200000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 95 * 3600  # sec
        elif hpso == HPSOs.hpso37b:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 1e9
            o.freq_max = 1.7e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 700  # 700 channels required in output continuum cubes
            o.Bmax = 200000  # m
            o.Texp = 2000 * 3600  # sec
            o.Tpoint = 2000 * 3600  # sec
        elif hpso == HPSOs.hpso37c:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 1e9
            o.freq_max = 1.5e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 500  # 500 channels in output cube
            o.Bmax = 50000  # m
            o.Texp = 5300 * 3600  # sec
            o.Tpoint = 95 * 3600  # sec
        elif hpso == HPSOs.hpso38a:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 7e9
            o.freq_max = 11e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 1000
            o.Bmax = 200000  # m
            o.Texp = 1000 * 3600  # sec
            o.Tpoint = 16.4 * 3600  # sec
        elif hpso == HPSOs.hpso38b:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode = ImagingModes.Continuum
            o.freq_min = 7e9
            o.freq_max = 11e9
            o.Tobs = 6 * 3600  # sec
            o.Nf_out = 1000
            o.Bmax = 200000  # m
            o.Texp = 1000 * 3600  # sec
            o.Tpoint = 1000 * 3600  # sec
        else:
            raise Exception('Unknown HPSO!')