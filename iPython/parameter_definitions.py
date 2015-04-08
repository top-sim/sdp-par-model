import sympy.physics.units as u
from sympy import symbols
from astropy import constants as const
import numpy as np

# Enumerate the possible telescopes to choose from (used in the ParameterDefinitions class)
class Telescopes:
    # The originally planned SKA1 telescopes
    SKA1_Low_old = 'SKA1Low_old'
    SKA1_Mid_old = 'SKA1Mid_old'
    SKA1_Sur_old = 'SKA1Survey_old'
    # The rebaselined SKA1 telescopes
    SKA1_Low = 'SKA1Low_rebaselined'
    SKA1_Mid = 'SKA1Mid_rebaselined'
    # Proposed SKA2 telescopes
    SKA2_Low = 'SKA2Low'
    SKA2_Mid = 'SKA2Mid'

# Enumerate all possible bands (used in the ParameterDefinitions class)
class Bands :
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
                           hpso38b,  hpso14c, hpso14s,  hpso14sfull}
    hpsos_using_SKA1Sur = {hpso13, hpso15, hpso27, hpso33, hpso35, hpso37c, hpso13c,  hpso13s,  hpso15c,  hpso15s}

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
        o.binfrac = symbols("f_bin", positive=True) # The fraction of baselines that fall in a given bin (used for baseline-dependent calculations)
        o.Nfacet = symbols("N_facet", integer=True, positive=True)

    @staticmethod
    def apply_global_parameters(o):
        """
        Applies the global parameters to the parameter container object o
        @param o:
        """
        o.Omega_E = 7.292115e-5  # In PDR05 Excel sheet a value of 0.0000727 was used. value based on rotation relative to the fixed stars
        o.R_Earth = const.R_earth.value * u.m  # In the original PDR05 Excel sheet a value of 6400,000 was used
        o.epsilon_w = 0.01
        o.Mvis = 12.0  # back to 12. Likely to change in future
        o.Naa = 10  # Changed to 10, after PDR submission
        o.Nmm = 4
        o.Npp = 4
        o.Nw  = 2  # Bytes per value
        o.Qbw  = 1.0
        o.Qfcv = 10.0
        o.Qgcf = 8.0
        o.Qw   = 1.0
        o.Tion = 60.0
        o.Tsnap_min = 1.2
        o.amp_f_max = 1.01  # Added by Rosie Bolton
        o.BL_dep_time_av = False #New parameter to act as a switch for BL dependent time averaging

    @staticmethod
    def apply_telescope_parameters(o, telescope):
        """
        Applies the parameters that apply to the supplied telescope to the parameter container object o
        @param o:
        @param telescope:
        """
        if telescope == Telescopes.SKA1_Low:
            o.Bmax = 80 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 35 * u.m        # station "diameter" in meters
            o.Na = 512            # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 65356      # maximum number of channels
            o.Tdump_ref = 0.6 * u.s # Correlator dump time in reference design
            o.baseline_bins = np.array((4.9, 7.1, 10.4, 15.1, 22.1, 32.2, 47.0, 80.0)) * u.km
            o.baseline_bin_counts = np.array((5031193, 732481, 796973, 586849, 1070483, 939054, 820834, 202366))
        elif telescope == Telescopes.SKA1_Mid:
            o.Bmax = 150 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 133+64          # number of antennas (expressed as the sum between new and Meerkat antennas)
            o.Nbeam = 1            # number of beams
            o.Nf_max = 65356      # maximum number of channels
            o.Tdump_ref = 0.08 * u.s # Correlator dump time in reference design
            o.baseline_bins = np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 130.8, 150)) * u.km
            o.baseline_bin_counts = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
        elif telescope == Telescopes.SKA1_Low_old:
            o.Bmax = 100 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 35 * u.m        # station "diameter" in meters
            o.Na = 1024            # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.6* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.9, 7.1, 10.4, 15.1, 22.1, 32.2, 47.0, 68.5, 100)) * u.km
            o.baseline_bin_counts  = np.array((5031193, 732481, 796973, 586849, 1070483, 939054, 820834, 202366, 12375))
        elif telescope == Telescopes.SKA1_Mid_old:
            o.Bmax = 200 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 190+64          # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.08* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 130.8, 200)) * u.km
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
        elif telescope == Telescopes.SKA1_Sur_old:
            o.Bmax = 50 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 96            # number of antennas
            o.Nbeam = 36            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.3* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((3.8, 5.5, 8.0, 11.5, 16.6, 24.0, 34.6, 50)) * u.km
            o.baseline_bin_counts  = np.array((81109, 15605, 15777, 16671, 16849, 17999, 3282, 324))
        elif telescope == Telescopes.SKA2_Low:
            o.Bmax = 180 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 180 * u.m        # station "diameter" in meters
            o.Na = 155            # number of antennas
            o.Nbeam = 200            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.08* u.s # Correlator dump time in reference design
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
            o.baseline_bins  = np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 130.8, 180)) * u.km
        elif telescope == Telescopes.SKA2_Mid:
            o.Bmax = 1800 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 155            # number of antennas
            o.Nbeam = 200            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.008* u.s # Correlator dump time in reference design
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
            o.baseline_bins  = np.array((44, 67, 103, 157, 240, 367, 560, 856, 1308, 1800)) * u.km

    @staticmethod
    def get_telescope_from_band(band):
        """
        Returns the telescope that is associated with the specified band
        @param band:
        @return: @raise Exception:
        """
        telescope = None
        if band in Bands.low_bands:
            telescope = Telescopes.SKA1_Low_old
        elif band in Bands.mid_bands:
            telescope = Telescopes.SKA1_Mid_old
        elif band in Bands.survey_bands:
            telescope = Telescopes.SKA1_Sur_old
        elif band in Bands.low_ska2_bands:
            telescope = Telescopes.SKA2_Low
        elif band in Bands.mid_ska2_bands:
            telescope = Telescopes.SKA2_Mid
        else:
            raise Exception("Unknown band %s" % band)
        return telescope

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
            telescope = Telescopes.SKA1_Low_old
        elif hpso in HPSOs.hpsos_using_SKA1Mid:
            telescope = Telescopes.SKA1_Mid_old
        elif hpso in HPSOs.hpsos_using_SKA1Sur:
            telescope = Telescopes.SKA1_Sur_old
        else:
            raise Exception()

        return telescope

    @staticmethod
    def apply_band_parameters(o, band):
        """
        Applies the parameters that apply to the band to the parameter container object o
        @param o:
        @param band:
        """
        if band == Bands.Low:
            o.telescope = Telescopes.SKA1_Low_old
            o.freq_min =  50e6 * u.Hz
            o.freq_max = 350e6 * u.Hz
        elif band == Bands.Mid1:
            o.telescope = Telescopes.SKA1_Mid_old
            o.freq_min =  350e6 * u.Hz
            o.freq_max = 1.05e9 * u.Hz
        elif band == Bands.Mid2:
            o.telescope = Telescopes.SKA1_Mid_old
            o.freq_min = 949.367e6 * u.Hz
            o.freq_max = 1.7647e9 * u.Hz
        elif band == Bands.Mid3:
            o.telescope = Telescopes.SKA1_Mid_old
            o.freq_min = 1.65e9 * u.Hz
            o.freq_max = 3.05e9 * u.Hz
        elif band == Bands.Mid4:
            o.telescope = Telescopes.SKA1_Mid_old
            o.freq_min = 2.80e9 * u.Hz
            o.freq_max = 5.18e9 * u.Hz
        elif band == Bands.Mid5A:
            o.telescope = Telescopes.SKA1_Mid_old
            o.freq_min = 4.60e9 * u.Hz
            o.freq_max = 7.10e9 * u.Hz
        elif band == Bands.Mid5B:
            o.telescope = Telescopes.SKA1_Mid_old
            o.freq_min = 11.3e9 * u.Hz
            o.freq_max = 13.8e9 * u.Hz
        elif band == Bands.Sur1:
            o.telescope = Telescopes.SKA1_Sur_old
            o.freq_min = 350e6 * u.Hz
            o.freq_max = 850e6 * u.Hz
        elif band == Bands.Sur2A:
            o.telescope = Telescopes.SKA1_Sur_old
            o.freq_min =  650e6 * u.Hz
            o.freq_max = 1.35e9 * u.Hz
        elif band == Bands.Sur2B:
            o.telescope = Telescopes.SKA1_Sur_old
            o.freq_min = 1.17e9 * u.Hz
            o.freq_max = 1.67e9 * u.Hz
        elif band == Bands.Sur3A:
            o.telescope = Telescopes.SKA1_Sur_old
            o.freq_min = 1.5e9 * u.Hz
            o.freq_max = 2.0e9 * u.Hz
        elif band == Bands.Sur3B:
            o.telescope = Telescopes.SKA1_Sur_old
            o.freq_min = 3.5e9 * u.Hz
            o.freq_max = 4.0e9 * u.Hz
        elif band == Bands.SKA2Low:
            o.telescope = Telescopes.SKA2_Low
            o.freq_min =  70e6 * u.Hz
            o.freq_max = 350e6 * u.Hz
        elif band == Bands.SKA2Mid:
            o.telescope = Telescopes.SKA2_Mid
            o.freq_min =  450e6 * u.Hz
            o.freq_max = 1.05e9 * u.Hz

    @staticmethod
    def apply_imaging_mode_parameters(o, mode):
        """
        Applies the parameters that apply to the imaging mode to the parameter container object o
        @param o:
        @param mode:
        @raise Exception:
        """
        if mode == ImagingModes.Continuum:
            o.Qfov =  1.8 # Field of view factor
            o.Nmajor = 10 # Number of major CLEAN cycles to be done
            o.Qpix =  2.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = min(500, o.Nf_max)
            o.Tobs  = 6 * u.hours

        elif mode == ImagingModes.Spectral:
            o.Qfov = 1.0 # Field of view factor
            o.Nmajor = 1.5 # Number of major CLEAN cycles to be done: updated to 1.5 as post-PDR fix.
            o.Qpix = 2.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = o.Nf_max #The same as the maximum number of channels
            o.Tobs  = 6 * u.hours

        elif mode == ImagingModes.SlowTrans:
            o.Qfov = 0.9 # Field of view factor
            o.Nmajor = 1 # Number of major CLEAN cycles to be done
            o.Qpix = 1.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = min(500, o.Nf_max)  # Initially this value was computed, but Rosie has since specified that it should just be set to 500.
            o.Tobs  = 1.2 * u.s  # Used to be equal to Tdump but after talking to Rosie set this to 1.2 sec
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
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  50e6 * u.Hz
            o.freq_max  = 200e6 * u.Hz
            o.Nbeam     = 2 #using 2 beams as per HPSO request...
            o.Nf_out    = 1500 # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 1500
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 2500 * u.hours
            o.Tpoint    = 1000 * u.hours
        elif hpso == HPSOs.hpso02A:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  50e6 * u.Hz
            o.freq_max  = 200e6 * u.Hz
            o.Nbeam     = 2 #using 2 beams as per HPSO request...
            o.Tobs      = 6 * u.hours
            #o.Nf_max    = 256000
            o.Nf_out    = 1500 # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 2500 * u.hours
            o.Tpoint    = 100 * u.hours
        elif hpso == HPSOs.hpso02B:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  50e6 * u.Hz
            o.freq_max  = 200e6 * u.Hz
            o.Nbeam     = 2 #using 2 beams as per HPSO request...
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 256000
            o.Nf_out    = 1500 # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 2500 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso03A:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode      = ImagingModes.SlowTrans
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate'
            o.freq_min  = 150e6 * u.Hz
            o.freq_max  = 350e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 12800 * u.hours
            o.Tpoint    = 0.17 * u.hours
            o.Nmajor    = 10
        elif hpso == HPSOs.hpso03B:
            o.telescope = Telescopes.SKA1_Low_old
            o.mode      = ImagingModes.SlowTrans
            o.comment   = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate'
            o.freq_min  = 150e6 * u.Hz
            o.freq_max  = 350e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 4300 * u.hours
            o.Tpoint    = 0.17 * u.hours
            o.Nmajor    = 10
        elif hpso == HPSOs.hpso04A:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.SlowTrans
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 650e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 800 * u.hours
            o.Tpoint    = 10/60.0 * u.hours
            o.Nmajor    = 10
        elif hpso == HPSOs.hpso04B:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.SlowTrans
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 1.25e9 * u.Hz
            o.freq_max  = 1.55e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 800 * u.hours
            o.Tpoint    = 10/60.0 * u.hours
            o.Nmajor    = 10
        elif hpso == HPSOs.hpso05A:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.SlowTrans
            o.comment = 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 0.95e9 * u.Hz
            o.freq_max  = 1.76e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 15 * u.kilometer
            o.Texp      = 1600 * u.hours
            o.Tpoint    = (10/60.0) * u.hours
            o.Nmajor    = 10
        elif hpso == HPSOs.hpso05B:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.SlowTrans
            o.comment = 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 1.65e9 * u.Hz
            o.freq_max  = 3.05e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 15 * u.kilometer
            o.Texp      = 1600 * u.hours
            o.Tpoint    = (10/60.0) * u.hours
            o.Nmajor    = 10
        elif hpso == HPSOs.hpso13:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.ContAndSpectral
            o.comment = 'HI, limited BW'
            o.freq_min  = 790e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 3200 #Assume 500 in continuum as well - defualt.
            o.Bmax      = 40 * u.kilometer
            o.Texp      = 5000 * u.hours
            o.Tpoint    = 2500 * u.hours
        elif hpso == HPSOs.hpso13c:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.Continuum
            o.comment = 'HI, limited BW'
            o.freq_min  = 790e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 3200 #Assume 500 in continuum as well - defualt.
            o.Bmax      = 40 * u.kilometer
            o.Texp      = 5000 * u.hours
            o.Tpoint    = 2500 * u.hours
        elif hpso == HPSOs.hpso13s:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.Spectral
            o.comment = 'HI, limited BW'
            o.freq_min  = 790e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 3200 #Assume 500 in continuum as well - defualt.
            o.Bmax      = 40 * u.kilometer
            o.Texp      = 5000 * u.hours
            o.Tpoint    = 2500 * u.hours
        elif hpso == HPSOs.hpso14:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.ContAndSpectral
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 5000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso14c:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Continuum
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 5000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso14s:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Spectral
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 5000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso14sfull:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Spectral
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 50000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso15:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.ContAndSpectral
            o.comment = 'HI, limited spatial resolution'
            o.freq_min  =  1.415e9 * u.Hz
            o.freq_max  =  1.425e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 2500 #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 13 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso15c:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.Continuum
            o.comment = 'HI, limited spatial resolution'
            o.freq_min  =  1.415e9 * u.Hz
            o.freq_max  =  1.425e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 2500 #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 13 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso15s:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.Spectral
            o.comment = 'HI, limited spatial resolution'
            o.freq_min  =  1.415e9 * u.Hz
            o.freq_max  =  1.425e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 2500 #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 13 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso19:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.SlowTrans
            o.comment = 'Transients. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 650e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 10000 * u.hours
            o.Tpoint    = (10/60.0) * u.hours
            o.Nmajor    = 10
        elif hpso == HPSOs.hpso22:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Continuum
            o.comment = 'Cradle of life'
            o.freq_min  =  10e9 * u.Hz
            o.freq_max  =  12e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 4000
            o.Nf_out    = 4000 #4000 channel continuum observation - band 5.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 6000 * u.hours
            o.Tpoint    = 600 * u.hours
        elif hpso == HPSOs.hpso27:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  1.0e9 * u.Hz
            o.freq_max  =  1.5e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 256000
            o.Nf_out    = 500 #continuum experiment with 500 output channels
            o.Bmax      = 50 * u.kilometer
            o.Texp      = 17500 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso33:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  1.0e9 * u.Hz
            o.freq_max  =  1.5e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 500 #continuum experiment with 500 output channels
            o.Bmax      = 50 * u.kilometer
            o.Texp      = 17500 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == HPSOs.hpso35:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.SlowTrans
            o.comment = 'Autocorrelation'
            o.freq_min  =   650e6 * u.Hz
            o.freq_max  =  1.15e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Nf_out    = 500
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 5500 * u.hours
            o.Tpoint    = 3.3 * u.hours
        elif hpso == HPSOs.hpso37a:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  1e9 * u.Hz
            o.freq_max  =  1.7e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 700 #700 channels required in output continuum cubes
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 95 * u.hours
        elif hpso == HPSOs.hpso37b:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  1e9 * u.Hz
            o.freq_max  =  1.7e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 700 #700 channels required in output continuum cubes
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 2000 * u.hours
        elif hpso == HPSOs.hpso37c:
            o.telescope = Telescopes.SKA1_Sur_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  1e9 * u.Hz
            o.freq_max  =  1.5e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 500 #500 channels in output cube
            o.Bmax      = 50 * u.kilometer
            o.Texp      = 5300 * u.hours
            o.Tpoint    = 95 * u.hours
        elif hpso == HPSOs.hpso38a:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  7e9 * u.Hz
            o.freq_max  =  11e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 1000
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 1000 * u.hours
            o.Tpoint    = 16.4 * u.hours
        elif hpso == HPSOs.hpso38b:
            o.telescope = Telescopes.SKA1_Mid_old
            o.mode      = ImagingModes.Continuum
            o.freq_min  =  7e9 * u.Hz
            o.freq_max  =  11e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 1000
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 1000 * u.hours
            o.Tpoint    = 1000 * u.hours