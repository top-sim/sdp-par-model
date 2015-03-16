import sympy.physics.units as u
from sympy import symbols
from astropy import constants as const
import numpy as np

class parameter_definitions:
    telescope_list = ['SKA1_Low', 'SKA1_Mid', 'SKA1_Survey', 'SKA2_LOW', 'SKA2_MID']
    imaging_modes = ['Continuum', 'Spectral', 'SlowTrans', 'CS']   # CS = Continuum, followed by spectral. Used for HPSOs
    
    @staticmethod
    def define_symbolic_variables(o):
        """
        This method defines the *symbolic* variables that we will use during computations
        and that need to be kept symbolic during evaluation of formulae. One reason to do this would be to allow
        the output formula to be optimized by varying this variable (such as with Tsnap and Nfacet)
        """
        o.Tsnap = symbols("T_snap", positive=True)  # Snapshot timescale implemented
        o.Bmax_bin = symbols("Bmax\,bin", positive=True)
        o.binfrac = symbols("f_bin", positive=True) # The fraction of baselines that fall in a given bin (used for baseline-dependent calculations)
        o.Nfacet = symbols("N_facet", integer=True, positive=True)

    @staticmethod
    def apply_global_parameters(o):
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

    @staticmethod
    def apply_telescope_parameters(o, telescope):
        if telescope == 'SKA1_Low':
            o.Bmax = 100 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 35 * u.m        # station "diameter" in meters
            o.Na = 1024            # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.6* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.9, 7.1, 10.4, 15.1, 22.1, 32.2, 47.0, 68.5, 100)) * u.km
            o.baseline_bin_counts  = np.array((5031193, 732481, 796973, 586849, 1070483, 939054, 820834, 202366, 12375))
        elif telescope == 'SKA1_Mid':
            o.Bmax = 200 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 190+64          # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.08* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 130.8, 200)) * u.km
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
        elif telescope == 'SKA1_Survey':
            o.Bmax = 50 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 96            # number of antennas
            o.Nbeam = 36            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.3* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((3.8, 5.5, 8.0, 11.5, 16.6, 24.0, 34.6, 50)) * u.km
            o.baseline_bin_counts  = np.array((81109, 15605, 15777, 16671, 16849, 17999, 3282, 324))
        elif telescope == 'SKA2_LOW':
            o.Bmax = 180 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 180 * u.m        # station "diameter" in meters
            o.Na = 155            # number of antennas
            o.Nbeam = 200            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.08* u.s # Correlator dump time in reference design
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
            o.baseline_bins  = np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 130.8, 180)) * u.km
        elif telescope == 'SKA2_MID':
            o.Bmax = 1800 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 155            # number of antennas
            o.Nbeam = 200            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.008* u.s # Correlator dump time in reference design
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
            o.baseline_bins  = np.array((44, 67, 103, 157, 240, 367, 560, 856, 1308, 1800)) * u.km
    @staticmethod
    def get_telescope_from_band(band_string):
        if band_string.upper().startswith('L'):
            telescope_string = 'SKA1_Low'
        elif band_string.upper().startswith('M'):
            telescope_string = 'SKA1_Mid'
        elif band_string.upper().startswith('S'):
            telescope_string = 'SKA1_Survey'
        else:
            raise Exception("Unknown band string %s" % band_string)
        return telescope_string

    @staticmethod
    def get_telescope_from_hpso(hpso_string):
        telescope_string = ""
        hpsos_using_low = ('01','02A','02B','03A','03B')
        hpsos_using_mid = ('04A','04B','05A','05B','14','19','22','37a','37b','38a','38b', '14c', '14s', '14sfull')
        hpsos_using_sur = ('13','15','27','33','35','37c', '13c', '13s', '15c', '15s')
        if hpso_string in hpsos_using_low:
            telescope_string = "SKA1_Low"
        elif hpso_string in hpsos_using_mid:
            telescope_string = "SKA1_Mid"
        elif hpso_string in hpsos_using_sur:
            telescope_string = "SKA1_Survey"
        else:
            raise Exception()

        return telescope_string

    @staticmethod
    def apply_band_parameters(o, band):
        if band == 'Low':
            o.telescope = 'SKA1_Low'
            o.freq_min =  50e6 * u.Hz
            o.freq_max = 350e6 * u.Hz
        elif band == 'Mid1':
            o.telescope = 'SKA1_Mid'
            o.freq_min =  350e6 * u.Hz
            o.freq_max = 1.05e9 * u.Hz
        elif band == 'Mid2':
            o.telescope = 'SKA1_Mid'
            o.freq_min = 949.367e6 * u.Hz
            o.freq_max = 1.7647e9 * u.Hz
        elif band == 'Mid3':
            o.telescope = 'SKA1_Mid'
            o.freq_min = 1.65e9 * u.Hz
            o.freq_max = 3.05e9 * u.Hz
        elif band == 'Mid4':
            o.telescope = 'SKA1_Mid'
            o.freq_min = 2.80e9 * u.Hz
            o.freq_max = 5.18e9 * u.Hz
        elif band == 'Mid5A':
            o.telescope = 'SKA1_Mid'
            o.freq_min = 4.60e9 * u.Hz
            o.freq_max = 7.10e9 * u.Hz
        elif band == 'Mid5B':
            o.telescope = 'SKA1_Mid'
            o.freq_min = 11.3e9 * u.Hz
            o.freq_max = 13.8e9 * u.Hz
        elif band == 'Sur1':
            o.telescope = 'SKA1_Survey'
            o.freq_min = 350e6 * u.Hz
            o.freq_max = 850e6 * u.Hz
        elif band == 'Sur2A':
            o.telescope = 'SKA1_Survey'
            o.freq_min =  650e6 * u.Hz
            o.freq_max = 1.35e9 * u.Hz
        elif band == 'Sur2B':
            o.telescope = 'SKA1_Survey'
            o.freq_min = 1.17e9 * u.Hz
            o.freq_max = 1.67e9 * u.Hz
        elif band == 'Sur3A':
            o.telescope = 'SKA1_Survey'
            o.freq_min = 1.5e9 * u.Hz
            o.freq_max = 2.0e9 * u.Hz
        elif band == 'Sur3B':
            o.telescope = 'SKA1_Survey'
            o.freq_min = 3.5e9 * u.Hz
            o.freq_max = 4.0e9 * u.Hz
        elif band == 'LOWSKA2':
            o.telescope = 'SKA2_LOW'
            o.freq_min =  70e6 * u.Hz
            o.freq_max = 350e6 * u.Hz
        elif band == 'MIDSKA2':
            o.telescope = 'SKA2_MID'
            o.freq_min =  450e6 * u.Hz
            o.freq_max = 1.05e9 * u.Hz


    @staticmethod
    def apply_imaging_mode_parameters(o, mode):
        if mode == 'Continuum':
            o.Qfov =  1.8 # Field of view factor
            o.Nmajor = 10 # Number of major CLEAN cycles to be done
            o.Qpix =  2.5 # Quality factor of synthesised beam oversampling
            print o.Nf_max
            o.Nf_out  = min(500, o.Nf_max)
            o.Tobs  = 6 * u.hours

        elif mode == 'Spectral':
            o.Qfov = 1.0 # Field of view factor
            o.Nmajor = 1.5 # Number of major CLEAN cycles to be done: updated to 1.5 as post-PDR fix.
            o.Qpix = 2.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = o.Nf_max #The same as the maximum number of channels
            o.Tobs  = 6 * u.hours

        elif mode == 'SlowTrans':
            o.Qfov = 0.9 # Field of view factor
            o.Nmajor = 1 # Number of major CLEAN cycles to be done
            o.Qpix = 1.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = min(500, o.Nf_max)  # Initially this value was computed, but Rosie has since specified that it should just be set to 500.
            o.Tobs  = 1.2 * u.s  # Used to be equal to Tdump but after talking to Rosie set this to 1.2 sec
        else:
            raise Exception('Unknown mode!')

    @staticmethod
    def apply_hpso_parameters(o, hpso):
        if hpso == '01':
            o.telescope = 'SKA1_Low'
            o.mode      = 'Continuum'
            o.freq_min  =  50e6 * u.Hz
            o.freq_max  = 200e6 * u.Hz
            o.Nbeam     = 2 #using 2 beams as per HPSO request...
            o.Nf_out    = 1500 # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 1500
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 2500 * u.hours
            o.Tpoint    = 1000 * u.hours
        elif hpso == '02A':
            o.telescope = 'SKA1_Low'
            o.mode      = 'Continuum'
            o.freq_min  =  50e6 * u.Hz
            o.freq_max  = 200e6 * u.Hz
            o.Nbeam     = 2 #using 2 beams as per HPSO request...
            o.Tobs      = 6 * u.hours
            #o.Nf_max    = 256000
            o.Nf_out    = 1500 # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 2500 * u.hours
            o.Tpoint    = 100 * u.hours
        elif hpso == '02B':
            o.telescope = 'SKA1_Low'
            o.mode      = 'Continuum'
            o.freq_min  =  50e6 * u.Hz
            o.freq_max  = 200e6 * u.Hz
            o.Nbeam     = 2 #using 2 beams as per HPSO request...
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 256000
            o.Nf_out    = 1500 # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 2500 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '03A':
            o.telescope = 'SKA1_Low'
            o.mode      = 'SlowTrans'
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate'
            o.freq_min  = 150e6 * u.Hz
            o.freq_max  = 350e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 12800 * u.hours
            o.Tpoint    = 0.17 * u.hours
            o.Nmajor    = 10
        elif hpso == '03B':
            o.telescope = 'SKA1_Low'
            o.mode      = 'SlowTrans'
            o.comment   = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate'
            o.freq_min  = 150e6 * u.Hz
            o.freq_max  = 350e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 4300 * u.hours
            o.Tpoint    = 0.17 * u.hours
            o.Nmajor    = 10
        elif hpso == '04A':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'SlowTrans'
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 650e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 800 * u.hours
            o.Tpoint    = 10/60.0 * u.hours
            o.Nmajor    = 10
        elif hpso == '04B':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'SlowTrans'
            o.comment = 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 1.25e9 * u.Hz
            o.freq_max  = 1.55e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 800 * u.hours
            o.Tpoint    = 10/60.0 * u.hours
            o.Nmajor    = 10
        elif hpso == '05A':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'SlowTrans'
            o.comment = 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 0.95e9 * u.Hz
            o.freq_max  = 1.76e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 15 * u.kilometer
            o.Texp      = 1600 * u.hours
            o.Tpoint    = (10/60.0) * u.hours
            o.Nmajor    = 10
        elif hpso == '05B':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'SlowTrans'
            o.comment = 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 1.65e9 * u.Hz
            o.freq_max  = 3.05e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 15 * u.kilometer
            o.Texp      = 1600 * u.hours
            o.Tpoint    = (10/60.0) * u.hours
            o.Nmajor    = 10
        elif hpso == '13':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'CS'
            o.comment = 'HI, limited BW'
            o.freq_min  = 790e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 3200 #Assume 500 in continuum as well - defualt.
            o.Bmax      = 40 * u.kilometer
            o.Texp      = 5000 * u.hours
            o.Tpoint    = 2500 * u.hours
        elif hpso == '13c':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'Continuum'
            o.comment = 'HI, limited BW'
            o.freq_min  = 790e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 3200 #Assume 500 in continuum as well - defualt.
            o.Bmax      = 40 * u.kilometer
            o.Texp      = 5000 * u.hours
            o.Tpoint    = 2500 * u.hours
        elif hpso == '13s':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'Spectral'
            o.comment = 'HI, limited BW'
            o.freq_min  = 790e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 3200 #Assume 500 in continuum as well - defualt.
            o.Bmax      = 40 * u.kilometer
            o.Texp      = 5000 * u.hours
            o.Tpoint    = 2500 * u.hours
        elif hpso == '14':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'CS'
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 5000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '14c':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Continuum'
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 5000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '14s':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Spectral'
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 5000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '14sfull':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Spectral'
            o.comment = 'HI'
            o.freq_min  =  1.3e9 * u.Hz
            o.freq_max  =  1.4e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 50000 #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '15':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'CS'
            o.comment = 'HI, limited spatial resolution'
            o.freq_min  =  1.415e9 * u.Hz
            o.freq_max  =  1.425e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 2500 #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 13 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '15c':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'Continuum'
            o.comment = 'HI, limited spatial resolution'
            o.freq_min  =  1.415e9 * u.Hz
            o.freq_max  =  1.425e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 2500 #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 13 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '15s':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'Spectral'
            o.comment = 'HI, limited spatial resolution'
            o.freq_min  =  1.415e9 * u.Hz
            o.freq_max  =  1.425e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 2500 #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
            o.Bmax      = 13 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '19':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'SlowTrans'
            o.comment = 'Transients. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.'
            o.freq_min  = 650e6 * u.Hz
            o.freq_max  = 950e6 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 10000 * u.hours
            o.Tpoint    = (10/60.0) * u.hours
            o.Nmajor    = 10
        elif hpso == '22':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Continuum'
            o.comment = 'Cradle of life'
            o.freq_min  =  10e9 * u.Hz
            o.freq_max  =  12e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 4000
            o.Nf_out    = 4000 #4000 channel continuum observation - band 5.
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 6000 * u.hours
            o.Tpoint    = 600 * u.hours
        elif hpso == '27':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'Continuum'
            o.freq_min  =  1.0e9 * u.Hz
            o.freq_max  =  1.5e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_max    = 256000
            o.Nf_out    = 500 #continuum experiment with 500 output channels
            o.Bmax      = 50 * u.kilometer
            o.Texp      = 17500 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '33':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'Continuum'
            o.freq_min  =  1.0e9 * u.Hz
            o.freq_max  =  1.5e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 500 #continuum experiment with 500 output channels
            o.Bmax      = 50 * u.kilometer
            o.Texp      = 17500 * u.hours
            o.Tpoint    = 10 * u.hours
        elif hpso == '35':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'SlowTrans'
            o.comment = 'Autocorrelation'
            o.freq_min  =   650e6 * u.Hz
            o.freq_max  =  1.15e9 * u.Hz
            o.Tobs      = 0 * u.hours
            o.Nf_max    = 256000
            o.Nf_out    = 500
            o.Bmax      = 10 * u.kilometer
            o.Texp      = 5500 * u.hours
            o.Tpoint    = 3.3 * u.hours
        elif hpso == '37a':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Continuum'
            o.freq_min  =  1e9 * u.Hz
            o.freq_max  =  1.7e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 700 #700 channels required in output continuum cubes
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 95 * u.hours
        elif hpso == '37b':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Continuum'
            o.freq_min  =  1e9 * u.Hz
            o.freq_max  =  1.7e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 700 #700 channels required in output continuum cubes
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 2000 * u.hours
            o.Tpoint    = 2000 * u.hours
        elif hpso == '37c':
            o.telescope = 'SKA1_Survey'
            o.mode      = 'Continuum'
            o.freq_min  =  1e9 * u.Hz
            o.freq_max  =  1.5e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 500 #500 channels in output cube
            o.Bmax      = 50 * u.kilometer
            o.Texp      = 5300 * u.hours
            o.Tpoint    = 95 * u.hours
        elif hpso == '38a':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Continuum'
            o.freq_min  =  7e9 * u.Hz
            o.freq_max  =  11e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 1000
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 1000 * u.hours
            o.Tpoint    = 16.4 * u.hours
        elif hpso == '38b':
            o.telescope = 'SKA1_Mid'
            o.mode      = 'Continuum'
            o.freq_min  =  7e9 * u.Hz
            o.freq_max  =  11e9 * u.Hz
            o.Tobs      = 6 * u.hours
            o.Nf_out    = 1000
            o.Bmax      = 200 * u.kilometer
            o.Texp      = 1000 * u.hours
            o.Tpoint    = 1000 * u.hours