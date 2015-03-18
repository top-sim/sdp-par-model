import sympy.physics.units as u
from sympy import symbols
from astropy import constants as const
import numpy as np

class parameter_definitions:
    telescope_list = ['SKA1_Low', 'SKA1_Mid', 'SKA1_Low_OLD','SKA1_Mid_OLD']
    imaging_modes = ['Continuum', 'Spectral', 'SlowTrans', 'CS']   # CS = Continuum, followed by spectral. Used for HPSOs
    
    @staticmethod
    def define_symbolic_variables(o):
        '''
        This method defines the *symbolic* variables that we will use during
        computations. 
        All the commented out ones further down below used to be
        defined symbolically and then given numeric values. This was nice for displaying
        formulae, but slowed down computations unnecessarily.
        '''
        o.Tsnap = symbols("T_snap", positive=True)  # Snapshot timescale implemented
        o.Bmax_bin = symbols("Bmax\,bin", positive=True)
        o.binfrac = symbols("f_bin", positive=True) # The fraction of baselines that fall in a given bin (used for baseline-dependent calculations)
        o.Nfacet = symbols("N_facet", integer=True, positive=True)

        # Previously defined symbolic variables; now no longer necessary to be defined symbolically
        '''
        o.Rflop = symbols("R_flop", positive=True)
        o.Mbuf_vis = symbols("M_buf\,vis", positive=True)
        o.Rio = symbols("R_io", positive=True)
        o.Gcorr = symbols("G_corr", positive=True)
        o.Rphrot = symbols("R_phrot", positive=True)

        o.Blim_mid = symbols("B_lim\,mid", positive=True)
        o.Qfov = symbols("Q_FoV", positive=True)

        o.Nmajor = symbols("N_major", integer=True, positive=True)
        o.Nf_used = symbols("N_f\,used", integer=True, positive=True)
        o.Nf_out = symbols("N_f\,out", integer=True, positive=True)
        o.Nf_max = symbols("N_f\,max", integer=True, positive=True)
        o.Nf_no_smear = symbols("N_f\,no-smear", integer=True, positive=True)
        o.Npix_linear = symbols("N_pix\,linear ", integer=True, positive=True)
        o.Nminor = symbols("N_minor", integer=True, positive=True)

        o.Na = symbols("N_a", integer=True, positive=True)
        o.Nbeam = symbols("N_beam", integer=True, positive=True)
        o.Ngw = symbols("N_gw", positive=True)

        o.Tobs = symbols("T_obs", positive=True)
        o.Qpix = symbols("Q_pix", positive=True)

        o.Ds = symbols("D_s", positive=True)
        o.Bmax = symbols("B_max", positive=True)
        o.freq_min = symbols("f_min", positive=True)
        o.freq_max = symbols("f_max")

        # These two variables are for computing baseline-dependent variables (by approximating baseline distribution as a series of bins)
        o.baseline_bins  = symbols("B_bins", positive=True)
        o.baseline_bin_counts = symbols("B_bin_counts", positive=True, integer=True)

        o.Tdump_ref = symbols("T_dump\,ref", positive=True)

        # Wavelength variables, not (yet) enumerated in the table above
        o.wl = symbols("\lambda", positive=True)
        o.wl_max = symbols("\lambda_max", positive=True)
        o.wl_min = symbols("\lambda_min", positive=True)
        o.wl_sub_max = symbols("\lambda_{sub\,max}", positive=True)
        o.wl_sub_min = symbols("\lambda_{sub\,min}", positive=True)

        # Variables unique to the HPSO experiments
        o.Texp = symbols("T_exp", positive=True)
        o.Tpoint = symbols("T_point", positive=True)

        # Other variables (may be needed later on)
        o.Tdump = symbols("T_dump", positive=True)  # Correlator dump time (s)
        o.psf_angle= symbols(r"\theta_{PSF}", positive=True)
        o.pix_angle  = symbols(r"\theta_{pix}", positive=True)
        o.Theta_beam = symbols(r"\theta_{beam}", positive=True)
        o.Theta_fov = symbols(r"\theta_{FoV}", positive=True)
        o.Rrp = symbols("R_rp", positive=True) # Reprojection Flop rate, per output channel
        '''

    @staticmethod
    def apply_global_parameters(o):
        o.Omega_E = 7.292115e-5  # In PDR05 Excel sheet a value of 0.0000727 was used. This value based on rotation relative to the fixed stars
        o.R_Earth = const.R_earth.value * u.m # In PDR05 Excel sheet a value of 6400000 was used
        o.epsilon_w = 0.01
        o.Mvis = 12.0 #back to 12. Likely to change in future
        o.Naa = 10 #Changed to 10, after PDR submission
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
        if telescope == 'SKA1_Low_OLD':
            o.Bmax = 100 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 35 * u.m        # station "diameter" in meters
            o.Na = 1024            # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.6* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.9, 7.1, 10.4, 15.1, 22.1, 32.2, 47.0, 68.5, 100)) * u.km
            o.baseline_bin_counts  = np.array((5031193, 732481, 796973, 586849, 1070483, 939054, 820834, 202366, 12375))
        elif telescope == 'SKA1_Mid_OLD':
            o.Bmax = 200 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 190+64          # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 256000      # maximum number of channels
            o.Tdump_ref = 0.08* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 130.8, 200)) * u.km
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745))
        elif telescope == 'SKA1_Low':
            o.Bmax = 70 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 35 * u.m        # station "diameter" in meters
            o.Na = 512            # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 65356      # maximum number of channels
            o.Tdump_ref = 0.6* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.9, 7.1, 10.4, 15.1, 22.1, 32.2, 47.0, 80.0)) * u.km
            o.baseline_bin_counts  = np.array((5031193, 732481, 796973, 586849, 1070483, 939054, 820834, 214741))
        elif telescope == 'SKA1_Mid':
            o.Bmax = 150 * u.km     # Actually constructed kilometers of max baseline
            o.Ds = 15 * u.m        # station "diameter" in meters
            o.Na = 133+64          # number of antennas
            o.Nbeam = 1            # number of beams
            o.Nf_max = 65356      # maximum number of channels
            o.Tdump_ref = 0.08* u.s # Correlator dump time in reference design
            o.baseline_bins  = np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 150)) * u.km
            o.baseline_bin_counts  = np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 22280))
    @staticmethod
    def get_telescope_from_band(band_string):
        telescope_string = band_string
            #       if band_string.upper().startswith('L'):
            #telescope_string = 'SKA1_Low'
            #elif band_string.upper().startswith('M'):
            #telescope_string = 'SKA1_Mid'
            #elif band_string.upper().startswith('S'):
            #telescope_string = 'SKA1_Survey'
            #else:
            #raise Exception("Unknown band string %s" % band_string)
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
        if band == 'SKA1_Low':
            o.telescope = 'SKA1_Low'
            o.freq_min =  50e6 * u.Hz
            o.freq_max = 350e6 * u.Hz
        elif band == 'SKA1_Mid':
            o.telescope = 'SKA1_Mid'
            o.freq_min =  350e6 * u.Hz
            o.freq_max = 1.05e9 * u.Hz
        elif band == 'SKA1_Low_OLD':
            o.telescope = 'SKA1_Low_OLD'
            o.freq_min =  50e6 * u.Hz
            o.freq_max = 350e6 * u.Hz
        elif band == 'SKA1_Mid_OLD':
            o.telescope = 'SKA1_Mid_OLD'
            o.freq_min =  350e6 * u.Hz
            o.freq_max = 1.05e9 * u.Hz

    @staticmethod
    def apply_imaging_mode_parameters(o, mode):
        if mode == 'Continuum':
            o.Qfov =  1.8 # Field of view factor
            o.Nmajor = 10 # Number of major CLEAN cycles to be done
            o.Qpix =  2.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = 500
            o.Tobs  = 6 * u.hours

        elif mode == 'Spectral':
            o.Qfov = 1.0 # Field of view factor
            o.Nmajor = 1.5 # Number of major CLEAN cycles to be done: updated to 1.5 as post-PDR fix.
            o.Qpix = 2.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = o.Nf_max #The same as the number of channels
            o.Tobs  = 6 * u.hours

        elif mode == 'SlowTrans':
            o.Qfov = 0.9 # Field of view factor
            o.Nmajor = 1 # Number of major CLEAN cycles to be done
            o.Qpix = 1.5 # Quality factor of synthesised beam oversampling
            o.Nf_out  = 500  # Initially this value was computed (see line above) but Rosie has since specified that it should just be set to 500.
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
            o.Nf_max    = 64000
            o.Nf_out    = 1500 # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
            o.Bmax      = 100 * u.kilometer
            o.Texp      = 2500 * u.hours
            o.Tpoint    = 10 * u.hours
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