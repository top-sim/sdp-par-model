# from sympy import symbols, pi, log, ln, Max, sqrt, sign

telescope_list = ['SKA1_Low', 'SKA1_Mid', 'SKA1_Survey']
imaging_modes = ['Continuum', 'Spectral', 'SlowTrans', 'CS']   # CS = Continuum, followed by spectral. Used for HPSOs

def apply_global_parameters(o):
    o.Omega_E = 7.292115e-5  # In PDR05 Excel sheet a value of 0.0000727 was used. This value based on rotation relative to the fixed stars
    o.R_Earth = const.R_earth.value * u.m # In PDR05 Excel sheet a value of 6400000 was used
    o.epsilon_w = 0.01
    o.Mvis = 12.0
    o.Naa = 9
    o.Nmm = 4
    o.Npp = 4
    o.Nw  = 2  # Bytes per value
    o.Qbw  = 1.0
    o.Qfcv = 10.0
    o.Qgcf = 8.0
    o.Qw   = 1.0
    o.Tion = 60.0
    o.Tsnap_min = 1.2

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

def apply_band_parameters(o, band):
    if band == 'Low':
        telescope = 'SKA1_Low'
        o.freq_min =  50e6 * u.Hz
        o.freq_max = 350e6 * u.Hz
    elif band == 'Mid1':
        telescope = 'SKA1_Mid'
        o.freq_min =  350e6 * u.Hz
        o.freq_max = 1.05e9 * u.Hz
    elif band == 'Mid2':
        telescope = 'SKA1_Mid'
        o.freq_min = 949.367e6 * u.Hz
        o.freq_max = 1.7647e9 * u.Hz
    elif band == 'Mid3':
        telescope = 'SKA1_Mid'
        o.freq_min = 1.65e9 * u.Hz
        o.freq_max = 3.05e9 * u.Hz
    elif band == 'Mid4':
        telescope = 'SKA1_Mid'
        o.freq_min = 2.80e9 * u.Hz
        o.freq_max = 5.18e9 * u.Hz
    elif band == 'Mid5A':
        telescope = 'SKA1_Mid'
        o.freq_min = 4.60e9 * u.Hz
        o.freq_max = 7.10e9 * u.Hz
    elif band == 'Mid5B':
        telescope = 'SKA1_Mid'
        o.freq_min = 11.3e9 * u.Hz
        o.freq_max = 13.8e9 * u.Hz
    elif band == 'Sur1':
        telescope = 'SKA1_Survey'
        o.freq_min = 350e6 * u.Hz
        o.freq_max = 850e6 * u.Hz
    elif band == 'Sur2A':
        telescope = 'SKA1_Survey'
        o.freq_min =  650e6 * u.Hz
        o.freq_max = 1.35e9 * u.Hz
    elif band == 'Sur2B':
        telescope = 'SKA1_Survey'
        o.freq_min = 1.17e9 * u.Hz
        o.freq_max = 1.67e9 * u.Hz
    elif band == 'Sur3A':
        telescope = 'SKA1_Survey'
        o.freq_min = 1.5e9 * u.Hz
        o.freq_max = 2.0e9 * u.Hz
    elif band == 'Sur3B':
        telescope = 'SKA1_Survey'
        o.freq_min = 3.5e9 * u.Hz
        o.freq_max = 4.0e9 * u.Hz

def apply_imaging_mode_parameters(o, mode):
    if mode == 'Continuum':
        o.Qfov =  1.8 # Field of view factor
        o.Nmajor = 10 # Number of major CLEAN cycles to be done
        o.Qpix =  2.5 # Quality factor of synthesised beam oversampling
        o.Nf_out  = 500
        o.Tobs  = 6 * u.hours
        o.Nf_no_smear  = log(wl_max/wl_min) / log(3*wl/(2*Bmax_bin)/(Theta_fov*Qbw)+1)
        o.Rrp  = Nfacet**2 * 50 * Npix_linear**2 / Tsnap #(Consistent with PDR05 280115)
        o.Nf_used  = log(wl_max/wl_min) / log(3*wl/(2*Bmax_bin)/(Theta_fov*Qbw)+1) #Number of channels for gridding at longest baseline

    elif mode == 'Spectral':
        o.Qfov = 1.0 # Field of view factor
        o.Nmajor = 1 # Number of major CLEAN cycles to be done
        o.Qpix = 2.5 # Quality factor of synthesised beam oversampling
        o.Nf_out  = Nf_max #The same as the number of channels
        o.Nf_no_smear  = log(wl_max/wl_min) / log(3*wl/(2*Bmax_bin)/(Theta_fov*Qbw)+1)
        o.Nf_used  = Nf_max
        o.Tobs  = 6 * u.hours
        o.Rrp  = Nfacet**2 * 50 * Npix_linear**2 / Tsnap #(Consistent with PDR05 280115)

    elif mode == 'SlowTrans':
        o.Qfov = 0.9 # Field of view factor
        o.Nmajor = 1 # Number of major CLEAN cycles to be done
        o.Qpix = 1.5 # Quality factor of synthesised beam oversampling
        o.Nf_out  = 500  # Initially this value was computed (see line above) but Rosie has since specified that it should just be set to 500.
        o.Nf_used  = log(wl_max/wl_min) / log(3*wl/(2*Bmax_bin)/(Theta_fov*Qbw)+1) #Number of bands for gridding at longest baseline
        o.Tobs  = 1.2 * u.s  # Used to be equal to Tdump but after talking to Rosie set this to 1.2 sec
        o.Nf_no_smear  = log(wl_max/wl_min) / log(3*wl/(2*Bmax_bin)/(Theta_fov*Qbw)+1)
        o.Rrp  = 0 * Tsnap #(Consistent with PDR05 280115)

def apply_hpso_parameters(o, hpso):
    if hpso == '01':
        o.telescope = 'SKA1_Low'
        o.mode      = 'Continuum'
        o.freq_min  =  50e6 * u.Hz
        o.freq_max  = 200e6 * u.Hz
        o.Nbeam     = 2, #using 2 beams as per HPSO request...
        o.Nf_out    = 1500, # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
        o.Tobs      = 6 * u.hours
        o.Nf_max    = 256000
        o.Bmax      = 100 * u.kilometer
        o.Texp      = 2500 * u.hours
        o.Tpoint    = 1000 * u.hours
    elif hpso == '02A':
        o.telescope = 'SKA1_Low'
        o.mode      = 'Continuum'
        o.freq_min  =  50e6 * u.Hz
        o.freq_max  = 200e6 * u.Hz
        o.Nbeam     = 2, #using 2 beams as per HPSO request...
        o.Tobs      = 6 * u.hours
        o.Nf_max    = 256000
        o.Nf_out    = 1500, # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
        o.Bmax      = 100 * u.kilometer
        o.Texp      = 2500 * u.hours
        o.Tpoint    = 100 * u.hours
    elif hpso == '02B':
        o.telescope = 'SKA1_Low'
        o.mode      = 'Continuum'
        o.freq_min  =  50e6 * u.Hz
        o.freq_max  = 200e6 * u.Hz
        o.Nbeam     = 2, #using 2 beams as per HPSO request...
        o.Tobs      = 6 * u.hours
        o.Nf_max    = 256000
        o.Nf_out    = 1500, # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
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
        o.Tpoint    = (10/60.0) * u.hours
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
        o.Tpoint    = (10/60.0) * u.hours
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
        o.Nf_max    = 3200, #Assume 500 in continuum as well - defualt.
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
        o.Nf_max    = 5000, #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
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
        o.Nf_max    = 2500, #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
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
        o.Nf_out    = 4000, #4000 channel continuum observation - band 5.
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
        o.Nf_out    = 500, #continuum experiment with 500 output channels
        o.Bmax      = 50 * u.kilometer
        o.Texp      = 17500 * u.hours
        o.Tpoint    = 10 * u.hours
    elif hpso == '33':
        o.telescope = 'SKA1_Survey'
        o.mode      = 'Continuum'
        o.freq_min  =  1.0e9 * u.Hz
        o.freq_max  =  1.5e9 * u.Hz
        o.Tobs      = 6 * u.hours
        o.Nf_out    = 500, #continuum experiment with 500 output channels
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
        o.Nf_out    = 700, #700 channels required in output continuum cubes
        o.Bmax      = 200 * u.kilometer
        o.Texp      = 2000 * u.hours
        o.Tpoint    = 95 * u.hours
    elif hpso == '37b':
        o.telescope = 'SKA1_Mid'
        o.mode      = 'Continuum'
        o.freq_min  =  1e9 * u.Hz
        o.freq_max  =  1.7e9 * u.Hz
        o.Tobs      = 6 * u.hours
        o.Nf_out    = 700, #700 channels required in output continuum cubes
        o.Bmax      = 200 * u.kilometer
        o.Texp      = 2000 * u.hours
        o.Tpoint    = 2000 * u.hours
    elif hpso == '37c':
        o.telescope = 'SKA1_Survey'
        o.mode      = 'Continuum'
        o.freq_min  =  1e9 * u.Hz
        o.freq_max  =  1.5e9 * u.Hz
        o.Tobs      = 6 * u.hours
        o.Nf_out    = 500, #500 channels in output cube
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