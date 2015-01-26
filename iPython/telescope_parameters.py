from definitions_derivations import *
#from sympy import symbols, pi, log, ln, Max, sqrt, sign

universal = {Omega_E : 7.292115e-5,  # In PDR05 Excel sheet a value of 0.0000727 was used. This value based on rotation relative to the fixed stars,
             R_Earth : const.R_earth.value * u.m, # In PDR05 Excel sheet a value of 6400000 was used,
             Naa   : 9,
             Npp  : 4,
             Nmm  : 4,
             Nf_max: 256000,        # maximum number of channels (not actually universal, as can be lower depending on spectral line requirements)
             Tion : 60.0,
             epsilon_w : 0.01,
             Mvis : 12.0,
             Qbw  : 1.0,
             Qw   : 1.0,
             Qgcf : 8.0,
             Qfcv : 10.0,
             Tsnap_min : 1.2,
             Blim_mid : 20000, # Baseline length that(Fbshort+Fbmid, i.e. 90%) of baselines are shorter than.
             Nfacet : 1  # THIS VALUE SHOULD BE DYNAMICALLY MINIMIZED - THIS IS A TEST! TODO.
             # ** Is this value valid ?? Especially for all telescopes, as they have different baselines ** ??? 
             }  

# The three telescopes, along with their unique characteristics
telescope_info = {
    'SKA1_Low': {
        Qw2  : 1, #0.0458053,  # Weighing value for baseline length distribution (Ask Rosie Bolton for interpretation)
        Qw32 : 1, #0.0750938,  # Weighing value for baseline length distribution (Ask Rosie Bolton for interpretation)
        Tdump_ref: 0.6* u.s, # Correlator dump time in reference design
        Na: 1024,            # number of antennas
        Nbeam: 1,            # number of beams
        Ds: 35 * u.m,        # station "diameter" in meters
        Bmax: 100 * u.km,     # Actually constructed kilometers of max baseline
        Bmax_ref: 100 * u.km, # kilometers of max baseline in reference design
        baseline_bins : np.array((4.9, 7.1, 10.4, 15.1, 22.1, 32.2, 47.0, 68.5, 100)) * u.km,
        baseline_bin_counts : np.array((5031193, 732481, 796973, 586849, 1070483, 939054, 820834, 202366, 12375)),
        Fb_short_tel : 0.5,   # Fraction of baselines short enough for data to be averaged to frequency
                              # resolution of output prior to gridding. Only relevant for continuum modes
    },
    'SKA1_Mid': {  # Assuming band 1, for the moment. TODO: allow all bands to be computed.
        Qw2  : 1, #0.0278115,  # Weighing value for baseline length distribution (Ask Rosie Bolton for interpretation)
        Qw32 : 1, #0.0462109,  # Weighing value for baseline length distribution (Ask Rosie Bolton for interpretation)
        Tdump_ref: 0.08 * u.s, # Correlator dump time in reference design
        Na: 190+64,            # number of antennas
        Nbeam: 1,          # number of beams
        Ds: 15 * u.m,      # station "diameter" in meters
        Bmax: 200 * u.km,     # Actually constructed kilometers of max baseline
        Bmax_ref: 200 * u.km, # kilometers of max baseline in reference design
        baseline_bins : np.array((4.4, 6.7, 10.3, 15.7, 24.0, 36.7, 56.0, 85.6, 130.8, 200)) * u.km,
        baseline_bin_counts : np.array((669822, 61039, 64851, 66222, 70838, 68024, 74060, 68736, 21523, 745)),
        Fb_short_tel : 0.5, # Fraction of baselines short enough for data to be averaged to frequency
                            # resolution of output prior to gridding. Only relevant for continuum modes
    },
    'SKA1_Survey': { # Assuming band 1, for the moment. TODO: allow all bands to be computed.
        Qw2  : 1, #0.0569392,  # Weighing value for baseline length distribution (Ask Rosie Bolton for interpretation)
        Qw32 : 1, #0.0929806,  # Weighing value for baseline length distribution (Ask Rosie Bolton for interpretation)
        Tdump_ref: 0.3 * u.s, # Correlator dump time in reference design
        Na: 96,               # number of antennas
        Nbeam: 36,            # number of beams (36 because this is a PAF)
        Ds: 15 * u.m,         # station "diameter" in meters
        Bmax: 50 * u.km,     # Actually constructed kilometers of max baseline
        Bmax_ref: 50 * u.km, # kilometers of max baseline in reference design
        baseline_bins : np.array((3.8, 5.5, 8.0, 11.5, 16.6, 24.0, 34.6, 50)) * u.km,
        baseline_bin_counts : np.array((81109, 15605, 15777, 16671, 16849, 17999, 3282, 324)),
        Fb_short_tel : 0.1,  # Fraction of baselines short enough for data to be averaged to frequency
                             # resolution of output prior to gridding. Only relevant for continuum modes
    },
}

# The 'standard' bands
band_info = {
    'Low' : {
        'telescope' : 'SKA1_Low',
        freq_min :  50e6 * u.Hz,
        freq_max : 350e6 * u.Hz,
    },
    'Mid1' : {
        'telescope' : 'SKA1_Mid',
        freq_min :  350e6 * u.Hz,
        freq_max : 1.05e9 * u.Hz,
    },
    'Mid2' : {
        'telescope' : 'SKA1_Mid',
        freq_min : 949.367e6 * u.Hz,
        freq_max : 1.7647e9 * u.Hz,
    },
    'Mid3' : {
        'telescope' : 'SKA1_Mid',
        freq_min : 1.65e9 * u.Hz,
        freq_max : 3.05e9 * u.Hz,
    },
    'Mid4' : {
        'telescope' : 'SKA1_Mid',
        freq_min : 2.80e9 * u.Hz,
        freq_max : 5.18e9 * u.Hz,
    },
    'Mid5A' : {
        'telescope' : 'SKA1_Mid',
        freq_min : 4.60e9 * u.Hz,
        freq_max : 7.10e9 * u.Hz,
    },
    'Mid5B' : {
        'telescope' : 'SKA1_Mid',
        freq_min : 11.3e9 * u.Hz,
        freq_max : 13.8e9 * u.Hz,
    },
    'Sur1' : {
        'telescope' : 'SKA1_Survey',
        freq_min : 350e6 * u.Hz,
        freq_max : 850e6 * u.Hz,
    },
    'Sur2A' : {
        'telescope' : 'SKA1_Survey',
        freq_min :  650e6 * u.Hz,
        freq_max : 1.35e9 * u.Hz,
    },
    'Sur2B' : {
        'telescope' : 'SKA1_Survey',
        freq_min : 1.17e9 * u.Hz,
        freq_max : 1.67e9 * u.Hz,
    },
    'Sur3A' : {
        'telescope' : 'SKA1_Survey',
        freq_min : 1.5e9 * u.Hz,
        freq_max : 2.0e9 * u.Hz,
    },
    'Sur3B' : {
        'telescope' : 'SKA1_Survey',
        freq_min : 3.5e9 * u.Hz,
        freq_max : 4.0e9 * u.Hz,
    },
}

# Imaging Modes
imaging_mode_info = {
    'Continuum': {
        Qfov:  1.8, # Field of view factor
        Nmajor: 10, # Number of major CLEAN cycles to be done
        Qpix:  2.5, # Quality factor of synthesised beam oversampling
        Nf_max: 256000, 
        Nf_out : 500,
        Fb_short : 0* Fb_short_tel,
        Tobs : 6 * u.hours,
        Fb_mid  : 1-0.1-Fb_short,
        Nf_no_smear : log(wl_max/wl_min) / log(3*(wl/u.m) /(2*Blim_mid)/(Theta_fov*Qbw)+1) ,
        Rrp : 50 * Npix**2 / Tsnap,
        Nf_used : log(wl_max/wl_min) / log(3*(wl/u.m) /(2*Blim_mid)/(Theta_fov*Qbw)+1), #Number of channels for gridding at longest baseline

    }, 
    'Spectral': {
        Qfov: 1.0, # Field of view factor
        Nmajor: 1, # Number of major CLEAN cycles to be done        
        Qpix: 2.5, # Quality factor of synthesised beam oversampling
        Nf_out : Nf_max, #The same as the number of channels
        Nf_no_smear : log(wl_max/wl_min) / log(3*(wl/u.m) /(2*Blim_mid)/(Theta_fov*Qbw)+1) ,
        Nf_used : Nf_max,
        Fb_short : 0 * Fb_short_tel, # Need a symbolic expression to be able to be substuted; hence multiply by 0 
        Tobs : 6 * u.hours,
        Fb_mid : Fb_short,
        Rrp : 50 * Npix**2 / Tsnap,
    },
    'SlowTrans': {
        Qfov: 0.9, # Field of view factor
        Nmajor: 1, # Number of major CLEAN cycles to be done
        Qpix: 1.5, # Quality factor of synthesised beam oversampling
        Nf_out : 500,  # Initially this value was computed (see line above), but Rosie has since specified that it should just be set to 500.
        Nf_used : ln(wl_max / wl_min) / ln((Theta_beam/(Theta_fov*Qbw)+1)), #Number of bands for gridding at longest baseline
        Fb_short : 0 * Fb_short_tel,
        Tobs : 1.2 * u.s,  # Used to be equal to Tdump, but after talking to Rosie set this to 1.2 sec
        Fb_mid  : 1-0.1-Fb_short,
        Nf_no_smear : log(wl_max/wl_min) / log(3*(wl/u.m)/(2*Blim_mid)/(Theta_fov*Qbw)+1),
        Rrp : 0 * Tsnap,
    },
}

## High Priority Science Objectives (overwrite any previously defined settings)
hpsos = {
    '01' : {
        'telescope' : 'SKA1_Low',
        'mode'      : 'Continuum',
        freq_min  :  50e6 * u.Hz,
        freq_max  : 200e6 * u.Hz,
        Nbeam     : 2, #using 2 beams as per HPSO request...
        Nf_out    : 1500, # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
        Tobs      : 6 * u.hours,
        Nf_max    : 256000,
        Bmax      : 100 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 2500 * u.hours,
        Tpoint    : 1000 * u.hours,
    },
    '02A' : {
        'telescope' : 'SKA1_Low',
        'mode'      : 'Continuum',
        freq_min  :  50e6 * u.Hz,
        freq_max  : 200e6 * u.Hz,
        Nbeam     : 2, #using 2 beams as per HPSO request...
        Tobs      : 6 * u.hours,
        Nf_max    : 256000,
        Nf_out    : 1500, # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
        Bmax      : 100 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 2500 * u.hours,
        Tpoint    : 100 * u.hours,
    },
    '02B' : {
        'telescope' : 'SKA1_Low',
        'mode'      : 'Continuum',
        freq_min  :  50e6 * u.Hz,
        freq_max  : 200e6 * u.Hz,
        Nbeam     : 2, #using 2 beams as per HPSO request...
        Tobs      : 6 * u.hours,
        Nf_max    : 256000,
        Nf_out    : 1500, # 1500 channels in output - simpler to just run as a continuum experiment - though an alternative would be to run as CS mode with 500+1500 channels
        Bmax      : 100 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 2500 * u.hours,
        Tpoint    : 10 * u.hours,
    },
    '03A' : {
        'telescope' : 'SKA1_Low',
        'mode'      : 'SlowTrans',
        'comment' : 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate',
        freq_min  : 150e6 * u.Hz,
        freq_max  : 350e6 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Bmax      : 100 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 12800 * u.hours,
        Tpoint    : 0.17 * u.hours,
        Nmajor    : 10,
    },
    '03B' : {
        'telescope' : 'SKA1_Low',
        'mode'      : 'SlowTrans',
        'comment' : 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate',
        freq_min  : 150e6 * u.Hz,
        freq_max  : 350e6 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Bmax      : 100 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 4300 * u.hours,
        Tpoint    : 0.17 * u.hours,
        Nmajor    : 10,
    },
    '04A' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'SlowTrans',
        'comment' : 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.',
        freq_min  : 650e6 * u.Hz,
        freq_max  : 950e6 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Bmax      : 10 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 800 * u.hours,
        Tpoint    : (10/60.0) * u.hours,
        Nmajor    : 10,
    },
    '04B' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'SlowTrans',
        'comment' : 'Pulsar Search. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.',
        freq_min  : 1.25e9 * u.Hz,
        freq_max  : 1.55e9 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Bmax      : 10 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 800 * u.hours,
        Tpoint    : (10/60.0) * u.hours,
        Nmajor    : 10,
    },
    '05A' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'SlowTrans',
        'comment' : 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.',
        freq_min  : 0.95e9 * u.Hz,
        freq_max  : 1.76e9 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Bmax      : 15 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 1600 * u.hours,
        Tpoint    : (10/60.0) * u.hours,
        Nmajor    : 10,
    },
    '05B' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'SlowTrans',
        'comment' : 'Pulsar Timing. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 15km for real time calibration, allowing 10 major cycles.',
        freq_min  : 1.65e9 * u.Hz,
        freq_max  : 3.05e9 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Bmax      : 15 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 1600 * u.hours,
        Tpoint    : (10/60.0) * u.hours,
        Nmajor    : 10,
    },
    '13' : {
        'telescope' : 'SKA1_Survey',
        'mode'      : 'CS',
        'comment' : 'HI, limited BW',
        freq_min  : 790e6 * u.Hz,
        freq_max  : 950e6 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_max    : 3200, #Assume 500 in continuum as well - defualt.
        Bmax      : 40 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 5000 * u.hours,
        Tpoint    : 2500 * u.hours,
    },
    '14' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'CS',
        'comment' : 'HI',
        freq_min  :  1.3e9 * u.Hz,
        freq_max  :  1.4e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_max    : 5000, #Only 5,000 spectral line channels. Assume 500 - default - for continuum as well.
        Bmax      : 200 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 2000 * u.hours,
        Tpoint    : 10 * u.hours,
    },
    '15' : {
        'telescope' : 'SKA1_Survey',
        'mode'      : 'CS',
        'comment' : 'HI, limited spatial resolution',
        freq_min  :  1.415e9 * u.Hz,
        freq_max  :  1.425e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_max    : 2500, #Only 2,500 spectral line channels. Assume 500 - default - for continuum as well.
        Bmax      : 13 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 2000 * u.hours,
        Tpoint    : 10 * u.hours,
    },
    '19' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'SlowTrans',
        'comment' : 'Transients. Real time calibration loading; assume this is like SlowTrans FLOP rate. Assuming using only baselines out to 10km for real time calibration, allowing 10 major cycles.',
        freq_min  : 650e6 * u.Hz,
        freq_max  : 950e6 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Bmax      : 10 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 10000 * u.hours,
        Tpoint    : (10/60.0) * u.hours,
        Nmajor    : 10,
    },
    '22' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'Continuum',
        'comment' : 'Cradle of life',
        freq_min  :  10e9 * u.Hz,
        freq_max  :  12e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_max    : 4000,
        Nf_out    : 4000, #4000 channel continuum observation - band 5.
        Bmax      : 200 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 6000 * u.hours,
        Tpoint    : 600 * u.hours,
    },
    '27' : {
        'telescope' : 'SKA1_Survey',
        'mode'      : 'Continuum',
        freq_min  :  1.0e9 * u.Hz,
        freq_max  :  1.5e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_max    : 256000,
        Nf_out    : 500, #continuum experiment with 500 output channels
        Bmax      : 50 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 17500 * u.hours,
        Tpoint    : 10 * u.hours,
    },
    '33' : {
        'telescope' : 'SKA1_Survey',
        'mode'      : 'Continuum',
        freq_min  :  1.0e9 * u.Hz,
        freq_max  :  1.5e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_out    : 500, #continuum experiment with 500 output channels
        Bmax      : 50 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 17500 * u.hours,
        Tpoint    : 10 * u.hours,
    },
    '35' : {
        'telescope' : 'SKA1_Survey',
        'mode'      : 'SlowTrans',
        'comment' : 'Autocorrelation',
        freq_min  :   650e6 * u.Hz,
        freq_max  :  1.15e9 * u.Hz,
        Tobs      : 0 * u.hours,
        Nf_max    : 256000,
        Nf_out    : 500,
        Bmax      : 10 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 5500 * u.hours,
        Tpoint    : 3.3 * u.hours,
    },
    '37a' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'Continuum',
        freq_min  :  1e9 * u.Hz,
        freq_max  :  1.7e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_out    : 700, #700 channels required in output continuum cubes
        Bmax      : 200 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 2000 * u.hours,
        Tpoint    : 95 * u.hours,
    },
    '37b' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'Continuum',
        freq_min  :  1e9 * u.Hz,
        freq_max  :  1.7e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_out    : 700, #700 channels required in output continuum cubes
        Bmax      : 200 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 2000 * u.hours,
        Tpoint    : 2000 * u.hours,
    },
    '37c' : {
        'telescope' : 'SKA1_Survey',
        'mode'      : 'Continuum',
        freq_min  :  1e9 * u.Hz,
        freq_max  :  1.5e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_out    : 500, #500 channels in output cube
        Bmax      : 50 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 5300 * u.hours,
        Tpoint    : 95 * u.hours,
    },
    '38a' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'Continuum',
        freq_min  :  7e9 * u.Hz,
        freq_max  :  11e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_out    : 1000,
        Bmax      : 200 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 1000 * u.hours,
        Tpoint    : 16.4 * u.hours,
    },
    '38b' : {
        'telescope' : 'SKA1_Mid',
        'mode'      : 'Continuum',
        freq_min  :  7e9 * u.Hz,
        freq_max  :  11e9 * u.Hz,
        Tobs      : 6 * u.hours,
        Nf_out    : 1000,
        Bmax      : 200 * u.kilometer,
        Fb_mid    : 0, # Check
        Texp      : 1000 * u.hours,
        Tpoint    : 1000 * u.hours,
    }
}