import sympy.physics.units as u
from astropy import constants as const
import numpy as np
from sympy import symbols

class symbolic_definitions:
    @staticmethod
    def define_symbolic_variables(o):
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