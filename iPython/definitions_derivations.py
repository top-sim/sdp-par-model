from env_setup import *

Rflop, Mbuf_vis, Rio, Gcorr, Rphrot = symbols("R_flop, M_buf\,vis, R_io G_corr R_phrot", positive=True)

Omega_E, R_Earth = symbols("Omega_E, R_Earth", positive=True)

Naa, Nmm, Npp, Nw = symbols("N_k N_mm N_pp N_w", integer=True, positive=True)
Tion = symbols("T_ion", positive=True)
epsilon_w = symbols("\epsilon_w", positive=True)
Mvis = symbols("M_vis", positive=True)
Qbw, Qw, Qgcf, Qfcv = symbols("Q_bw, Q_w, Q_GCF, Q_fcv", positive=True)

Blim_mid = symbols("B_lim\,mid", positive=True)
Qfov = symbols("Q_FoV", positive=True)

Nmajor, Nf_used, Nf_out, Nf_max, Nf_no_smear = symbols("N_major N_f\,used N_f\,out N_f\,max N_f\,no-smear", integer=True, positive=True)
Npix_linear, Nminor, Nfacet = symbols("N_pix\,linear N_minor N_facet", integer=True, positive=True)
Na, Nbeam = symbols("N_a N_beam", integer=True, positive=True)
Ngw = symbols("N_gw", positive=True)

Tobs = symbols("T_obs", positive=True)
Tsnap_min = symbols("T_snap\,min", positive=True)
Qpix = symbols("Q_pix", positive=True)

Ds = symbols("D_s", positive=True)
Bmax, Bmax_bin = symbols("B_max Bmax\,bin", positive=True)
freq_min, freq_max = symbols("f_min f_max")

# These two variables are for computing baseline-dependent variables (by approximating baseline distribution as a series of bins)
baseline_bins  = symbols("B_bins", positive=True)
baseline_bin_counts = symbols("B_bin_counts", positive=True, integer=True)
binfrac = symbols("f_bin", positive=True) # The fraction of baselines that fall in a given bin (used for baseline-dependent calculations)

Tdump_ref = symbols("T_dump\,ref", positive=True)
Qw2, Qw32 = symbols("Q_w2 Q_w32", positive=True)

# Wavelength variables, not (yet) enumerated in the table above
wl, wl_max, wl_min, wl_sub_max, wl_sub_min = symbols("\lambda \lambda_max \lambda_max \lambda_{sub\,max} \lambda_{sub\,min}", positive=True)

# Variables unique to the HPSO experiments
Texp, Tpoint = symbols("T_exp T_point", positive=True)

# Other variables (may be needed later on)
Tsnap, Tsnap_min = symbols("T_snap T_{snap\,min}", positive=True)  # Snapshot timescale implemented
Tdump = symbols("T_dump", positive=True)  # Correlator dump time (s)
psf_angle= symbols(r"\theta_{PSF}", positive=True)
pix_angle  = symbols(r"\theta_{pix}", positive=True)
Theta_beam = symbols(r"\theta_{beam}", positive=True)
Theta_fov = symbols(r"\theta_{FoV}", positive=True)
Rrp = symbols("R_rp", positive=True) # Reprojection Flop rate, per output channel

'''
Parameters derived in terms of those above:
'''
Tdump = Min(Tdump_ref * floor(Bmax / Bmax_bin), 1.2 * u.s) # Correlator dump time; limit this at 1.2s maximum
wl_max = u.c / freq_min             # Maximum Wavelength
wl_min = u.c / freq_max             # Minimum Wavelength
wl = 0.5*(wl_max + wl_min)          # Representative Wavelength
Theta_fov = 7.66 * wl * Qfov / (pi * Ds * Nfacet) # added Nfacet dependence
Theta_beam = 3 * wl/(2*Bmax) #bmax here is for the overall experiment (so use Bmax), not the specific bin...
Theta_pix = Theta_beam/(2*Qpix)
Npix_linear = Theta_fov / Theta_pix  # The linear number of pixels along the image's side (assumed to be square)
Rfft = Nfacet**2 * 5 * Npix_linear**2 * log(Npix_linear,2) / Tsnap # added Nfacet dependence

Qw2  = 1  # Obsolete variable (ask Rosie about what it meant)
Qw32 = 1  # Obsolete variable (ask Rosie about what it meant)
DeltaW_max = Qw * Max(Bmax_bin*Tsnap*Omega_E/(2*wl), Bmax_bin**2/(8*R_Earth*wl)) #W deviation catered for by W kernel, in units of typical wavelength, for the specific baseline bin being considered
Ngw = 2*Theta_fov * sqrt((Qw2 * DeltaW_max**2 * Theta_fov**2/4.0)+(Qw32 * DeltaW_max**1.5 * Theta_fov/(epsilon_w*2*pi)))
Ncvff = Qgcf*sqrt(Naa**2+Ngw**2)

#Nf_vis=(Nf_out*Fb_short)+(Nf_used*(1-Fb_short-Fb_mid))+(Nf_no_smear*Fb_mid) #no dependence on nfacet. (new: just use Nf_used)
#Nf_vis= Max(Nf_out,Nf_used) #need to take max here so that gridding visibilities cannot be put on channels which are too coarse

Nf_vis=(Nf_out*sign(floor(Nf_out/Nf_no_smear)))+(Nf_no_smear*sign(floor(Nf_no_smear/Nf_out))) #Boom! Workaround to avoid recursive errors...

#Nf_vis=Nf_no_smear
Nvis = binfrac*Na*(Na-1)*Nf_vis/(2*Tdump) * u.s # Number of visibilities per second to be gridded (after averaging short baselines to coarser freq resolution). Note multiplication by u.s to get rid of /s
Rgrid = Nfacet*8*Nmm*Nvis*(Ngw**2+Naa**2) #added Nfacet dependence. Linear becuase tehre are Nfacet^2 facets but can integrate Nfacet times longer at gridding as fov is lower.

Rccf = Nfacet**2 * 5 * binfrac *(Na-1)*Na*Nmm*Ncvff**2 * log(Ncvff,2)/(Tion*Qfcv) #reduce by multiplication by binfrac (RCB), add in extra multiplication by Nfacet-squared.

Rphrot = 2 * Nmajor * Npp * Nbeam * Nvis * Nfacet**2 * 25 * sign(Nfacet-1)  # Last factor ensures that answer is zero if Nfacet is 1.

#Rrp is handled in the telescope parameters file

Rflop = Rphrot + 2 * Nmajor*Nbeam*Npp*(Nf_out*(Rrp + Rfft) + (Nf_vis*Rccf) + Rgrid) # Overall flop rate
Mbuf_vis = 2 * Mvis * Nbeam * Npp * Nvis * Tobs / u.s # Note the division by u.s to get rid of pesky SI unit. Also note the factor 2 -- we have a double buffer (allowing storage of a full observation while simultaneously capturing the next)
Rio = Mvis * Nmajor * Nbeam * Npp * Nvis * Nfacet**2 #added Nfacet dependence

Gcorr = Na * (Na - 1) * Nf_max * Nbeam * Nw * Npp / Tdump  # Minimum correlator output data rate, after baseline dependent averaging 

# Split the FLOP rate into constituent parts, for plotting
Rflop_common_factor = 2 * Nmajor * Nbeam * Npp
Rflop_grid = Rflop_common_factor * Rgrid
Rflop_conv = Rflop_common_factor * Nf_vis * Rccf #changed to be consistent in using Nf_vis .
Rflop_fft  = Rflop_common_factor * Nf_out  * Rfft
Rflop_proj = Rflop_common_factor * Nf_out  * Rrp
Rflop_phrot = Rphrot