import sympy.physics.units as u
from sympy import log, Min, Max, sqrt, sign, ceiling, floor
from numpy import pi
from parameter_definitions import ImagingModes

class Formulae:
    def __init__(self):
		pass
    @staticmethod
    def compute_derived_parameters(telescope_parameters, imaging_mode, BL_dep_time_av=False, On_the_fly=0, verbose=False):
        """
                Computes a host of important values from the originally supplied telescope parameters, using the parametric
		equations. These equations are based on the PDR05 document
		@param o:
		@param imaging_mode:
		@param BL_dep_time_av: True iff baseline dependent time averaging should be used.
		@param verbose: displays verbose command-line output
		@raise Exception:
        """
        o = telescope_parameters  # Used for shorthand in the equations below

        # ===============================================================================================
        # PDR05 Sec 9.2
        #
        #
        # ===============================================================================================
        
        o.wl_max = u.c / o.freq_min             						# Maximum Wavelength
        o.wl_min = u.c / o.freq_max             						# Minimum Wavelength
        o.wl = 0.5*(o.wl_max + o.wl_min)          						# Representative Wavelength
        o.Theta_fov = 7.66 * o.wl * o.Qfov / (pi * o.Ds * o.Nfacet)		# Facet Field-of-view **PDR05 uses lambda_max, not mean
        o.Theta_beam = 3. * o.wl/(2.*o.Bmax)								# Synthesized beam
        o.Theta_pix = o.Theta_beam/(2.*o.Qpix) 							# Pixel size
        o.Npix_linear = o.Theta_fov / o.Theta_pix 						# Number of pixels on side of facet
        o.epsilon_f_approx = sqrt(6.*(1.-(1.0/o.amp_f_max))) 				# expansion of sine to solve epsilon = arcsinc(1/amp_f_max).
        #print "Epsilon approx :", o.epsilon_f_approx
        o.Qbw=1.47/o.epsilon_f_approx
        if verbose:
            print "Image Characteristics:"
            print "----------------------"
            print ""
            print "Facet FOV: ",o.Theta_fov," rads"
            print "PSF size:  ",o.Theta_beam," rads"
            print "Pixel size:",o.Theta_pix," rads"
            print "No. pixels on facet side:",o.Npix_linear
            print "Found Qbw =", o.Qbw, "and cell frac error, epsilon =", o.epsilon_f_approx
            print "----------------------"
            print ""

    # ===============================================================================================
        print "Found Qbw = %8.3f, and cell frac error, epsilon,  %8.3f"  % (o.Qbw, o.epsilon_f_approx)
     
       
# ===============================================================================================
# PDR05 Sec 9.1
#
#
# ===============================================================================================
        

        o.Nf_no_smear_predict = log(o.wl_max/o.wl_min) / log((3.*o.wl/(2.*o.Bmax)/(o.Theta_fov*o.Nfacet*o.Qbw))+1.) # Eq. 4 for full FOV, at max baseline
        o.Nf_no_smear_backward = log(o.wl_max/o.wl_min) / log((3.*o.wl/(2.*o.Bmax_bin)/(o.Theta_fov*o.Qbw))+1.)			# Eq. 4 for facet FOV. Ensures correct location of visibility on grid.

        o.Tdump_scaled = o.Tdump_ref * o.B_dump_ref / o.Bmax #get correlator output averaging time as scaled for max baseline.
        o.combine_time_samples = Max(floor((o.epsilon_f_approx * o.wl/(o.Theta_fov * o.Nfacet * o.Omega_E * o.Bmax_bin) * u.s) / o.Tdump_scaled), 1)
        o.Tdump_skipper=o.Tdump_scaled * o.combine_time_samples
        if BL_dep_time_av:
            o.Tdump_predict = Min(o.Tdump_skipper, 1.2 * u.s, o.Tion * u.s) # use whatever is smaller. Don't let any BLdep averaging be for longer than 1.2s or Tion.
            o.Tdump_backward = Min(o.Tdump_skipper*o.Nfacet, o.Tion * u.s) #For backward step at gridding only, allow coalescance of visibility points at Facet FoV smearing limit only for BLDep averaging case.
        else:
            o.Tdump_predict = o.Tdump_scaled
            o.Tdump_backward = o.Tdump_scaled
        if verbose:
			print "Channelization Characteristics:"
			print "-------------------------------"
			print ""
			print "Ionospheric timescale: ", o.Tion," sec"
			print "T_dump predict: ", o.Tdump_predict," sec"
			print "T_dump backward: ", o.Tdump_backward," sec"
			print ""
			print "No. freq channels for predict: ",o.Nf_no_smear_predict
			print "No. freq channels for backward step: ",o.Nf_no_smear_backward
			print ""
			if BL_dep_time_av:
				print "USING BASELINE DEPENDENT TIME AVERAGING, combining this number of time samples: ", o.combine_time_samples
			else:
				print "NOT IMPLEMENTING BASELINE DEPENDENT TIME AVERAGING"
			print ""
			print "------------------------------"
			print ""
            
# ===============================================================================================
        if BL_dep_time_av:
            print "USING BASELINE DEPENDENT TIME AVERAGING, combining this number of time samples: ", o.combine_time_samples
        else:
            print "NOT IMPLEMENTING BASELINE DEPENDENT TIME AVERAGING"
        if On_the_fly==1:
            print "On the fly kernels..."
        else:
            print "Not on the fly kernels..."
    
		

# ===============================================================================================
# PDR05 Sec 12.2 - 12.5
#
#
# ===============================================================================================
        o.DeltaW_Earth = o.Bmax_bin**2/(o.R_Earth*o.wl*8.) # Eq. 19
        o.DeltaW_SShot = o.Bmax_bin*o.Tsnap*o.Omega_E/(o.wl*2.) # Eq. 26 w-deviation for snapshot **PDR05 uses lambda_min, not mean
        o.DeltaW_max = o.Qw * Max(o.DeltaW_SShot, o.DeltaW_Earth)
        # Qw = (0,1) allows for not maximum dev. corrected
        o.Ngw = 2.*o.Theta_fov * sqrt((o.DeltaW_max**2 * o.Theta_fov**2/4.0)+(o.DeltaW_max**1.5 * o.Theta_fov/(o.epsilon_w*pi*2.))) # Eq. 25 w-kernel support size **Note difference in cellsize assumption**
        if On_the_fly==1:
            o.Ncvff = sqrt(o.Naa**2+o.Ngw**2) #Test: what happens if we calculate convolution functions on the fly for each visibility? Make smaller kernels, no reuse.
        else:
            o.Ncvff = o.Qgcf*sqrt(o.Naa**2+o.Ngw**2)
    
        # Eq. 23 combined kernel support size

        if verbose:
            print "Geometry Assumptions:"
            print "-------------------------------"
            print ""
            print "Delta W Earth: ", o.DeltaW_Earth," lambda"
            print "Delta W Snapshot: ", o.DeltaW_SShot," lambda"
            print "Delta W max: ",o.DeltaW_max," lambda"
            print ""
            print "------------------------------"
            print ""
            print "Kernel Sizes:"
            print "-------------------------------"
            print ""
            print "Support of w-kernel: ",o.Ngw," pixels"
            print "Support of combined GCF: ",o.Ncvff," sub-pixels"
            print ""
            print "------------------------------"
            print ""
        if On_the_fly ==1:
                print "WARNING! Experimental!: On the fly kernels in use. (Set On_the_fly =0 to disable)"
                print "On the fly kernels is a new option forcing convolution kernels to be recalculated"
                print "for each and every viibility point, but only at the actual size needed  - i.e. not"
                print "oversampled by a factor of Qgcf (8)."

# ===============================================================================================
        
        
# The following workaround is (still) needed. Note: gridding done at maximum of either Nf_out or Nf_no_smear.
        o.Nf_vis_backward=(o.Nf_out*sign(floor(o.Nf_out/o.Nf_no_smear_backward)))+(o.Nf_no_smear_backward*sign(floor(o.Nf_no_smear_backward/o.Nf_out)))
        o.Nf_vis_predict=(o.Nf_out*sign(floor(o.Nf_out/o.Nf_no_smear_predict)))+(o.Nf_no_smear_predict*sign(floor(o.Nf_no_smear_predict/o.Nf_out)))


# ===============================================================================================
# PDR05 Sec 12.8
#
# Eq. 30 says that R_flop = 2 x N_maj x N_pp x N_beam x ( R_grid + R_fft + R_rp + R_ccf)
#
# ===============================================================================================

        o.Nvis_backward = o.binfrac*o.Na*(o.Na-1)*o.Nf_vis_backward/(2.*o.Tdump_backward) * u.s 	# Eq. 31 Visibility rate for backward step, allow coalescing in time and freq prior to gridding
        o.Nvis_predict = o.binfrac*o.Na*(o.Na-1)*o.Nf_vis_predict/(2.*o.Tdump_predict) * u.s 		# Eq. 31 Visibility rate for predict step
        
        Rflop_common_factor = o.Nmajor * o.Nbeam * o.Npp # no factor 2 because forward & backward steps are both in Rflop numbers
        # Gridding:
        o.Rgrid_backward = 8.*o.Nmm*o.Nvis_backward*(o.Ngw**2+o.Naa**2)	# Eq 32
        o.Rgrid_predict = 8.*o.Nmm*o.Nvis_predict*(o.Ngw**2+o.Naa**2)	# Eq 32
        o.Rgrid = o.Rgrid_backward + o.Rgrid_predict
        o.Rflop_grid = Rflop_common_factor * o.Rgrid
            
		# Do FFT:
        if imaging_mode == ImagingModes.Continuum:
            o.Nf_FFT_backward = o.minimum_channels #make only enough FFT grids to extract necessary spectral info and retain distributability.
        elif imaging_mode == ImagingModes.Spectral:
            o.Nf_out = o.Nf_max
            o.Nf_FFT_backward = o.Nf_max
        elif imaging_mode == ImagingModes.SlowTrans:
            o.Nf_FFT_backward = o.minimum_channels
        else:
            raise Exception("Unknown Imaging Mode %s" % imaging_mode)

        o.Nf_FFT_predict=o.Nf_vis_predict
        o.Rfft_backward = o.binfrac * o.Nfacet**2 * 5. * o.Npix_linear**2 * log(o.Npix_linear,2) / o.Tsnap 	# Eq. 33, per output grid (i.e. frequency)
        o.Rfft_predict = o.binfrac * 5. * (o.Npix_linear*o.Nfacet)**2 * log((o.Npix_linear*o.Nfacet),2) / o.Tsnap 	# Eq. 33 per predicted grid (i.e. frequency)
        o.Rfft_intermediate_cycles = ( o.Rfft_backward * o.Nf_FFT_backward) + ( o.Rfft_predict*o.Nf_FFT_predict )
        o.Rfft_final_cycle = ( o.Rfft_backward * o.Nf_out) + ( o.Rfft_predict*o.Nf_FFT_predict )#final major cycle, create final data products (at Nf_out channels)
        o.Rflop_fft  = o.Nmajor * o.Nbeam * o.Npp * ((o.Nmajor-1)* o.Rfft_intermediate_cycles + o.Rfft_final_cycle) #do nmajor-1 cycles before doing the final major cycle.


		# Do re-projection for snapshots:
        if imaging_mode == ImagingModes.Continuum:
            o.Rrp = o.Nfacet**2 * 50. * o.Npix_linear**2 / o.Tsnap # Eq. 34
        elif imaging_mode == ImagingModes.Spectral:
			o.Rrp = o.Nfacet**2 * 50. * o.Npix_linear**2 / o.Tsnap # Eq. 34
        elif imaging_mode == ImagingModes.SlowTrans:
            o.Rrp = 0*u.s / u.s  # (Consistent with PDR05 280115)
        else:
            raise Exception("Unknown Imaging Mode %s" % imaging_mode)

        o.Rflop_proj = (o.Nbeam * o.Npp) * ((o.Nmajor-1) * o.Nf_FFT_backward + o.Nf_out) # Reproj intermetiate major cycle FFTs (Nmaj -1) times, then do the final ones for the last cycle at the full output spectral resolution.
        
		# Make GCF
        o.grid_cell_error=o.epsilon_f_approx
        o.dfonF=o.grid_cell_error/o.Qkernel/(o.Ncvff/o.Qgcf)
        o.Nf_gcf_backward_nosmear=log(o.wl_max/o.wl_min) / log(o.dfonF+1.) # allow uv positional errors up to grid_cell_error*1/Qkernel of a cell from frequency smearing.
        o.Nf_gcf_predict_nosmear=log(o.wl_max/o.wl_min) / log(o.dfonF+1.)  # allow uv positional errors up to grid_cell_error*1/Qkernel of a cell from frequency smearing.
        if On_the_fly ==1:
            o.Nf_gcf_backward=o.Nf_vis_backward
            o.Nf_gcf_predict=o.Nf_vis_predict
            o.Tkernel_backward=o.Tdump_backward/u.s
            o.Tkernel_predict=o.Tdump_predict/u.s

        else:
            o.Nf_gcf_backward=Max(o.Nf_gcf_backward_nosmear, o.minimum_channels) #maintain distributability, need at least minimum_channels (500) kernels.
            o.Nf_gcf_predict=Max(o.Nf_gcf_predict_nosmear, o.minimum_channels) #maintain distributability, need at least minimum_channels (500) kernels.
            o.Tkernel_backward=o.Tion
            o.Tkernel_predict=o.Tion


        if verbose:
            print "Number of kernels to cover freq axis is Nf_FFT_backward: ", o.Nf_gcf_backward
            print "Number of kernels to cover freq axis is Nf_FFT_predict: ", o.Nf_gcf_predict
                
        o.Rccf_backward = o.Nf_gcf_backward * o.Nfacet**2 * 5. * o.binfrac *(o.Na-1)* o.Na * o.Nmm * o.Ncvff**2 * log(o.Ncvff,2)/(2. * o.Tkernel_backward) 	# Eq. 35
        o.Rccf_predict = o.Nf_gcf_predict * o.Nfacet**2 * 5. * o.binfrac *(o.Na-1)* o.Na * o.Nmm * o.Ncvff**2 * log(o.Ncvff,2)/(2. * o.Tkernel_predict) 	# Eq. 35
        o.Rccf = o.Rccf_backward + o.Rccf_predict
        o.Rflop_conv = Rflop_common_factor * o.Rccf
        
		# Add in some phase rotation for the faceting:
        o.Rflop_phrot =o .Nmajor * o.Nbeam * o.Npp * (o.Nvis_predict + o.Nvis_backward) * o.Nfacet**2 * 25. * sign(o.Nfacet-1) 	# Eq. 29

		# Calculate revised Eq. 30:
        o.Rflop = o.Rflop_phrot + o.Rflop_proj + o.Rflop_fft + o.Rflop_conv + o.Rflop_grid  # Overall flop rate
		# ===============================================================================================
        o.Mbuf_vis = 2 * o.Mvis * o.Nbeam * o.Npp * o.Nvis_predict * o.Tobs / u.s # Note the division by u.s to get rid of pesky SI unit.
        #Also note the factor 2 -- we have a double buffer (allowing storage of a full observation while simultaneously capturing the next)

        o.Mw_cache = o.Ngw**3 * o.Qgcf**3 * o.Nbeam * o.Nf_vis_predict * 8

        o.Rio = o.Mvis * (o.Nmajor+1) * o.Nbeam * o.Npp * o.Nvis_predict * o.Nfacet**2 #added o.Nfacet dependence; changed Nmajor factor to Nmajor+1 as part of post PDR fixes.


        o.Npix_linear = o.Npix_linear * o.binfrac
            #        if o.Bmax_bin/o.Bmax != 1:
#o.Npix_linear = 0. #only output non-zero value of Npix_linear once, not for each baseline bin
#       else:
#           print "Npix linear, per facet:", o.Npix_linear, "for max baseline", o.Bmax

