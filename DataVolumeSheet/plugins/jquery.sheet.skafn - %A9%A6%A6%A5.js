(function(jFN){
    var jSF = jQuery.sheet.skafn = {
		CALCFIDFREQ:function (fmin,fmax,ffin) {
			/* CALCFIDFREQ calculates the Fiducial Frequency at which the calculations of data are correct. 
				Data volume is assumed to scale as \nu^-3     
				Input: Fmin & Fmax (& The used Fiducial Freq) (in any units)
				Output: FF (in same units) if only 2 inputs. The ratio between actual and used, if FFin is provided.          
			*/
			var ff=0;  
			if (fmin==0 || fmax==0) {ff="Input Value(s) Zero";return ff;}
			if (fmin>fmax) {ff=fmax;fmax=fmin;fmin=ff;ff=0;}  
			ff=Math.pow((1/fmin/fmin-1/fmax/fmax)/(fmax-fmin)/2,-1/3);
			if (ffin) ff=ff/ffin; 
			return ff;
		},
		CALCFOV2:function (freq,diam, nb) {
			/* CALCFIDFREQ calculates the Field of View in Deg^2.               
			Input: Freq (Fiducial Freq) (in MHz)
            Diam (Dish or station size) (in meters)
			Output: FoV in deg^2.
			*/
			var fov=0;
			var pi=3.141;
			if (!nb) {nb=1;}
			fov=nb*pi/4*((66*300/freq/diam)*(66*300/freq/diam));
			return fov;
		},
		CALCNCHAN:function (freq,bw,arr,dvel,diam,bmax,oversamp) {  
			/* Calculate the number of channels needed
				Input: Freq, BW, class, delta vel, diam, maximum baseline, oversampl
						[MHz]             [km/s]     [m]   [km]
				Output: nChan
				The number of channels required to produce either no bandwidth smearing or the vel resol (class=line) required
				for the whole of the band - which is to say at fmin
				For class=array the array crossing time matches the channel width
			*/
			var ddv=1e32; // Velocity res
			var dbw=1e32; // Bandwidth Smearing
			var nc=0;
  
			if (oversamp==0) {oversamp=4;} 
			// d nu in MHz
			{ddv=freq*dvel/3e5;} //in which case should this use fmin?
			{dbw=freq*diam/(bmax*1000)/oversamp;} // Should this use the fiucial? No as it smearing at fmin
			// Changed to the input freq. User decides!
			if (arr!="line" || dbw<ddv) {nc=bw/dbw;} // Should have a DM option .. 
			else {nc=bw/ddv;}
			if (arr=="array") {nc=bw/0.3*bmax;} // Array crossing time (d/c) should = d\nu  
			if (nc>262144) {nc=262144;} // Limit at 2^18 channels (256K channels)	
			return nc;
		},
		CALCNCHAN2FREQ:function (fmin,fmax,bw,arr,dvel,diam,bmax,oversamp) {  
			/* Calculate the number of channels needed
				Input: Freq min, Freq max, BW, class, delta vel, diam, maximum baseline, oversampl
						[MHz]                           [km/s]     [m]   [km]
				Output: nChan
				The number of channels required to produce either no bandwidth smearing or the vel resol (class=line) required
				for the whole of the band - which is to say at fmin
			*/
			var ddv=1e32; // Velocity res
			var dbw=1e32; // Bandwidth Smearing
			var nc=0;
			var ff=0; //Fiducial Freq
			ff=CALCFIDFREQ(fmin,fmax);
			if (oversamp==0) {oversamp=4;}  
			// d nu in MHz
			{ddv=fmin*dvel/3e5;} //in which case should this use fmin?
			{dbw=fmax*diam/(bmax*1000)/oversamp;} // Should this use the fiucial? No as it smearing at fmin
			// Changed to both freq Max and Min. User decides if to use this function!
			if (arr!="line" || dbw<ddv) {nc=bw/dbw;}
			else {nc=bw/ddv;}
			return nc;
		},
		RAWIN:function (bw,b_raw,nant) {
			/* Raw input
			Input BANDWIDTH [MHz],b_raw [bytes], n_antennas
			Output Data Rate B/s 
			Assumes both pols and Single Beam!
			*/
			var dr=0; //data rate
			dr=bw*2*nant*2*b_raw*1000000;  
			return dr;
		},
		CORRELATORINT:function (d,bmax,oversamp) {  
			/*
			Input Diam [m], B_max [km],  Oversamping 
			Output t_int [s]
			*/
			var t_int=0;
			t_int=d*13.713/bmax/oversamp;  
			return t_int;
		},
		CORRELATORINT2:function (d,bmax,oversamp) {  
			/*
			Input Diam [m], B_max [km],  Oversamping 
			Output t_int [s]
			*/
			var t_int=0;
			t_int=d*13.713/bmax/oversamp;
			return t_int;
		},
		CORRELATOROUT:function (nant,nbeam,nchan,t_int,b_vis,npol) {
			/* Input: nchan n_ant t_int b_vis
			Output: Data Rate B/s */
			var dr=0; // Data Rate
			var np=0; // Internal NPol
			np=1; if (npol>1) np=4;
			dr=nbeam*nchan*nant*(nant+1)/2/t_int*np*b_vis;
			return dr;
		},
		BANDWIDTH:function (fmin,fmax,bw,dv) {
			// Fmin, Fmax, maximum BW in MHz, delta velocity in kms
			var rv=0;
			var nc=0;
			rv=fmax-fmin;
			if (rv>bw) rv=bw;
			if (dv) {
			nc=(3e5/dv*rv)/fmin;
			// Limit at 2^18 channels (256K channels)
			if (nc>262144) {rv=Math.ceil(262144*fmin*dv/3e5);}
			}
			return rv;			
		},		
		RESOLUTION:function (freq,bmax) {
			/* Input Freq [MHz], Bmax [km]
			Output Res [arcsec] */  
			var res=0;
			res=61884/freq/bmax;  
			return res;
		},
		CONFUSION:function (freq,res) {
			/* Input: Freq [MHz], Resol [arcsec]
			Ouput: CONFUSION (uJy/beam) */
			var conf=0;
			conf=1.2*Math.pow(freq/3020,-0.7)*Math.pow(res/8,10/3);
			return conf;
		},
		DAILYIMAGE:function (nchan,nbyte,bmsize,fov,npol,oversamp,pstamp,arr) { 
			/* Input: Nchans, Nbytes, Beam Size, Fov,  Npol, N beam, Over Sampling, No. Postage Stamps, class
										[arcs]  [deg^2]                                                 [str]  
			Ouput: Image Size in bytes     
			If no postage stamps are given the Image size is nchan*nbyte*Fov*((beam*oversamp)^-2)     
			If Postage Stamps is greater than zero it is assumed that this is the total for all beams (i.e. pstamp_per_beam*nbeam) 
			In this case the Image is 10*10*1000 pixels (10 for X and Y, 1000 for velocity).      
			If class is given and does not equal "line" image is averaged to 1000 channels (or nchan if less than 1000)
			That is the default is class=="line" (i.e. if not given this is assumed)     
			To account for weights we are using (in the call) npol+1 (i.e. 5 in most cases). That is a 3D weight map     
			*/
			var im_size=0;
			var my_nc=1000; // Channels for final non-line and postage stamp images
			var my_xy=100; // Dimension of postage stamp
			if (my_nc>nchan) {my_nc=nchan;} // In case the number of channels is less than 1000 
			if (pstamp>0) {
				im_size=pstamp*my_nc*nbyte*npol*Math.pow(oversamp*my_xy,2);
			} 
			else { if (!arr||arr=="line") {im_size=nchan*nbyte*npol*(fov/Math.pow(bmsize/3600/oversamp,2));}
				else {im_size=my_nc*nbyte*npol*(fov/Math.pow(bmsize/3600/oversamp,2));}
			}  
			return im_size;
		},
		CALCSEFD:function (diam, arr, freq, nant) {
			/* Input: Diam, Array Class, Freq, Nant
						[m]   [str]      [MHz]                     
			Ouput: SEFD
					[Jy]
			Array can "AA"/"Low", "Mid" or "Survey"
			Freq is the Fiducial Freq in MHz (only needed for AA)
			Nant is the number of antennas to be summed
			Diam is the diameter in meters
			*/
			var sefd=-1;
			var tsys=20;
			var eff=1;
			var area=0;
			area=3.141*diam*diam/4; // pi.r^2
			if (nant==0) {nant=1;}
			if (arr=="AA"||arr=="Low") {
				tsys = 60*Math.pow(300/freq,2.55);
				eff = 1.0;
			}
			if (arr=="Mid") {
				tsys = 20;
				eff = 0.78;
			}
			if (arr=="Survey") {
				tsys = 30;
				eff = 0.8;
			}   
			sefd = 2760*tsys/area/eff/nant;
			return sefd;
		},
		FINALSEN:function (sefd,tobs,bw,bwreq,npol,nchan,target_area,fov2,arr) {
			// Calculate sensitivty based on SEFD and observing time
			//  SEFD in Jy/bm, Tobs in kHr, BW & BW required in MHz, Npol, Nchan,
			//  Targeted area and FoV in deg^2
			var sen=0; // uJy/bm
			sen=sefd/Math.sqrt(npol*(bw*1e6)*(tobs*3600e3/(target_area/fov2)))*1e6;
			if (arr=="line") {
				sen = sen * Math.sqrt(nchan); // uJy/bm/channel
			}
			if ((bwreq/bw)>1) {
				sen=sen*(Math.ceil(bwreq/bw));
			}
			// return in uJy
			return sen;
		},
		BPASSLOSS:function (a) {
			return a;
		},
		AVFACTOR:function (bmax,diam,oversamp,arr2,freq,bw,dvel,arr) { 
			/*  To return an averaging factor for the effect that some baselines are shorter
				and so do not need a fast sampling time and narrow channel width */
			/* For now I will guess a value .. based on the integrated area of a guassian distrubtion of 
				baselines lengths */
			/*    The approximation I propose is pretty crude, but I think will turn out to be quite accurate. 
				1) Calculate the DP correlator output for B_max
				2) Assume that the baseline distrubution is approximately linear in log, as in the contained parameters
				3) Find (with AV_BMAX) where 68% of the baselines would be, and where 28% of baselines would be 
				and calculate the resolution. 
				For the continuum projects a similar correction is made for the channel width. 
				4) Scale 1 with this factor
				-- This routine calculates the factor as described above --
			*/  
			/*  Input Bmax (km) Diam (m) Oversamp Class=[line,!line]
			*/
			var af=1;
			var t1,t2,t3;
			var nch1,nch2,nch3;
			var loc_os=4;
			var loc_class="line";
			t1=t2=t3=0;
			nch1=nch2=nch3=0;
			/*   array       m      c
			------------------------------
				Low      -2.58    12.94
				Mid      -1.90     10.33
				Survey -1.14     5.28
			*/
			var log_lin;
			// Results from fitting: polyfit(log10(bb_x),log10(sba_x/sba_x(length(sba_x))),N-1)
			var n_low=8,log_low=[3.967956e-02,-9.545386e-01,9.599892e+00,-5.211806e+01,1.642835e+02,-2.997817e+02,2.940071e+02,-1.220649e+02];
			var n_mid=5,log_mid=[-0.0091473,0.1974332,-1.5980989,5.7945376,-7.9909332];
			var n_sur=7,log_sur=[-3.472990e-02,7.388065e-01,-5.996752e+00,2.286750e+01,-3.949642e+01,2.223534e+01];
			var cutoffs=[0.68,0.28,0.04];// Based on Guassian values - could be changed?
			cutoffs[2]=1-cutoffs[0]-cutoffs[1]; // Just to make sure :) 
			log_lin=log_low;n_lin=n_low;
			if (arr=="AA"||arr=="Low") {log_lin=log_low;n_lin=n_low;};
			if (arr=="Mid") {log_lin=log_mid;n_lin=n_mid;};
			if (arr=="Survey") {log_lin=log_sur;n_lin=n_sur;};
			if (oversamp>0) loc_os=oversamp;
			if (arr2) loc_class=arr2;
			if (diam>0&&bmax>0)
			{ 
				t1=CORRELATORINT(diam,AV_BMAX(bmax,log_lin,n_lin,cutoffs[0]),loc_os);
				t2=CORRELATORINT(diam,AV_BMAX(bmax,log_lin,n_lin,cutoffs[1]+cutoffs[0]),loc_os);
				t3=CORRELATORINT(diam,bmax,loc_os);
				af=af*(t1*cutoffs[0]+t2*cutoffs[1]+t3*cutoffs[2])/t3;
			}
			if (arr2!="line")
			{ 
				nch1=CALCNCHAN(freq,bw,loc_class,dvel,diam,AV_BMAX(bmax,log_lin,n_lin,cutoffs[0]),loc_os);
				nch2=CALCNCHAN(freq,bw,loc_class,dvel,diam,AV_BMAX(bmax,log_lin,n_lin,cutoffs[1]+cutoffs[0]),loc_os);
				nch3=CALCNCHAN(freq,bw,loc_class,dvel,diam,bmax,loc_os);   
				af=af/((nch1*cutoffs[0]+nch2*cutoffs[1]+nch3*cutoffs[2])/nch3);
			}
			return af;
		},
		AV_BMAX:function (bmax,log_lin,n_lin,cutoff) {
			// Return the baseline length which contains the fraction of baselines at cutoff, 
			// following the n=10^(d*log_lin[0]+log_lin[1]).
			// On failure returns 0
			var avb=1;
			var cd,c,b,co,cnt=0;
			var b0;
			co=cutoff;b=bmax*cutoff*1e3;
			// First Guess
			c=Math.pow(10,POLYVAL(log_lin,n_lin,Math.log(b)));
			cd=Math.pow(10,POLYVAL(log_lin,n_lin,Math.log(b*1.01)));
			// Newton Approx here ... TOBE FINISH <<Friday 6pm//Stop>>
			while (Math.abs(c-co)>0.05) { // 5%
				b0=b;
				if (avb<1) {avb=avb*2;}
				b=b0+(c-co)/(c-cd)*(b0*0.01)*avb; // moderate it?
				while (b<0||b>bmax*1e3) { avb=avb/2,b=b0+(c-co)/(c-cd)*(b0*0.01)*avb; }
				c=Math.pow(10,POLYVAL(log_lin,n_lin,Math.log(b)));
				cd=Math.pow(10,POLYVAL(log_lin,n_lin,Math.log(b*1.01)));
				cnt++;
				if (cnt>10) {b=0;c=co;}
			}
			return b/1e3;
		},
		POLYVAL:function (p,n,x) {
			//Mimic Matlab function
			var y;
			y=0;
			for (i=0;i<n;i++) y=y+p[i]*Math.pow(x,n-1-i);
			return y;
		},
		AVFACTORGAUSS:function (bmax,diam,oversamp,arr,freq,bw,dvel) { 
			/*  To return an averaging factor for the effect that some baselines are shorter
				and so do not need a fast sampling time and narrow channel width */
			/* For now I will guess a value .. based on the integrated area of a guassian distrubtion of 
				baselines lengths */
			/*    The approximation I propose is pretty crude, but I think will turn out to be quite accurate. 
				1) Calculate the DP correlator output for B_max
				2) Assume that the baseline distrubution is approximately Gaussian
				3) Assume B-max represents 3 sigma point in that Gaussian and calculate the T_int for B_max/3, 
				which is 68% of the data, B_max*.667, which is 28% of the data and the remaining 4% is kept at full resolution. 
				For the continuum projects a similar correction is made for the channel width. 
				4) Scale 1 with this factor   
				-- This routine calculates the factor as described above --
			*/  
			/*  Input Bmax (km) Diam (m) Oversamp Class=[line,!line]
			*/
			var af=1;
			var t1,t2,t3;
			var nch1,nch2,nch3;
			var loc_os=4;
			var loc_class="line";
			t1=t2=t3=0;
			nch1=nch2=nch3=0;
			if (oversamp>0) loc_os=oversamp;
			if (arr) loc_class=arr;
			if (diam>0&&bmax>0)
			{ 
				t1=CORRELATORINT(diam,bmax/3,loc_os);
				t2=CORRELATORINT(diam,bmax*2/3,loc_os);
				t3=CORRELATORINT(diam,bmax,loc_os);
				af=af*(t1*0.68+t2*0.28+t3*0.04)/t3;
			}
			if (arr!="line")
			{ 
				nch1=CALCNCHAN(freq,bw,loc_class,dvel,diam,bmax/3,loc_os);
				nch2=CALCNCHAN(freq,bw,loc_class,dvel,diam,bmax*2/3,loc_os);
				nch3=CALCNCHAN(freq,bw,loc_class,dvel,diam,bmax,loc_os);
				af=af/((nch1*0.68+nch2*0.28+nch3*0.04)/nch3);
			}	  
			return af;
		},
		AVFACTORFIXED:function (bmax,diam,oversamp,arr,freq,bw,dvel,arr) { 
			/*  To return an averaging factor for the effect that some baselines are shorter
				and so do not need a fast sampling time and narrow channel width */
			/* This returns the fixed value for a precalculated uv distribution */
			var af=1;
			if (diam>0&&bmax>0)
			{ 
				af=1;
				if (arr=="Low"||arr=="AA") {
				af=14.7;
				} else if (arr=="Mid") { 
				af=13.8;
				} else if (arr=="Survey") { 
				af=4.81; 
				}
			}
			if (arr!="line")
			{ 
				af=af*af;
			}  
			return af;
		}
	};
})(jQuery.sheet.fn);