from telescope_parameters import *
#from definitions_derivations import *
#from sympy import symbols, pi, log, ln, Max, sqrt, sign

telescope_list = ['SKA1_Low', 'SKA1_Mid', 'SKA1_Survey'] 
imaging_modes = ['Continuum', 'Spectral', 'SlowTrans', 'CS']   # CS = Continuum, followed by spectral

telescope_labels = dict(
    zip(
        telescope_list,
        symbols('Low, Mid, Sur')
    )
)

mode_labels = dict(
    zip(
        imaging_modes,
        symbols('C, S, ST, C+S')
    )
)

def optimize_expr(expression, free_var, bound_lower, bound_upper):
    '''
    Optimized Tsnap so that the supplied expression is minimized
    '''
    expr_eval = lambdify(free_var, expression, modules=("sympy",))
    expr_eval_float = lambda x: float(expr_eval(x))
        
    # Lower bound cannot be higher than the uppper bound.
    if bound_upper <= bound_lower:
        # print 'Unable to optimize free variable as upper bound is lower that the lower bound
        return bound_lower
    else:
        result = opt.minimize_scalar(expr_eval_float, bounds=(bound_lower, bound_upper), method='bounded')
        if not result.success:
            print 'WARNING! : Was unable to optimize free variable. Using a value of: %f' % result.x 
        else:
            # print 'Optimized free variable = %f' % result.x
            pass
        return result.x
    
def calc_tel_params(band=None, mode=None, hpso=None):
    '''
    Calculate telescope parameters for a supplied band, mode or hpso
    '''
    freq_range = {}
    hpso_params = None
    if hpso is not None:
        assert hpso in hpsos
        hpso_params = hpsos[hpso]
        assert hpso_params is not None
        assert 'telescope' in hpso_params
        telescope = hpso_params['telescope']
        if 'comment' in hpso_params:
            hpso_params.pop('comment')
    else:
        freq_range = copy.copy(band_info[band])
        telescope = freq_range['telescope']
        freq_range.pop('telescope')  # This key currently is not a defined variable, so we need to lose it for evaluations.
        
    #######################################################################################################################
    # Concatenate all relevant dicionaries to make a unified dictionary of all parameters to use (for this evaluation run)
    # The sequence of concatenation is important - each added parameter set overwrites 
    # any existing parameters that have duplicate keys.
    #
    # Concatenations are done with the following syntax: extended_params = dict(base_params, **additional_params)
    #
    # First, universal (default) parameters. These will most probably not be overwritten as they relate to physical constants.
    # Then, the telescope-specific parameters.
    # Then, the frequency range that was explicitly specified (if any).
    # Then, the parameters belonging to the selected imaging mode
    # Lastly, if specified, the parameters of the specific high priority science objective (HPSO)
    #######################################################################################################################

    telescope_params = dict(universal, **telescope_info[telescope])  # universal and telescope parameters
    telescope_params = dict(telescope_params, **freq_range)  # Overwrite specific frequency range information
        
    answer = {}
    if mode == 'CS':
        telescope_params = dict(telescope_params, **imaging_mode_info['Continuum'])
        telescope_params_C = telescope_params.copy()
        if hpso_params is not None:
            telescope_params_C = dict(telescope_params_C, **hpso_params)
                    
        telescope_params = dict(telescope_params, **imaging_mode_info['Spectral'])
        telescope_params_S = telescope_params.copy()
        if hpso_params is not None:
            telescope_params_S = dict(telescope_params_S, **hpso_params)

        telescope_params = {'C'  : telescope_params_C, 
                            'S'  : telescope_params_S}
    else:
        telescope_params = dict(telescope_params, **imaging_mode_info[mode])
        if hpso_params is not None:
            telescope_params = dict(telescope_params, **hpso_params)
                
    return telescope_params

def calc_tel_expression(expression, telescope_parameters, mode=None):
    '''
    Calculates the answer to an expression by substituting the supplied telescope parameters into the expression.
    '''
    tp = telescope_parameters.copy()  # Make a copy, as we will be locally modifying the dictionary
    # Remove specific arrays from the telescope parameters, as these can't be parsed by subst
    if baseline_bins in tp:
        tp.pop(baseline_bins)
    if baseline_bin_counts in tp:
        tp.pop(baseline_bin_counts)

    Tsnap_optimal = optimize_Tsnap(tp)  # Solution to Tsnap has to be bounded
    tp[Tsnap] = Tsnap_optimal
    answer = expression.subs(tp).subs(tp)
    if (mode is not None) and (mode != 'CS'):
        answer = {mode : answer}
    if Tsnap in tp:
        tp.pop(Tsnap) # for safety: forgets the value of Tsnap again for subsequent evaluations
    return answer

def calc_tel_expression_binned(expression, telescope_parameters, mode=None, verbose_variables=()):
    '''
    Calculates the answer to an expression by substituting the supplied telescope parameters into the expression.
    '''
    tp = telescope_parameters.copy()  # Make a copy, as we will be locally modifying the dictionary
    if Tsnap in tp:
        tp.pop(Tsnap) # We will need to optimize Tsnap, so it has to be undefined

    bins = tp.pop(baseline_bins) # Remove the array of baselines from the parameter dictionary
    counts = tp.pop(baseline_bin_counts) # Remove the array of baselines from the parameter dictionary

    bins_unitless = bins / u.m
    nbins = bins_unitless.searchsorted(remove_units(tp[Bmax])) + 1  # Gives the index of the first bin whose baseline exceeds the max baseline used    
    bins[:nbins]  # Restrict the bins used to only those bins that are used
    counts = counts[:nbins]  # Restrict the bins counts used to only those bins that are used

    nbaselines = sum(counts)
    temp_result = 0
    for i in range(nbins):
        binfrac_value = float(counts[i]) / nbaselines
        tp[Bmax_bin] = bins[i] #use Bmax for the bin only,  not to determine map size
        tp[binfrac] = binfrac_value  # Make sure that this is a floating point division, otherwise we get zero!

        if len(verbose_variables) > 0:
            print 'Bin %i with Bmax %.2f km contains %.3f %% of the baselines for this telescope' % (i, remove_units(tp[Bmax_bin]/1e3), binfrac_value*100)
            print 'Verbose variable printout:'
            verbose_output = []
            for variable in verbose_variables:
                verbose_output.append(float(remove_units(variable.subs(tp).subs(tp))))
            print verbose_output
            # Compute the actual result

        temp_result += expression.subs(tp).subs(tp) #do need to separate out Mwcache? (i.e. is it the maximum Wkernel memort size we're interested in or the total memory required?).
        #fmalan - TODO yes we need to; I was just thinking along the links of RFLOP when implementing this.
    if mode == 'CS':
        raise Exception('Cannot yet handle CS mode when using binned baselines. Should be simple to implement though.')

    # Remove string literals from the telescope_params, as they can't be evaluated by lambdify    
    bound_lower = remove_units(Tsnap_min.subs(tp).subs(tp))
    bound_upper = 0.5 * remove_units(Tobs.subs(tp).subs(tp))
    Tsnap_optimal = optimize_expr(temp_result, Tsnap, bound_lower, bound_upper)
    value_optimal = temp_result.subs({Tsnap : Tsnap_optimal})
    print "Tsnap has been optimized as : %f (for the binned case), yielding a minimum value of %f Peta-units" % (Tsnap_optimal, value_optimal / 1e15)     # Temp TODO remove later
    return {Tsnap : Tsnap_optimal, 'value' : value_optimal}  # Replace Tsnap with its optimal value

def calc_tel_old(band=None, mode=None, hpso=None, expression=None):
    '''
    Evaluate an expression for a single telescope.
    This method is no longer strictly necessary in this form, but is provided for backward compatibility
    '''
    answer = None
    
    if hpso is not None: # Get the 'mode' that corresponds to the hpso
        assert hpso in hpsos
        hpso_params = hpsos[hpso]
        assert hpso_params is not None
        assert 'mode' in hpso_params
        if mode is not None:
            assert mode == hpso_params['mode']
        mode = hpso_params['mode']
    
    params = calc_tel_params(band, mode, hpso)
    params[binfrac] = 1.0  # The binning fraction (that was introduced for the newer binning code, here statically set to 1)

    if band is not None:
        telescope = band_info[band]['telescope']
        if telescope == 'SKA1_Low':
            params[Qw2]  = 0.0458053
            params[Qw32] = 0.0750938
        elif telescope == 'SKA1_Low':
            params[Qw2]  = 0.0278115
            params[Qw32] = 0.0462109
        elif telescope == 'SKA1_Low':
            params[Qw2]  = 0.0569392
            params[Qw32] = 0.0929806
            
    if mode == 'CS':
        answer_C = calc_tel_expression(expression, params['C'], mode)
        answer_S = calc_tel_expression(expression, params['S'], mode)
        answer = {'Continuum' : answer_C, 
                  'Spectral'  : answer_S}
    else:
        answer = calc_tel_expression(expression, params, mode)
    return answer