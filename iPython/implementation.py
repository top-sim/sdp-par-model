from parameters import parameters as p
from formulae import formulae as f
from variable_definitions import symbolic_definitions as s
import sys

class parameter_container:
    def __init__(self):
        pass

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
    
def calc_tel_params(band=None, mode=None, hpso_key=None):
    '''
    Calculate telescope parameters for a supplied band, mode or hpso
    '''
    telescope_params = parameter_container()
    p.apply_global_parameters(telescope_params)
    s.define_symbolic_variables(telescope_params)

    assert mode is not None
    if band is not None:
        telescope_string = p.get_telescope_from_band(band)
    elif hpso_key is not None:
        telescope_string = p.get_telescope_from_hpso(hpso_key)
    else:
        raise Exception("Either band or hpso must not be None")

    # Note the order in which these settings are applied, with each one (possibly) overwriting previous definitions,
    # should they overlap (as happens with e.g. frequency bands)
    p.apply_telescope_parameters(telescope_params, telescope_string)
    p.apply_imaging_mode_parameters(telescope_params, mode)
    if band is not None:
        p.apply_band_parameters(telescope_params, band)
    elif hpso_key is not None:
        p.apply_hpso_parameters(telescope_params, hpso_key)
    else:
        raise Exception("Either band or hpso must not be None")

    f.compute_derived_parameters(telescope_params, mode)
    return telescope_params

    '''
    freq_range = {}
    hpso_params = None
    if hpso_key is not None:
        assert hpso_key in hpsos
        hpso_params = hpsos[hpso_key]
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

    telescope_params = dict(telescope_info[telescope], **freq_range)  # universal and telescope parameters
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
    '''


def substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose_variables=()):
    nbaselines = sum(counts)
    temp_result = 0
    for i in range(nbins_used):
        binfrac_value = float(counts[i]) / nbaselines  # NB: Ensure that this is a floating point division
        tp[Bmax_bin] = bins[i]  #use Bmax for the bin only,  not to determine map size
        tp[binfrac] = binfrac_value
        if len(verbose_variables) > 0:
            print 'Bin with Bmax %.2f km contains %.3f %% of the baselines for this telescope' % (remove_units(tp[Bmax_bin]/1e3), tp[binfrac]*100)
            print 'Verbose variable printout:'
            verbose_output = []
            for variable in verbose_variables:
                verbose_output.append(float(remove_units(variable.subs(tp).subs(tp))))
            print verbose_output
        # Compute the actual result
        temp_result += expression.subs(tp).subs(tp)
    return temp_result

def evaluate_expression(expression, telescope_parameters, verbose_variables=()):
    tp = telescope_parameters.copy()  # Make a copy, as we will be locally modifying the dictionary
    bins = tp.pop(baseline_bins) # Remove the array of baselines from the parameter dictionary
    counts = tp.pop(baseline_bin_counts) # Remove the array of baselines from the parameter dictionary
    bins_unitless = bins / u.m
    nbins_used = bins_unitless.searchsorted(remove_units(tp[Bmax])) + 1  # Gives the index of the first bin whose baseline exceeds the max baseline used
    bins[:nbins_used]  # Restrict the bins used to only those bins that are used
    counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

    result = substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose_variables)
    return float(result)

def minimize_expression(expression, telescope_parameters, mode=None, verbose=False, verbose_variables=()):
    '''
    Minimizes an expression by substituting the supplied telescope parameters into the expression, then minimizing it
    by varying the free parameter, Tsnap
    '''
    #TODO: can make the free expression a parameter of this method (should something else than Tsnap be desired)
    if mode == 'CS':
        raise Exception('Cannot yet handle CS mode when using binned baselines. Should be simple to implement though.')

    tp = telescope_parameters.copy()  # Make a copy, as we will be locally modifying the dictionary
    if Tsnap in tp:
        tp.pop(Tsnap) # We will need to optimize Tsnap, so it has to be undefined

    bins = tp.pop(baseline_bins) # Remove the array of baselines from the parameter dictionary
    counts = tp.pop(baseline_bin_counts) # Remove the array of baselines from the parameter dictionary

    bins_unitless = bins / u.m
    nbins_used = bins_unitless.searchsorted(remove_units(tp[Bmax])) + 1  # Gives the index of the first bin whose baseline exceeds the max baseline used
    bins[:nbins_used]  # Restrict the bins used to only those bins that are used
    counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

    result = substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose_variables)

    # Remove string literals from the telescope_params, as they can't be evaluated by lambdify    
    bound_lower = Tsnap_min
    bound_upper = 0.5 * remove_units(Tobs.subs(tp).subs(tp))
    Tsnap_optimal = optimize_expr(result, Tsnap, bound_lower, bound_upper)
    value_optimal = result.subs({Tsnap : Tsnap_optimal})
    if verbose:
        print "Tsnap has been optimized as : %f, yielding a minimum value of %f Peta-units" % (Tsnap_optimal, value_optimal / 1e15)
    return {Tsnap : Tsnap_optimal, 'value' : value_optimal}  # Replace Tsnap with its optimal value

def find_optimal_Tsnap_Nfacet(band=None, mode=None, hpso=None, max_number_nfacets=200, verbose=False):
    '''
    Computes the optimal value for Tsnap and Nfacet that minimizes the value of Rflop for a given telescope, band and mode
    Returns result as a 2-tuple (Tsnap_opt, Nfacet_opt)
    '''
    if verbose:
        if hpso is None:
            print 'Finding optimal values for Tsnap and Nfacet for (band = %s, mode = %s)' % (band, mode)
        else:
            print 'Finding optimal values for Tsnap and Nfacet for HPSO %s (mode = %s)' % (hpso, mode)
    
    flop_results = {} # Maps nfacet values to flops
    flop_array = []
    Tsnap_array = []
    warned = False
    for nfacets in range(1, max_number_nfacets+1):  # Loop over the different integer values of NFacet (typically 1..10)
        # Warn if large values of nfacets are reached, as it may indicate an error and take long!
        if (nfacets > 20) and not warned:
            print 'Searching for minimum Rflop with nfacets > 20... this is a bit odd (and may take long)'
            warned = True

        i = nfacets-1 # zero-based index
        if verbose:
            print 'Evaluating Nfacets = %d' % nfacets
        parameters = calc_tel_params(band=band, mode=mode, hpso_key=hpso)
        parameters[Nfacet] = nfacets
        answer = minimize_expression(expression=Rflop, telescope_parameters=parameters, mode=mode)
        flop_array.append(float(answer['value']))
        Tsnap_array.append(answer[Tsnap])
        flop_results[nfacets] = flop_array[i]/1e15
        if nfacets >= 2:
            if flop_array[i] >= flop_array[i-1]:
                if verbose:
                    print '\nExpression increasing with number of facets; aborting exploration of Nfacets > %d' % nfacets
                break

    i = np.argmin(np.array(flop_array))
    nfacets = i + 1
    if verbose:
        print '\n%f PetaFLOPS was the lowest FLOP value, found for (Nfacet, Tsnap) = (%d, %.2f)' \
              % (flop_array[i]/1e15, nfacets,  Tsnap_array[i])
    return (Tsnap_array[i], nfacets)