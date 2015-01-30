import sympy.physics.units as u
from parameter_definitions import parameter_definitions as p
from formulae import formulae as f
from variable_definitions import symbolic_definitions as s
from sympy import simplify, lambdify
from scipy import optimize as opt
import numpy as np

class parameter_container:
    def __init__(self):
        pass

def remove_units(expression):
    return expression.replace(lambda el: hasattr(u, str(el)), lambda el: 1)


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

def find_optimal_Tsnap_Nfacet(definitions, max_number_nfacets=200, verbose=False):
    '''
    Computes the optimal value for Tsnap and Nfacet that minimizes the value of Rflop
    according to its definition in the supplied definitions object
    Returns result as a 2-tuple (Tsnap_opt, Nfacet_opt)
    '''

    flop_results = {} # Maps nfacet values to flops
    flop_array = []
    Tsnap_array = []
    warned = False

    for nfacets in range(1, max_number_nfacets+1):  # Loop over the different integer values of NFacet (typically 1..10)
        expression_original = definitions.Rflop
        # Warn if large values of nfacets are reached, as it may indicate an error and take long!
        if (nfacets > 20) and not warned:
            print 'Searching for minimum Rflop with nfacets > 20... this is a bit odd (and may take long)'
            warned = True

        i = nfacets-1 # zero-based index
        if verbose:
            print 'Evaluating Nfacets = %d' % nfacets

        Rflops = expression_original.subs({definitions.Nfacet : nfacets})
        answer = minimize_binned_expression_by_Tsnap(expression=Rflops, telescope_parameters=definitions, verbose=verbose)

        flop_array.append(float(answer['value']))
        Tsnap_array.append(answer[definitions.Tsnap])
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

def substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose=False):
    '''
    Substitute relevant variables for each bin, summing the result
    '''
    nbaselines = sum(counts)
    temp_result = 0
    for i in range(nbins_used):
        binfrac_value = float(counts[i]) / nbaselines  # NB: Ensure that this is a floating point division
        # Substitute bin-dependent variables
        expr_subst = expression.subs({tp.Bmax_bin: bins[i], tp.binfrac : binfrac_value})

        if verbose:
            print 'Bin with Bmax %.2f km contains %.3f %% of the baselines for this telescope' % (bins[i]/(u.m*1e3), binfrac_value*100)
            print 'Verbose variable printout: <empty>'

        temp_result += expr_subst
    return temp_result

def evaluate_binned_expression(expression, telescope_parameters, verbose=False):
    tp = telescope_parameters
    bins = tp.baseline_bins # Remove the array of baselines from the parameter dictionary
    counts = tp.baseline_bin_counts # Remove the array of baselines from the parameter dictionary

    bins_unitless = bins / u.m
    assert tp.Bmax is not None
    Bmax = tp.Bmax / u.m
    nbins_used = bins_unitless.searchsorted(Bmax) + 1  # Gives the index of the first bin whose baseline exceeds the max baseline used
    bins = bins[:nbins_used]  # Restrict the bins used to only those bins that are used
    counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

    result = substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose)
    return float(result)

def minimize_binned_expression_by_Tsnap(expression, telescope_parameters, verbose=False):
    '''
    Minimizes an expression by substituting the supplied telescope parameters into the expression, then minimizing it
    by varying the free parameter, Tsnap
    '''
    #TODO: can make the free expression a parameter of this method (should something else than Tsnap be desired)

    tp = telescope_parameters
    bins = tp.baseline_bins # Remove the array of baselines from the parameter dictionary
    counts = tp.baseline_bin_counts # Remove the array of baselines from the parameter dictionary

    bins_unitless = bins / u.m
    assert tp.Bmax is not None
    Bmax = tp.Bmax / u.m
    nbins_used = bins_unitless.searchsorted(Bmax) + 1  # Gives the index of the first bin whose baseline exceeds the max baseline used
    bins = bins[:nbins_used]  # Restrict the bins used to only those bins that are used
    counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

    result = substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose=False)

    # Remove string literals from the telescope_params, as they can't be evaluated by lambdify    
    bound_lower = tp.Tsnap_min
    bound_upper = 0.5 * tp.Tobs / u.s

    Tsnap_optimal = optimize_expr(result, tp.Tsnap, bound_lower, bound_upper)
    value_optimal = result.subs({tp.Tsnap : Tsnap_optimal})
    if verbose:
        print "Tsnap has been optimized as : %f, yielding a minimum value of %f Peta-units" % (Tsnap_optimal, value_optimal / 1e15)
    return {tp.Tsnap : Tsnap_optimal, 'value' : value_optimal}  # Replace Tsnap with its optimal value

