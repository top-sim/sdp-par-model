import sympy.physics.units as u
from parameter_definitions import ParameterDefinitions as p
from formulae import Formulae as f
from sympy import simplify, lambdify, Max
from scipy import optimize as opt
import numpy as np

class ParameterContainer:
    def __init__(self):
        pass

class Implementation:
    def __init__(self):
        pass

    @staticmethod
    def remove_units(expression):
        return expression.replace(lambda el: hasattr(u, str(el)), lambda el: 1)

    @staticmethod
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

    @staticmethod
    def calc_tel_params(band=None, mode=None, hpso=None):
        """
        This is a very important method - Calculates telescope parameters for a supplied band, mode or HPSO
        """
        telescope_params = ParameterContainer()
        p.apply_global_parameters(telescope_params)
        p.define_symbolic_variables(telescope_params)

        assert mode is not None
        if band is not None:
            assert hpso is None
            telescope = p.get_telescope_from_band(band)
        elif hpso is not None:
            telescope = p.get_telescope_from_hpso(hpso)
        else:
            raise Exception("Either band or hpso must not be defined")

        # Note the order in which these settings are applied, with each one (possibly) overwriting previous definitions,
        # should they overlap (as happens with e.g. frequency bands)

        # First: The telescope's parameters (Primarily the number of dishes, bands, beams and baselines)
        p.apply_telescope_parameters(telescope_params, telescope)
        # Second: The imaging mode (Observation time, number of cycles, quality factor, (possibly) number of channels)
        p.apply_imaging_mode_parameters(telescope_params, mode)
        # Third: Frequency-band (frequency range - and in the case of HPSOs - other application-dependent settings)
        if band is not None:
            p.apply_band_parameters(telescope_params, band)
        elif hpso is not None:
            p.apply_hpso_parameters(telescope_params, hpso)
        else:
            raise Exception("Either band or hpso must not be None")

        f.compute_derived_parameters(telescope_params, mode)
        return telescope_params

    @staticmethod
    def update_derived_parameters(telescope_params, mode):
        """
        Used for updating the derived parameters if, e.g., some of the initial parameters was manually changed
        """
        p.apply_imaging_mode_parameters(telescope_params, mode)
        f.compute_derived_parameters(telescope_params, mode)

    @staticmethod
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

        for nfacets in range(1, max_number_nfacets+1):  # Loop over the different integer values of NFacet
            expression_original = definitions.Rflop
            # Warn if large values of nfacets are reached, as it may indicate an error and take long!
            if (nfacets > 20) and not warned:
                print 'Searching for minimum Rflop with nfacets > 20... this is a bit odd (and may take long)'
                warned = True

            i = nfacets-1 # zero-based index
            if verbose:
                print 'Evaluating Nfacets = %d' % nfacets

            Rflops = expression_original.subs({definitions.Nfacet : nfacets})
            answer = Implementation.minimize_binned_expression_by_Tsnap(expression=Rflops,
                                                                        telescope_parameters=definitions,
                                                                        verbose=verbose)

            flop_array.append(float(answer['value']))
            Tsnap_array.append(answer[definitions.Tsnap])
            flop_results[nfacets] = flop_array[i]/1e15
            if nfacets >= 2:
                if flop_array[i] >= flop_array[i-1]:
                    if verbose:
                        print '\nExpression increasing with number of facets; aborting exploration of Nfacets > %d' \
                              % nfacets
                    break

        i = np.argmin(np.array(flop_array))
        nfacets = i + 1
        if verbose:
            print '\n%f PetaFLOPS was the lowest FLOP value, found for (Nfacet, Tsnap) = (%d, %.2f)' \
                  % (flop_array[i]/1e15, nfacets,  Tsnap_array[i])

        return (Tsnap_array[i], nfacets)

    @staticmethod
    def substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose=False, take_max=False):
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
                print 'Bin with Bmax %.2f km contains %.3f %% of the baselines for this telescope' % \
                      (bins[i]/(u.m*1e3), binfrac_value*100)
                print 'Verbose variable printout: <empty>'

            if take_max:  # For example when computing Npix, we take the max
                temp_result = Max(temp_result, expr_subst)
            else:         # For most other varibles we sum over all bins
                temp_result += expr_subst

        return temp_result

    @staticmethod
    def evaluate_binned_expression(expression, telescope_parameters, verbose=False, take_max=False):
        tp = telescope_parameters
        bins = tp.baseline_bins # Remove the array of baselines from the parameter dictionary
        counts = tp.baseline_bin_counts # Remove the array of baselines from the parameter dictionary

        bins_unitless = bins / u.m
        assert tp.Bmax is not None
        Bmax_num_value = tp.Bmax / u.m
        # Compute the index of the first bin whose baseline exceeds the max baseline used (must be <= number of bins)
        nbins_used = min(bins_unitless.searchsorted(Bmax_num_value) + 1, len(bins))
        bins = bins[:nbins_used]  # Restrict the bins used to only those bins that are used
        bins[nbins_used-1] = tp.Bmax
        counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

        result = Implementation.substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose,
                                                             take_max=take_max)
        return float(result)

    @staticmethod
    def minimize_binned_expression_by_Tsnap(expression, telescope_parameters, verbose=False, take_max=False):
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
        Bmax_num_value = tp.Bmax / u.m
        nbins_used = bins_unitless.searchsorted(Bmax_num_value) + 1  # Gives the index of the first bin whose baseline exceeds the max baseline used
        bins = bins[:nbins_used]  # Restrict the bins used to only those bins that are used
        bins[nbins_used-1] = tp.Bmax
        counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

        result = Implementation.substitute_parameters_binned(expression, tp, bins, counts, nbins_used, verbose=False,
                                                             take_max=take_max)

        # Remove string literals from the telescope_params, as they can't be evaluated by lambdify
        bound_lower = tp.Tsnap_min
        bound_upper = 0.5 * tp.Tobs / u.s

        Tsnap_optimal = Implementation.optimize_expr(result, tp.Tsnap, bound_lower, bound_upper)
        value_optimal = result.subs({tp.Tsnap : Tsnap_optimal})
        if verbose:
            print "Tsnap has been optimized as : %f. (Cost function = %f)" % \
                  (Tsnap_optimal, value_optimal / u.peta)
        return {tp.Tsnap : Tsnap_optimal, 'value' : value_optimal}  # Replace Tsnap with its optimal value