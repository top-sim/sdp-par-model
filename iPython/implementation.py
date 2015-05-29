"""
This Python file contains two classes.

ParameterContainer is centrally important and used throughout the iPython model, but essentially is only a container
class that is passed around between modules, and contains a set of parameters, values and variables that constitute
the inputs and outputs of computations.

Implementation contains a collection of methods for performing computations, but do not define the equations
themselves. Instead, it specifies how values are substituted, optimized, and summed across bins.
"""

from parameter_definitions import Telescopes, ImagingModes, Bands
from parameter_definitions import ParameterDefinitions as p
from parameter_definitions import Constants as c
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
        """
        Optimized Tsnap so that the supplied expression is minimized
        """
        expr_eval = lambdify(free_var, expression, modules=("sympy",))
        expr_eval_float = lambda x: float(expr_eval(x))

        # Lower bound cannot be higher than the uppper bound.
        if bound_upper <= bound_lower:
            # print ('Unable to optimize free variable as upper bound is lower that the lower bound)
            return bound_lower
        else:
            result = opt.minimize_scalar(expr_eval_float, bounds=(bound_lower, bound_upper), method='bounded')
            if not result.success:
                print ('WARNING! : Was unable to optimize free variable. Using a value of: %f' % result.x)
            else:
                # print ('Optimized free variable = %f' % result.x)
                pass
            return result.x

    @staticmethod
    def calc_tel_params(telescope, mode, band=None, hpso=None, bldta=False, On_the_fly=0,
                        max_baseline=None, nr_frequency_channels=None, verbose=False):
        """
        This is a very important method - Calculates telescope parameters for a supplied band, mode or HPSO.
        Some default values may (optionally) be overwritten, e.g. the maximum baseline or nr of frequency channels.
        @param bldta: Baseline dependent time averaging
        """
        telescope_params = ParameterContainer()
        p.apply_global_parameters(telescope_params)
        p.define_symbolic_variables(telescope_params)

        assert (band is None) or (hpso is None)

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

        if max_baseline is not None:
            telescope_params.Bmax = max_baseline
        if nr_frequency_channels is not None:
            telescope_params.Nf_max = nr_frequency_channels

        f.compute_derived_parameters(telescope_params, mode, bldta, On_the_fly, verbose=verbose)
        #print "Using maximum baseline of", p.Bmax
        return telescope_params

    @staticmethod
    def telescope_and_band_are_compatible(telescope, band):
        """
        Checks whether the supplied telescope and band are compatible with each other
        @param telescope:
        @param band:
        @return:
        """
        is_compatible = False
        if telescope in {Telescopes.SKA1_Low_old, Telescopes.SKA1_Low}:
            is_compatible = (band in Bands.low_bands)
        elif telescope in {Telescopes.SKA1_Mid_old, Telescopes.SKA1_Mid}:
            is_compatible = (band in Bands.mid_bands)
        elif telescope == Telescopes.SKA1_Sur_old:
            is_compatible = (band in Bands.survey_bands)
        elif telescope == Telescopes.SKA2_Low:
            is_compatible = (band in Bands.low_ska2_bands)
        elif telescope == Telescopes.SKA2_Mid:
            is_compatible = (band in Bands.mid_ska2_bands)
        else:
            raise ValueError("Unknown telescope %s" % telescope)

        return is_compatible

    @staticmethod
    def find_optimal_Tsnap_Nfacet(definitions, max_number_nfacets=200, verbose=False):
        """
        Computes the optimal value for Tsnap and Nfacet that minimizes the value of Rflop
        according to its definition in the supplied definitions object
        Returns result as a 2-tuple (Tsnap_opt, Nfacet_opt)
        """

        flop_results = {} # Maps nfacet values to flops
        flop_array = []
        Tsnap_array = []
        warned = False

        for nfacets in range(1, max_number_nfacets+1):  # Loop over the different integer values of NFacet
            expression_original = definitions.Rflop
            # Warn if large values of nfacets are reached, as it may indicate an error and take long!
            if (nfacets > 20) and not warned:
                print ('Searching for minimum Rflop with nfacets > 20... this is a bit odd (and may take long)')
                warned = True

            i = nfacets-1 # zero-based index
            if verbose:
                print ('Evaluating Nfacets = %d' % nfacets)

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
                        print ('\nExpression increasing with number of facets; aborting exploration of Nfacets > %d' \
                              % nfacets)
                    break

        i = np.argmin(np.array(flop_array))
        nfacets = i + 1
        if verbose:
            print ('\n%f PetaFLOPS was the lowest FLOP value, found for (Nfacet, Tsnap) = (%d, %.2f)' \
                  % (flop_array[i]/1e15, nfacets,  Tsnap_array[i]))

        return (Tsnap_array[i], nfacets)

    @staticmethod
    def substitute_parameters_binned(expression, tp, bins, counts, verbose=False, take_max=False):
        """
        Substitute relevant variables for each bin, by defaukt summing the result. If take_max == True, then
        the maximum expression value over all bins is returns instead of the sum.
        @param expression:
        @param tp: ParameterContainer containing the telescope parameters
        @param bins: An array containing the max baseline length of each bin
        @param counts: The number of baselines in each of the bins
        @param verbose:
        @param take_max: iff True, returns the maximum instead of the sum of the bins' expression values
        """
        nbins_used = len(bins)
        assert nbins_used == len(counts)
        nbaselines = sum(counts)
        temp_result = 0
        for i in range(nbins_used):
            binfrac_value = float(counts[i]) / nbaselines  # NB: Ensure that this is a floating point division
            # Substitute bin-dependent variables
            expr_subst = expression.subs({tp.Bmax_bin: bins[i], tp.binfrac : binfrac_value})

            if take_max:  # For example when computing Npix, we take the max
                temp_result = Max(temp_result, expr_subst)
            else:         # For most other varibles we sum over all bins
                temp_result += expr_subst

        return temp_result

    @staticmethod
    def evaluate_binned_expression(expression, telescope_parameters, verbose=False, take_max=False):
        """
        Calculate an expression using baseline binning
        @param expression:
        @param telescope_parameters:
        @param verbose:
        @param take_max:
        @return:
        """
        tp = telescope_parameters
        bins = tp.baseline_bins         # Remove the array of baselines from the parameter dictionary
        counts = tp.nr_baselines * tp.baseline_bin_distribution # Remove the array of baselines from the parameter dictionary

        bins_unitless = bins
        assert tp.Bmax is not None
        Bmax_num_value = tp.Bmax
        # Compute the index of the first bin whose baseline exceeds the max baseline used (must be <= number of bins)
        nbins_used = min(bins_unitless.searchsorted(Bmax_num_value) + 1, len(bins))
        # Restrict the bins used to only those bins that are used
        bins = bins[:nbins_used]  # This operation creates a copy; i.e. does not modify tp.baseline_bins
        bins[nbins_used-1] = tp.Bmax
        counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

        result = Implementation.substitute_parameters_binned(expression, tp, bins, counts, verbose, take_max=take_max)
        return float(result)

    @staticmethod
    def minimize_binned_expression_by_Tsnap(expression, telescope_parameters, verbose=False, take_max=False):
        """
        Minimizes an expression by substituting the supplied telescope parameters into the expression, then minimizing it
        by varying the free parameter, Tsnap
        """
        #TODO: can make the free expression a parameter of this method (should something else than Tsnap be desired)

        tp = telescope_parameters
        bins = tp.baseline_bins # Remove the array of baselines from the parameter dictionary
        counts = tp.nr_baselines * tp.baseline_bin_distribution # Remove the array of baselines from the parameter dictionary

        bins_unitless = bins
        assert tp.Bmax is not None
        Bmax_num_value = tp.Bmax
        nbins_used = bins_unitless.searchsorted(Bmax_num_value) + 1  # Gives the index of the first bin whose baseline exceeds the max baseline used
        bins = bins[:nbins_used]  # Restrict the bins used to only those bins that are used
        bins[nbins_used-1] = tp.Bmax
        counts = counts[:nbins_used]  # Restrict the bins counts used to only those bins that are used

        result = Implementation.substitute_parameters_binned(expression, tp, bins, counts, verbose, take_max=take_max)

        # Remove string literals from the telescope_params, as they can't be evaluated by lambdify
        bound_lower = tp.Tsnap_min
        bound_upper = 0.5 * tp.Tobs

        Tsnap_optimal = Implementation.optimize_expr(result, tp.Tsnap, bound_lower, bound_upper)
        value_optimal = result.subs({tp.Tsnap : Tsnap_optimal})
        if verbose:
            print ("Tsnap has been optimized as : %f. (Cost function = %f)" % \
                  (Tsnap_optimal, value_optimal / c.peta))
        return {tp.Tsnap : Tsnap_optimal, 'value' : value_optimal}  # Replace Tsnap with its optimal value
