"""
This Python file contains two classes.

ParameterContainer is centrally important and used throughout the iPython model, but essentially is only a container
class that is passed around between modules, and contains a set of parameters, values and variables that constitute
the inputs and outputs of computations.

Implementation contains a collection of methods for performing computations, but do not define the equations
themselves. Instead, it specifies how values are substituted, optimized, and summed across bins.
"""

from parameter_definitions import ParameterContainer
from parameter_definitions import Telescopes, ImagingModes, Bands
from parameter_definitions import ParameterDefinitions as p
from parameter_definitions import Constants as c
from equations import Equations as f
from sympy import simplify, lambdify, Max
from scipy import optimize as opt
import numpy as np


class Implementation:
    EXPR_NOT_SUMMED = ('Npix_linear',)

    def __init__(self):
        pass

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
    def calc_tel_params(telescope, mode, band=None, hpso=None, bldta=False, otfk=False,
                        max_baseline=None, nr_frequency_channels=None, verbose=False):
        """
        This is a very important method - Calculates telescope parameters for a supplied band, mode or HPSO.
        Some default values may (optionally) be overwritten, e.g. the maximum baseline or nr of frequency channels.
        @param telescope:
        @param mode: (can be omitted if HPSO specified)
        @param band: (can be omitted if HPSO specified)
        @param hpso: High Priority Science Objective ID (can be omitted if band specified)
        @param bldta: Baseline dependent time averaging
        @param otfk: On the fly kernels (True or False)
        @param max_baseline:
        @param nr_frequency_channels:
        @param verbose:
        """
        telescope_params = ParameterContainer()
        p.apply_global_parameters(telescope_params)
        p.define_symbolic_variables(telescope_params)

        assert not ((band is None) and (hpso is None))  # At least one of the two must be True
        assert (band is None) or (hpso is None)  # These are mutually exclusive

        # Note the order in which these settings are applied.
        # Each one (possibly) overwrites previous definitions if they should they overlap
        # (as happens with e.g. frequency bands)

        # First: The telescope's parameters (Primarily the number of dishes, bands, beams and baselines)
        p.apply_telescope_parameters(telescope_params, telescope)

        # Then: Imaging mode and Frequency-band (Order depends on whether we are dealing with a defined HPSO)
        # Including: frequency range, Observation time, number of cycles, quality factor, number of channels, etc
        if band is not None:
            p.apply_band_parameters(telescope_params, band)
            p.apply_imaging_mode_parameters(telescope_params, mode)
        else:
            # This has to be an HPSO case
            p.apply_hpso_parameters(telescope_params, hpso)
            p.apply_imaging_mode_parameters(telescope_params, mode)

        if max_baseline is not None:
            telescope_params.Bmax = max_baseline
        if nr_frequency_channels is not None:
            telescope_params.Nf_max = nr_frequency_channels

        f.apply_imaging_equations(telescope_params, mode, bldta, otfk, verbose)
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
            is_compatible = (band in Bands.low_bands_ska2)
        elif telescope == Telescopes.SKA2_Mid:
            is_compatible = (band in Bands.mid_bands_ska2)
        else:
            raise ValueError("Unknown telescope %s" % telescope)

        return is_compatible

    @staticmethod
    def find_optimal_Tsnap_Nfacet(telescope_parameters, expr_to_minimize='Rflop', max_number_nfacets=200,
                                  verbose=False):
        """
        Computes the optimal value for Tsnap and Nfacet that minimizes the value of an expression (typically Rflop)
        Returns result as a 2-tuple (Tsnap_opt, Nfacet_opt)

        @param telescope_parameters: Contains the definition of the expression that needs to be minimzed. This should
                                     be a symbolic expression that involves Tsnap and/or Nfacet.
        @param expr_to_minimize: The expression that should be minimized. This is typically assumed to be the
                                 computational load, but may also be, for example, buffer size.
        @param max_number_nfacets: Provides an upper limit to Nfacet. Because we currently do a linear search for the
                                   minimum value, using a for loop, we need to know when to quit. Max should never be
                                   reached unless in pathological cases
        @param verbose:
        """
        assert isinstance(telescope_parameters, ParameterContainer)
        assert hasattr(telescope_parameters, expr_to_minimize)
        take_max = expr_to_minimize in Implementation.EXPR_NOT_SUMMED

        result_per_nfacet = {}
        result_array = []
        optimal_Tsnap_array = []
        warned = False
        expression_original = None

        for nfacets in range(1, max_number_nfacets+1):  # Loop over the different integer values of NFacet
            exec('expression_original = telescope_parameters.%s' % expr_to_minimize)
            # Warn if large values of nfacets are reached, as it may indicate an error and take long!
            if (nfacets > 20) and not warned:
                print ('Searching for minimum value by incrementing Nfacet; value of 20 exceeded... this is a bit odd '
                       '(search may take a long time; will self-terminate at Nfacet = %d' % max_number_nfacets)
                warned = True

            i = nfacets-1  # zero-based index
            if verbose:
                print ('Evaluating Nfacets = %d' % nfacets)

            expression = expression_original.subs({telescope_parameters.Nfacet : nfacets})
            result = Implementation.minimize_binned_expression_by_Tsnap(expression, telescope_parameters,
                                                                        take_max=take_max, verbose=verbose)

            result_array.append(float(result['value']))
            optimal_Tsnap_array.append(result[telescope_parameters.Tsnap])
            result_per_nfacet[nfacets] = result_array[i]
            if nfacets >= 2:
                if result_array[i] >= result_array[i-1]:
                    if verbose:
                        print ('\nExpression increasing with number of facets; aborting exploration of Nfacets > %d' \
                              % nfacets)
                    break

        index = np.argmin(np.array(result_array))
        nfacets = index + 1
        if verbose:
            print ('\n(Nfacet, Tsnap) = (%d, %.2f) yielded the lowest value of %s = %g'
                   % (nfacets,  optimal_Tsnap_array[index], expr_to_minimize, result_array[index]))

        return (optimal_Tsnap_array[index], nfacets)

    @staticmethod
    def substitute_parameters_binned(expression, tp, bins, counts, take_max, verbose=False):
        """
        Substitute relevant variables for each bin, by defaukt summing the result. If take_max == True, then
        the maximum expression value over all bins is returns instead of the sum.
        @param expression:
        @param tp: ParameterContainer containing the telescope parameters
        @param bins: An array containing the max baseline length of each bin
        @param counts: The number of baselines in each of the bins
        @param verbose:
        @param take_max: iff True, returns the maximum value across bins, instead of the bins' values' sum.
        """
        nbins_used = len(bins)
        assert nbins_used == len(counts)
        nbaselines = sum(counts)
        temp_result = 0
        for i in range(nbins_used):
            binfrac_value = float(counts[i]) / nbaselines  # NB: Ensure that this is a floating point division
            # Substitute bin-dependent variables
            if not (isinstance(expression, (int, long)) or isinstance(expression, float)):
                expr_subst = expression.subs({tp.Bmax_bin: bins[i], tp.binfrac : binfrac_value})
            else:
                expr_subst = expression

            if take_max:  # For example when computing Npix, we take the max
                temp_result = Max(temp_result, expr_subst)
            else:         # For most other varibles we sum over all bins
                temp_result += expr_subst

        return temp_result

    @staticmethod
    def evaluate_binned_expression(expression, telescope_parameters, take_max, verbose=False):
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

        result = Implementation.substitute_parameters_binned(expression, tp, bins, counts, take_max=take_max,
                                                             verbose=verbose)
        return float(result)

    @staticmethod
    def minimize_binned_expression_by_Tsnap(expression, telescope_parameters, take_max, verbose=False):
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

        result = Implementation.substitute_parameters_binned(expression, tp, bins, counts, take_max=take_max,
                                                             verbose=verbose)

        # Remove string literals from the telescope_params, as they can't be evaluated by lambdify
        bound_lower = tp.Tsnap_min
        bound_upper = 0.5 * tp.Tobs

        Tsnap_optimal = Implementation.optimize_expr(result, tp.Tsnap, bound_lower, bound_upper)
        value_optimal = result.subs({tp.Tsnap : Tsnap_optimal})
        if verbose:
            print ("Tsnap has been optimized as : %f. (Cost function = %f)" % \
                  (Tsnap_optimal, value_optimal / c.peta))
        return {tp.Tsnap : Tsnap_optimal, 'value' : value_optimal}  # Replace Tsnap with its optimal value
