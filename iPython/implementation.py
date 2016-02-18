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
import math

class Implementation:

    def __init__(self):
        pass

    @staticmethod
    def seconds_to_hms(seconds):
        """
        Converts a given number of seconds into hours, minutes and seconds, returned as a tuple. Useful for display output
        @param seconds:
        @return: (hours, minutes, seconds)
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return (h, m, s)

    @staticmethod
    def cheap_lambdify_curry(free_vars, expression):
        """Translate sympy expression to an actual Python expression that can
        be evaluated quickly. This is roughly the same as sympy's
        lambdify, with a number of differences:

        1. We only support a subset of functions. Note that sympy's
           list is incomplete as well, and actually has a wrong
           translation rule for "Max".

        2. The return is curried, so for multiple "free_vars" (x, y)
           you will have to call the result as "f(x)(y)" instead of
           "f(x,y)". This means we can easily obtain a function that
           is specialised for a certain value of the outer variable.

        """

        # Do "quick & dirty" translation. This map might need updating
        # when new functions get used in equations.py
        module = {
            'Max': 'max',
            'Min': 'min',
            'ln': 'math.log',
            'log': 'math.log',
            'sqrt': 'math.sqrt',
            'sign': 'np.sign', # No sign in math, apparently
            'Abs': 'abs',
            'acos': 'math.acos',
            'acosh': 'math.acosh',
            'arg': 'np.angle',
            'asin': 'math.asin',
            'asinh': 'math.asinh',
            'atan': 'math.atan',
            'atan2': 'math.atan2',
            'atanh': 'math.atanh',
            'ceiling': 'math.ceil',
        }
        expr_body = str(expression)
        for (sympy_name, numpy_name) in module.iteritems():
            expr_body = expr_body.replace(sympy_name + '(', numpy_name + '(')

        # Create head of lambda expression
        expr_head = ''
        for free_var in free_vars:
            expr_head += 'lambda ' + str(free_var) + ':'

        # Evaluate in order to build lambda
        return eval(expr_head + expr_body)

    @staticmethod
    def optimize_lambdified_expr(lam, bound_lower, bound_upper):

        # Lower bound cannot be higher than the uppper bound.
        if bound_upper <= bound_lower:
            print 'Unable to optimize free variable as upper bound is lower that the lower bound'
            return bound_lower
        else:
            result = opt.minimize_scalar(lam,
                                         bounds=(bound_lower, bound_upper),
                                         method='bounded')
            if not result.success:
                print ('WARNING! : Was unable to optimize free variable. Using a value of: %f' % result.x)
            else:
                # print ('Optimized free variable = %f' % result.x)
                pass
            return result.x

    @staticmethod
    def calc_tel_params(telescope, mode, band=None, hpso=None, bldta=True, otfk=False,
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

        # Note the order in which these settings are applied.
        # Each one (possibly) overwrites previous definitions if they should they overlap
        # (as happens with e.g. frequency bands)

        # First: The telescope's parameters (Primarily the number of dishes, bands, beams and baselines)
        p.apply_telescope_parameters(telescope_params, telescope)
        # Then define imaging mode and frequency-band
        # Includes frequency range, Observation time, number of cycles, quality factor, number of channels, etc.
        if (hpso is None) and (band is not None):
            p.apply_band_parameters(telescope_params, band)
            p.apply_imaging_mode_parameters(telescope_params, mode)
        elif (hpso is not None) and (band is None):
            # Note the ordering; HPSO parameters get applied last, and therefore have the final say
            p.apply_imaging_mode_parameters(telescope_params, mode)
            p.apply_hpso_parameters(telescope_params, hpso)
        else:
            raise Exception("Either the Imaging Band or an HPSO needs to be defined (either or; not both).")

        # Artificially limit max_baseline or nr_frequency_channels,
        # deviating from the default for this Band or HPSO
        if max_baseline is not None:
            telescope_params.Bmax = max_baseline
        assert telescope_params.Bmax is not None
        if nr_frequency_channels is not None:
            telescope_params.Nf_max = nr_frequency_channels

        # Limit bins to those shorter than Bmax
        bins = telescope_params.baseline_bins
        nbins_used = min(bins.searchsorted(telescope_params.Bmax) + 1, len(bins))
        bins = bins[:nbins_used]

        # Same for baseline sizes
        counts = telescope_params.nr_baselines * telescope_params.baseline_bin_distribution
        counts = counts[:nbins_used]

        # Set maximum on last bin (TODO: maximum? scale "counts"?)
        bins[nbins_used-1] = telescope_params.Bmax

        # Apply imaging equations
        f.apply_imaging_equations(telescope_params, mode, bldta, bins, counts, otfk, verbose)

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
    def find_optimal_Tsnap_Nfacet(telescope_parameters, expr_to_minimize='Rflop', max_number_nfacets=20,
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

        result_per_nfacet = {}
        result_array = []
        optimal_Tsnap_array = []
        warned = False
        expression_original = None

        # Construct lambda from our two parameters (facet number and
        # snapshot time) to the expression to minimise
        exec('expression_original = telescope_parameters.%s' % expr_to_minimize)
        expression_lam = Implementation.cheap_lambdify_curry((telescope_parameters.Nfacet,
                                                              telescope_parameters.Tsnap),
                                                             expression_original)

        for nfacets in range(1, max_number_nfacets+1):  # Loop over the different integer values of NFacet
            # Warn if large values of nfacets are reached, as it may indicate an error and take long!
            if (nfacets > 20) and not warned:
                print ('Searching for minimum value by incrementing Nfacet; value of 20 exceeded... this is a bit odd '
                       '(search may take a long time; will self-terminate at Nfacet = %d' % max_number_nfacets)
                warned = True

            i = nfacets-1  # zero-based index
            if verbose:
                print ('Evaluating Nfacets = %d' % nfacets)

            result = Implementation.minimize_by_Tsnap_lambdified(expression_lam(nfacets+1),
                                                                 telescope_parameters,
                                                                 verbose=verbose)

            result_array.append(float(result['value']))
            optimal_Tsnap_array.append(result[telescope_parameters.Tsnap])
            result_per_nfacet[nfacets] = result_array[i]
            if nfacets >= 3: #
                if result_array[i] >= result_array[i-1]:
                    if verbose:
                        print ('\nExpression increasing with number of facets; aborting exploration of Nfacets > %d' \
                              % nfacets)
                    break #don't stop search after just doing Nfacet=2, do at least Nfacet=3 first, bacause there can be a local increase between nfacet=1 and 2

        index = np.argmin(np.array(result_array))
        nfacets = index + 1
        if verbose:
            print ('\n(Nfacet, Tsnap) = (%d, %.2f) yielded the lowest value of %s = %g'
                   % (nfacets,  optimal_Tsnap_array[index], expr_to_minimize, result_array[index]))

        return (optimal_Tsnap_array[index], nfacets)

    @staticmethod
    def minimize_by_Tsnap_lambdified(lam, telescope_parameters, verbose=False):

        # Compute lower & upper bounds
        tp = telescope_parameters
        bound_lower = tp.Tsnap_min
        bound_upper = 0.5 * tp.Tobs

        # Do optimisation
        Tsnap_optimal = Implementation.optimize_lambdified_expr(lam, bound_lower, bound_upper)
        value_optimal = lam(Tsnap_optimal)
        if verbose:
            print ("Tsnap has been optimized as : %f. (Cost function = %f)" % \
                  (Tsnap_optimal, value_optimal / c.peta))
        return {tp.Tsnap : Tsnap_optimal, 'value' : value_optimal}  # Replace Tsnap with its optimal value
