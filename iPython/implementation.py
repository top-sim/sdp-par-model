"""
This Python file contains two classes.

ParameterContainer is centrally important and used throughout the iPython model, but essentially is only a container
class that is passed around between modules, and contains a set of parameters, values and variables that constitute
the inputs and outputs of computations.

Implementation contains a collection of methods for performing computations, but do not define the equations
themselves. Instead, it specifies how values are substituted, optimized, and summed across bins.
"""

from __future__ import print_function
from builtins import int

from parameter_definitions import ParameterContainer
from parameter_definitions import Telescopes, Pipelines, Bands
from parameter_definitions import ParameterDefinitions as p
from parameter_definitions import Constants as c
from equations import Equations as f
from sympy import simplify, lambdify, Max, Symbol
from scipy import optimize as opt
import numpy as np
import math
import warnings


class PipelineConfig:
    """
    A full SDP pipeline configuration. This collects all data required
    to parameterise a pipeline.
    """


    def __init__(self, telescope=None, pipeline=None, band=None, hpso=None,
                 adjusts={}, **kwargs):
        """
        @param telescope: Telescope to use (can be omitted if HPSO specified)
        @param pipeline: Pipeline mode (can be omitted if HPSO specified)
        @param band: Frequency band (can be omitted if HPSO specified)
        @param hpso: High Priority Science Objective ID (can be omitted if telescope, pipeline and band specified)
        @param adjusts: Values that should be adjusted in the
          telescope parameters. Keyword arguments get added to the
          adjustments automatically. Can be a string of the the form
          "name=val name2=val flag".
        """

        # Load HPSO parameters
        if hpso is not None:
            if not (telescope is None and band is None and pipeline is None):
                raise Exception("Either telescope/band/pipeline or an HPSO needs to be set, not both!")
            tp_default = ParameterContainer()
            p.apply_hpso_parameters(tp_default, hpso)
            telescope = tp_default.telescope
            pipeline = tp_default.pipeline
        else:
            if telescope is None or band is None or pipeline is None:
                raise Exception("Either telescope/band/pipeline or an HPSO needs to be set!")

        # Save parameters
        self.telescope = telescope
        self.hpso = hpso
        self.band = band
        self.pipeline = pipeline

        # Adjustments from keyword arguments
        if type(adjusts) == str:
            def mk_adjust(adjust):
                # Setting a field?
                fields = adjust.split('=')
                if len(fields) == 2:
                    return (fields[0], eval(fields[1]))
                # Otherwise assume that it's a flag
                return (adjust, True)
            adjusts = dict(map(mk_adjust, adjusts.split(' ')))
        self.adjusts = dict(adjusts)
        self.adjusts.update(**kwargs)
        assert 'max_baseline' not in self.adjusts, 'Please use Bmax for consistency!'

        # Determine relevant pipelines
        if isinstance(pipeline, list):
            self.relevant_pipelines = pipeline
        else:
            self.relevant_pipelines = [pipeline]

        # Load telescope parameters from apply_telescope_parameters.
        tp_default = ParameterContainer()
        p.apply_telescope_parameters(tp_default, telescope)
        if not hpso is None:
            p.apply_hpso_parameters(tp_default, hpso)

        # Store max allowed baseline length, load default parameters
        self.max_allowed_baseline = tp_default.baseline_bins[-1]
        if 'Bmax' not in self.adjusts:
            self.adjusts['Bmax'] = tp_default.Bmax
        self.default_frequencies = tp_default.Nf_max
        if 'Nf_max' not in self.adjusts:
            self.adjusts['Nf_max'] = tp_default.Nf_max

    def describe(self):
        """ Returns a name that identifies this configuration. """

        # Identify by either HPSO or telescope+band+pipeline name
        if self.hpso is not None:
            name = self.hpso
        else:
            name = self.pipeline + ' (' + self.band + ')'

        # Add modifiers
        for n, val in self.adjusts.items():
            if n == 'Nf_max' and self.adjusts[n] == self.default_frequencies:
                continue
            if n == 'Bmax' and self.adjusts[n] == self.max_allowed_baseline:
                continue
            if n == 'on_the_fly':
                n = 'otf'
            if n == 'scale_predict_by_facet':
                n = 'spbf'
            if val == True:
                name += ' [%s]' % n
            elif val == False:
                name += ' [!%s]' % n
            else:
                name += ' [%s=%s]' % (n, val)

        return name

    def telescope_and_band_are_compatible(self):
        """
        Checks whether the supplied telescope and band are compatible with each other
        @param telescope:
        @param band:
        @return:
        """
        is_compatible = False
        telescope = self.telescope
        band = self.band
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

    def is_valid(self, pure_pipelines=True):
        """Checks integrity of the pipeline configuration.

        @return: (okay?, list of errors/warnings)
        """
        messages = []
        okay = True

        # Maximum baseline
        if self.adjusts['Bmax'] > self.max_allowed_baseline:
            messages.append('WARNING: Bmax (%g m) exceeds the maximum ' \
                            'allowed baseline of %g m for telescope \'%s\'.' \
                % (self.adjusts['Bmax'], self.max_allowed_baseline, self.telescope))

        # Only pure pipelines supported?
        if pure_pipelines:
            for pipeline in self.relevant_pipelines:
                if pipeline not in Pipelines.pure_pipelines:
                    messages.append("ERROR: The '%s' imaging pipeline is currently not supported" % str(pipeline))
                    okay = False

        # Band compatibility. Can skip for HPSOs, as they override the
        # band manually.
        if self.hpso is None and not self.telescope_and_band_are_compatible():
            messages.append("ERROR: Telescope '%s' and band '%s' are not compatible" %
                              (str(self.telescope), str(self.band)))
            okay = False

        return (okay, messages)

class Implementation:

    def __init__(self):
        pass

    @staticmethod
    def is_literal(expression):
        """
        Returns true iff the expression is already a literal (e.g. float or integer) value that cannot be substituted
        or evaluated further. Used to halt attempts at further evaluating symbolic expressions
        """
        return isinstance(expression, (str, float, int, np.ndarray, list))

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
            'floor': 'math.floor'
        }
        expr_body = str(expression)
        for (sympy_name, numpy_name) in module.items():
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
        if bound_lower < bound_upper:
            result = opt.minimize_scalar(lam,
                                         bounds=(bound_lower, bound_upper),
                                         method='bounded')
            if not result.success:
                warnings.warn('WARNING! : Was unable to optimize free variable. Using a value of: %f' % result.x)
            # else:
            #     print ('Optimized free variable = %f' % result.x)
            #     pass
            return result.x
        elif bound_lower > bound_upper:
            warnings.warn('Unable to optimize free variable as upper bound %g is lower than lower bound %g.'
                          'Adhering to lower bound.' % (bound_upper, bound_lower))

            return bound_lower
        elif bound_lower == bound_upper:
            return bound_lower
        else:
            raise Exception("Computer says no.")  # This should be impossible

    @staticmethod
    def calc_tel_params(pipelineConfig, verbose=False, adjusts={}, symbolify=''):
        """
        This is a very important method - Calculates telescope parameters for a supplied band, pipeline or HPSO.
        Some default values may (optionally) be overwritten, e.g. the maximum baseline or nr of frequency channels.
        @param pipelineConfig: Valid pipeline configuration
        @param verbose:
        @param adjusts: Dictionary of telescope parameters to adjust
        """

        cfg = pipelineConfig

        telescope_params = ParameterContainer()
        p.apply_global_parameters(telescope_params)
        p.define_symbolic_variables(telescope_params)

        # Note the order in which these settings are applied.
        # Each one (possibly) overwrites previous definitions if they should they overlap
        # (as happens with e.g. frequency bands)

        # First: The telescope's parameters (Primarily the number of dishes, bands, beams and baselines)
        p.apply_telescope_parameters(telescope_params, cfg.telescope)
        # Then define pipeline and frequency-band
        # Includes frequency range, Observation time, number of cycles, quality factor, number of channels, etc.
        if (cfg.hpso is None) and (cfg.band is not None):
            p.apply_band_parameters(telescope_params, cfg.band)
            p.apply_pipeline_parameters(telescope_params, cfg.pipeline)
        elif (cfg.hpso is not None) and (cfg.band is None):
            # Note the ordering; HPSO parameters get applied last, and therefore have the final say
            p.apply_pipeline_parameters(telescope_params, cfg.pipeline)
            p.apply_hpso_parameters(telescope_params, cfg.hpso)
        else:
            raise Exception("Either the Imaging Band or an HPSO needs to be defined (either or; not both).")

        # Apply parameter adjustments. Needs to be done before bin
        # calculation in case Bmax gets changed.  Note that an
        # overwrite is required, i.e. the parameter must exist.
        for par, value in cfg.adjusts.items():
            telescope_params.__dict__[par] = value
        for par, value in adjusts.items():
            telescope_params.__dict__[par] = value

        if telescope_params.blcoal:
            # Limit bins to those shorter than Bmax
            bins = telescope_params.baseline_bins
            nbins_used = min(bins.searchsorted(telescope_params.Bmax) + 1, len(bins))
            bins = bins[:nbins_used]

            # Same for baseline sizes. Note that we normalise /before/
            # reducing the list.
            binfracs = telescope_params.baseline_bin_distribution
            binfracs /= sum(binfracs)
            binfracs = binfracs[:nbins_used]

            # Calculate old and new bin sizes
            binsize = bins[nbins_used-1]
            binsizeNew = telescope_params.Bmax
            if nbins_used > 1:
                binsize -= bins[nbins_used-2]
                binsizeNew -= bins[nbins_used-2]

            # Scale last bin
            bins[nbins_used-1] = telescope_params.Bmax
            binfracs[nbins_used-1] *= float(binsizeNew) / float(binsize)
            if verbose:
                print("Baseline coalescing on")
        else:
            if verbose:
                print("Baseline coalescing off")
            telescope_params.baseline_bins = np.array((telescope_params.Bmax,))  # m
            telescope_params.baseline_bin_distribution = np.array((1.0,))
            bins = [telescope_params.Bmax]
            binfracs=[1.0]

        # Apply imaging equations
        f.apply_imaging_equations(telescope_params, cfg.pipeline,
                                  bins, binfracs,
                                  verbose, symbolify)
        return telescope_params

    @staticmethod
    def find_optimal_Tsnap_Nfacet(telescope_parameters, expr_to_minimize_string='Rflop',
                                  max_number_nfacets=20, min_number_nfacets=1,
                                  verbose=False):
        """Computes the optimal value for Tsnap and Nfacet that minimizes the
        value of an expression (typically Rflop). Returns result as a
        2-tuple (Tsnap_opt, Nfacet_opt)

        @param telescope_parameters: Contains the definition of the
          expression that needs to be minimzed. This should be a
          symbolic expression that involves Tsnap and/or Nfacet.
        @param expr_to_minimize_string: The expression that should be
          minimized. This is typically assumed to be the computational
          load, but may also be, for example, buffer size.
        @param max_number_nfacets: Provides an upper limit to
          Nfacet. Because we currently do a linear search for the
          minimum value, using a for loop, we need to know when to
          quit. Max should never be reached unless in pathological
          cases
        @param verbose:

        """
        assert isinstance(telescope_parameters, ParameterContainer)
        assert hasattr(telescope_parameters, expr_to_minimize_string)

        if telescope_parameters.pipeline not in Pipelines.imaging: # Not imaging, return defaults
            if verbose:
                print(telescope_parameters.pipeline, "not imaging - no need to optimise Tsnap and Nfacet")
            return (telescope_parameters.Tobs, 1)

        # Construct lambda from our two parameters (facet number and
        # snapshot time) to the expression to minimise
        expression_original = eval('telescope_parameters.%s' % expr_to_minimize_string)
        params = []
        if isinstance(telescope_parameters.Nfacet, Symbol):
            params.append(telescope_parameters.Nfacet)
            nfacet_range = range(min_number_nfacets, max_number_nfacets+1)
        else:
            nfacet_range = [telescope_parameters.Nfacet]
        if isinstance(telescope_parameters.Tsnap, Symbol):
            params.append(telescope_parameters.Tsnap)
        expression_lam = Implementation.cheap_lambdify_curry(params, expression_original)

        # Loop over the different integer values of NFacet
        results = []
        warned = False
        for nfacets in nfacet_range:
            # Warn if large values of nfacets are reached, as it may indicate an error and take long!
            if (nfacets > 20) and not warned:
                warnings.warn('Searching minimum value by incrementing Nfacet; value of 20 exceeded... this is odd '
                              '(search may take a long time; will self-terminate at Nfacet = %d' % max_number_nfacets)
                warned = True

            if verbose:
                print ('Evaluating Nfacets = %d' % nfacets)

            # Find optimal Tsnap for this number of facets, obtaining result in "result"
            if isinstance(telescope_parameters.Nfacet, Symbol):
                expr = expression_lam(nfacets)
            else:
                expr = expression_lam
            if isinstance(telescope_parameters.Tsnap, Symbol):
                result = Implementation.minimize_by_Tsnap_lambdified(expr,
                                                                     telescope_parameters,
                                                                     verbose=verbose)
            else:
                result = (telescope_parameters.Tsnap, float(expr))
            results.append((nfacets, result[0], result[1]))

            # Continue to at least Nfacet==3 as there can be a local
            # increase between nfacet=1 and 2
            if len(results) >= 3:
                if results[-1][2] >= results[-2][2]:
                    if verbose:
                        print ('\nExpression increasing with number of facets; aborting exploration of Nfacets > %d' \
                              % nfacets)
                    break

        # Return parameters with lowest value
        nfacets, tsnap, val = results[np.argmin(np.array(results)[:,2])]
        if verbose:
            print ('\n(Nfacet, Tsnap) = (%d, %.2f) yielded the lowest value of %s = %g'
                   % (nfacets, tsnap, expr_to_minimize_string, val))
        return (tsnap, nfacets)

    @staticmethod
    def minimize_by_Tsnap_lambdified(lam, telescope_parameters, verbose=False):
        """
        The supplied lambda expression (a function of Tnsap) is minimized.
        @param lam: The lambda expression (a function of Tsnap)
        @param telescope_parameters: The telescope parameters
        @param verbose:
        @return: The optimal Tsnap value, along with the optimal value (as a pair)
        """
        assert telescope_parameters.pipeline in Pipelines.imaging

        # Compute lower & upper bounds
        bound_lower = telescope_parameters.Tsnap_min
        bound_upper = max(bound_lower, 0.5 * telescope_parameters.Tobs)

        # Do optimisation
        Tsnap_optimal = Implementation.optimize_lambdified_expr(lam, bound_lower, bound_upper)
        value_optimal = lam(Tsnap_optimal)
        if verbose:
            print ("Tsnap has been optimized as : %f. (Cost function = %f)" % \
                  (Tsnap_optimal, value_optimal / c.peta))
        return (Tsnap_optimal, value_optimal)  # Replace Tsnap with optimal value
