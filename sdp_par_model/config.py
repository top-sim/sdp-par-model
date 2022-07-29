from __future__ import print_function

# from builtins import int
import warnings

import numpy as np
import sympy
import pylru  # Install using pip (conda doesn't resolve this in my test)
import copy

from .parameters.container import ParameterContainer, BLDep
from .parameters import definitions as p
from .parameters.definitions import (Telescopes, Pipelines, Bands)
from .parameters import equations as f
# from .parameters.definitions import Constants as c
from . import evaluate

TEL_PARAM_CACHE = pylru.lrucache(1000)

class PipelineConfig:
    """
    A full SDP pipeline configuration. This collects all data required
    to parameterise a pipeline.
    """

    def __init__(self, telescope=None, pipeline=None, band=None, hpso=None, hpso_pipe=None,
                 adjusts={}, **kwargs):
        """
        :param telescope: Telescope to use (can be omitted if HPSO specified)
        :param pipeline: Pipeline mode (can be omitted if HPSO specified)
        :param band: Frequency band (can be omitted if HPSO specified)
        :param hpso: High Priority Science Objective ID (can be omitted if telescope, pipeline and band specified)
        :param hpso_pipe: The specific pipeline of the HPSO for which results are to be calculated
        :param adjusts: Values that should be adjusted in the
          telescope parameters. Keyword arguments get added to the
          adjustments automatically. Can be a string of the the form
          "name=val name2=val flag".
        """
        params = ParameterContainer()

        # Alias for now        cfg_name = PipelineConfig(pipeline=pipeline, **cfg).describe()

        if hpso_pipe is not None:
            pipeline = hpso_pipe

        # Load HPSO parameters
        if pipeline is None:
            raise ValueError("pipeline must be set!")
        if hpso is not None:
            assert hpso in p.HPSOs.hpso_telescopes
            assert pipeline is not None
            if telescope is not None or band is not None:
                raise Exception("(telescope + band) *XOR* hpso need to be set (i.e. not both)")

            self.hpso = hpso
            self.hpso_pipe = pipeline
            self.pipeline = pipeline
            self.telescope = p.HPSOs.hpso_telescopes[hpso]

            p.apply_telescope_parameters(params, self.telescope)
            p.apply_hpso_parameters(params, hpso, pipeline)
            if hasattr(params, 'pipeline'):
                pipeline = params.pipeline
        else:
            # This may be a bit of an outdated case; we mainly work in terms of HPSOs
            if (telescope is None) or (band is None):
                raise Exception("(telescope + band) *XOR* hpso need to be set (i.e. not both)")
            self.band = band
            self.telescope = telescope
            self.pipeline = pipeline
            p.apply_telescope_parameters(params, self.telescope)
            p.apply_band_parameters(params, self.band)

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

        # Store max allowed baseline length, load default parameters
        self.max_allowed_baseline = params.baseline_bins[-1]
        if 'Bmax' not in self.adjusts:
            self.adjusts['Bmax'] = params.Bmax
        self.default_frequencies = params.Nf_max
        if 'Nf_max' not in self.adjusts:
            self.adjusts['Nf_max'] = params.Nf_max

    def describe(self):
        """ Returns a name that identifies this configuration. """

        # Identify by either (HPSO + pipeline), or (pipeline + band)
        if hasattr(self, "hpso"):
            name = self.hpso + ' (' + self.pipeline + ')'
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
        Checks whether the supplied telescope and band are compatible with
        each other.
        """
        is_compatible = False
        telescope = self.telescope
        band = self.band
        if telescope in Telescopes.available_teles:
            is_compatible = (band in Bands.telescope_bands[telescope])
        else:
            raise ValueError("Unknown telescope %s" % telescope)

        return is_compatible

    def is_valid(self, pure_pipelines=True):
        """Checks integrity of the pipeline configuration.

        :return: (okay?, list of errors/warnings)
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
            if self.pipeline not in Pipelines.pure_pipelines:
                messages.append("ERROR: The '%s' imaging pipeline is currently not supported" % str(self.pipeline))
                okay = False

        # Band compatibility. Can skip for HPSOs, as they override the
        # band manually.
        if (hasattr(self, "hpso") and hasattr(self, "band")) or (not (hasattr(self, "hpso") or hasattr(self, "band"))):
                messages.append("ERROR: Either the Imaging Band or an HPSO needs to be defined (and not both).")
                okay = False

        if not (hasattr(self, "hpso") or self.telescope_and_band_are_compatible()):
            messages.append("ERROR: Telescope '%s' and band '%s' are not compatible" %
                              (str(self.telescope), str(self.band)))
            okay = False

        if not hasattr(self, "pipeline"):
            messages.append("ERROR: Configuration has no defined pipeline")
            okay = False

        return (okay, messages)

    def calc_tel_params(cfg, verbose=False, adjusts={}, symbolify='',
                        optimize_expression='Rflop',
                        clear_symbolised=None):
        """
        Calculates telescope parameters for this configuration.  Some
        values may (optionally) be overwritten, e.g. the
        maximum baseline or number of frequency channels.

        :param cfg: Valid pipeline configuration
        :param verbose: How chatty we are supposed to be
        :param adjusts: Dictionary of telescope parameters to adjust
        :param symbolify: Generate symbolified telescope parameters
        :param optimize_expression: Set free symbols in a way that minimises given telescope parameter
           (only if symbolify is not set)
        :param clear_symbolised: Whether to clear parameters with free symbols after optimisation.
           (only if symbolify is not set. Default on if optimize_expression is not None.)
        """

        assert cfg.is_valid()[0], "calc_tel_params must be called for a valid pipeline configuration!"
        if clear_symbolised is None:
            clear_symbolised = (optimize_expression is not None)

        # Check whether we have a cached copy
        tel_param_desc = (cfg.describe(), str(adjusts.items()), symbolify, optimize_expression, clear_symbolised)
        if not verbose:
            if tel_param_desc in TEL_PARAM_CACHE:
                return copy.deepcopy(TEL_PARAM_CACHE[tel_param_desc])

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

        if hasattr(cfg, "hpso"):
            # Note the ordering; HPSO parameters get applied last, and therefore have the final say
            p.apply_pipeline_parameters(telescope_params, cfg.pipeline)
            p.apply_hpso_parameters(telescope_params, cfg.hpso, cfg.pipeline)
        elif hasattr(cfg, "band"):
            p.apply_band_parameters(telescope_params, cfg.band)
            p.apply_pipeline_parameters(telescope_params, cfg.pipeline)

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

        # Free symbols to minimise?
        if symbolify == '' and optimize_expression is not None and \
           telescope_params.get(optimize_expression) is not None and \
           isinstance(telescope_params.get(optimize_expression), sympy.Expr) and \
           len(telescope_params.get(optimize_expression).free_symbols) > 0:

            # Minimise
            substs = evaluate.minimise_parameters(telescope_params, optimize_expression, verbose=verbose)
            telescope_params = telescope_params.subs(substs)

        # Clear unoptimised values?
        if symbolify == '' and clear_symbolised:
            telescope_params.clear_symbolised()

        # Cache, return
        TEL_PARAM_CACHE[tel_param_desc] = copy.deepcopy(telescope_params)
        return telescope_params

    def eval_expression_products(pipelineConfig, expression='Rflop', verbose=False):
        """
        Evaluate a parameter sum for each product

        :param pipelineConfig:  Pipeline configuration to use
        :param expression: Procuct parameter to evaluate
        :param verbose: Verbosity to use for `calc_tel_params`
        """

        values={}
        tp = pipelineConfig.calc_tel_params(verbose)

        # Loop through defined products, add to result
        for name, product in tp.products.items():
            if expression in product:
                values[name] = values.get(name, 0) + \
                               evaluate.evaluate_expression(product[expression], tp)

        return values


    def eval_param_sweep_1d(pipelineConfig, expression_string='Rflop',
                            parameter_string='Rccf', param_val_min=10,
                            param_val_max=10, number_steps=1,
                            verbose=False):
        """Evaluates an expression for a range of different parameter values,
        by varying the parameter linearly in a specified range in a
        number of steps

        :param pipelineConfig:
        :param expression_string: The expression that needs to be evaluated, as string (e.g. "Rflop")
        :param parameter_string: the parameter that will be swept - written as text (e.g. "Bmax")
        :param param_val_min: minimum value for the parameter's value sweep
        :param param_val_max: maximum value for the parameter's value sweep
        :param number_steps: the number of *intervals* that will be used to sweep
           the parameter from min to max

        :param verbose:
        :return: Pair of parameter values and results
        :raise AssertionError:
        """
        assert param_val_max > param_val_min

        print("Starting sweep of parameter %s, evaluating expression %s over range (%s, %s) in %d steps "
              "(i.e. %d data points)" %
              (parameter_string, expression_string, str(param_val_min), str(param_val_max), number_steps, number_steps + 1))

        param_values = np.linspace(param_val_min, param_val_max, num=number_steps + 1)

        results = []
        for i in range(len(param_values)):
            # Calculate telescope parameter with adjusted parameter
            adjusts = {parameter_string: param_values[i]}
            tp = pipelineConfig.calc_tel_params(verbose, adjusts=adjusts)

            percentage_done = i * 100.0 / len(param_values)
            print("> %.1f%% done: Evaluating %s for %s = %g" % (percentage_done, expression_string,
                                                                parameter_string, param_values[i]))

            # Perform a check to see that the value of the assigned parameter wasn't changed by the imaging equations,
            # otherwise the assigned value would have been lost (i.e. not a free parameter)
            parameter_final_value = tp.get(parameter_string)
            eta = 1e-10
            if abs((parameter_final_value - param_values[i])/param_values[i]) > eta:
                raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                     'by the method compute_derived_parameters(). (%g -> %g). '
                                     'Cannot peform parameter sweep.'
                                     % (parameter_string, param_values[i], parameter_final_value))

            if expression_string.find(".") >= 0:
                product, expr = expression_string.split(".")
                result_expression = tp.products[product].get(expr, 0)
            else:
                result_expression = tp.get(expression_string)
            results.append(evaluate.evaluate_expression(result_expression, tp))

        print('done with parameter sweep!')
        return (param_values, results)


    def eval_param_sweep_2d(pipelineConfig, expression_string='Rflop',
                            parameters=None, params_ranges=None,
                            number_steps=2, verbose=False):
        """
        Evaluates an expression for a 2D grid of different values for
        two parameters, by varying each parameter linearly in a
        specified range in a number of steps. Similar to
        :meth:`eval_param_sweep_1d`, except that it sweeps a 2D
        parameter space, returning a matrix of values.

        :param pipelineConfig:
        :param expression_string: The expression that needs to be evalued, as string (e.g. "Rflop")
        :param parameters: The two parameters to sweep
        :param params_ranges: Ranges to use for parameters
        :param number_steps: The number of *intervals* that will be used to sweep
           the parameters from min to max
        :param verbose:
        :returns: Triple of parameter values (both) and results
        """
        assert (parameters is not None) and (len(parameters) == 2)
        assert (params_ranges is not None) and (len(params_ranges) == 2)
        for prange in params_ranges:
            assert len(prange) == 2
            assert prange[1] > prange[0]

        n_param_x_values = number_steps + 1
        n_param_y_values = number_steps + 1
        nr_evaluations = n_param_x_values * n_param_y_values  # The number of function evaluations that will be required

        print("Evaluating expression %s while\nsweeping parameters %s and %s over 2D domain [%s, %s] x [%s, %s] in %d "
              "steps each,\nfor a total of %d data evaluation points" %
              (expression_string, parameters[0], parameters[1], str(params_ranges[0][0]), str(params_ranges[0][1]),
               str(params_ranges[1][0]), str(params_ranges[1][1]), number_steps, nr_evaluations))

        param_x_values = np.linspace(params_ranges[0][0], params_ranges[0][1], num=n_param_x_values)
        param_y_values = np.linspace(params_ranges[1][0], params_ranges[1][1], num=n_param_y_values)
        results = np.zeros((n_param_x_values, n_param_y_values))  # Create an empty numpy matrix to hold results

        # Nested 2D loop over all values for param1 and param2. Indexes iterate over y (inner loop), then x (outer loop)
        for ix in range(n_param_x_values):
            param_x_value = param_x_values[ix]
            for iy in range(n_param_y_values):
                param_y_value = param_y_values[iy]

                # Overwrite the corresponding fields of tp with the to-be-evaluated values
                adjusts = {
                    parameters[0]: param_x_value,
                    parameters[1]: param_y_value,
                }
                tp = pipelineConfig.calc_tel_params(verbose, adjusts=adjusts)

                percentage_done = (ix * n_param_y_values + iy) * 100.0 / nr_evaluations
                print("> %.1f%% done: Evaluating %s for (%s, %s) = (%s, %s)" % (percentage_done, expression_string,
                                                                                parameters[0], parameters[1],
                                                                                str(param_x_value), str(param_y_value)))

                # Perform a check to see that the value of the assigned parameters weren't changed by the imaging
                # equations, otherwise the assigned values would have been lost (i.e. not free parameters)
                parameter1_final_value = tp.get(parameters[0])
                parameter2_final_value = tp.get(parameters[1])
                eta = 1e-10
                if abs((parameter1_final_value - param_x_value) / param_x_value) > eta:
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[0])
                if abs((parameter2_final_value - param_y_value) / param_y_value) > eta:
                    print(parameter2_final_value)
                    print(param_y_value)
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[1])

                result_expression = tp.get(expression_string)
                results[iy, ix] = evaluate.evaluate_expression(result_expression, tp)

        print('done with parameter sweep!')
        return (param_x_values, param_y_values, results)


    def eval_products_symbolic(pipelineConfig, expression='Rflop', symbolify='product'):
        """
        Returns formulas for the given product property.

        :param pipelineConfig: Pipeline configuration to use.
        :param expression: Product property to query. FLOP rate by default.
        :param symbolify: How aggressively sub-formulas should be replaced by symbols.
        """

        # Create symbol-ified telescope model
        tp = pipelineConfig.calc_tel_params(symbolify=symbolify)

        # Collect equations and free variables
        eqs = {}
        for product in tp.products:
            eqs[product] = tp.products[product].get(expression, 0)
        return eqs


    def eval_symbols(pipelineConfig, symbols,
                     recursive=False, symbolify='', optimize_expression=None):
        """Returns formulas for the given symbol names. This can be used to
        look up the definitions behind sympy Symbols returned by
        eval_products_symbolic or this function.

        The returned dictionary will contain an entry for all symbols
        that we could look up sucessfully - this excludes symbols that
        are not defined or have only a tautological definition ("sym =
        sym").

        :param pipelineConfig: Pipeline configuration to use.
        :param symbols: Symbols to query
        :param recursive: Look up free symbols in symbol definitions?
        :param symbolify: How aggressively sub-formulas should be replaced by symbols.
        """

        # Create possibly symbol-ified telescope model
        tp = pipelineConfig.calc_tel_params(symbolify=symbolify)

        # Optimise to settle Tsnap and Nfacet
        if not optimize_expression is None:
            assert(symbolify == '') # Will likely fail otherwise
            tp = pipelineConfig.calc_tel_params(optimize_expression=optimize_expression)

        # Create lookup map for symbols
        symMap = {}
        for name, v in tp.__dict__.items():
            symMap[tp.make_symbol_name(name)] = v

        # Start collecting equations
        eqs = {}
        while len(symbols) > 0:
            new_symbols = set()
            for sym in symbols:
                if sym in eqs: continue

                # Look up
                if not sym in symMap: continue
                v = symMap[str(sym)]

                # If the equation is "name = name", it is not defined at this level. Push back to next level
                if isinstance(v, sympy.Symbol) and str(v) == sym:
                    continue
                eqs[str(sym)] = v
                if isinstance(v, sympy.Expr) or isinstance(v, BLDep):
                    new_symbols = new_symbols.union(evaluate.collect_free_symbols([v]))
            symbols = new_symbols
        return eqs
