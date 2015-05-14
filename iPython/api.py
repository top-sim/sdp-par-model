"""
This file contains methods for programmatically interacting with the SKA SDP Parametric Model using Python.
"""
import copy

from parameter_definitions import Telescopes, ImagingModes, Bands, ParameterDefinitions
from formulae import Formulae
from implementation import Implementation as imp
from implementation import ParameterContainer
import numpy as np


class SKAAPI:
    """
    This class (SKA API) represents an API by which the SKA Parametric Model can be called programmatically, without
    making use of the iPython Notebook infrastructure.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate_expression(expression, tp, tsnap, nfacet):
        """
        Evaluate an expression by substituting the telescopecparameters into them. Returns the result
        @param expression: the expression, expressed as a function of the telescope parameters, Tsnap and Nfacet
        @param tp: the telescope parameters (ParameterContainer object containing all relevant parameters)
        @param tsnap: The snapshot time to use
        @param nfacet: The number of facets to use
        @return:
        """
        try:
            expression_subst = expression.subs({tp.Tsnap: tsnap, tp.Nfacet: nfacet})
            result = imp.evaluate_binned_expression(expression_subst, tp)
        except Exception as e:
            result = expression
        return result

    @staticmethod
    def eval_param_sweep_1d(telescope, mode, band=None, hpso=None, bldta=False,
                            max_baseline=None, nr_frequency_channels=None, expression='Rflop',
                            parameter='Rccf', param_val_min=10, param_val_max=10, number_steps=1, verbose=False):
        """
        Evaluates an expression for a range of different parameter values, by varying the parameter linearly in
        a specified range in a number of steps

        @param telescope:
        @param mode:
        @param band:
        @param hpso:
        @param bldta:
        @param max_baseline:
        @param nr_frequency_channels:
        @param expression: The expression that needs to be evaluated, written as text (e.g. "Rflop")
        @param parameter: the parameter that will be swept - written as text (e.g. "Bmax")
        @param param_val_min: minimum value for the parameter's value sweep
        @param param_val_max: maximum value for the parameter's value sweep
        @param number_steps: the number of *intervals* that will be used to sweep the parameter from min to max

        @param verbose:
        @return: @raise AssertionError:
        """
        assert param_val_max > param_val_min

        print "Starting sweep of parameter %s, evaluating expression %s over range (%s, %s) in %d steps " \
              "(i.e. %d data points)" % \
              (parameter, expression, str(param_val_min), str(param_val_max), number_steps, number_steps+1)

        telescope_params = imp.calc_tel_params(telescope, mode, band, hpso, bldta, max_baseline,
                                               nr_frequency_channels, verbose)

        param_values = np.linspace(param_val_min, param_val_max, num=number_steps + 1)
        results = []
        for i in range(len(param_values)):
            tp = copy.deepcopy(telescope_params)
            param_value = param_values[i]

            exec ('tp.%s = %g' % (parameter, param_value))

            if verbose:
                print ">> Evaluating %s for %s = %s" % (expression, parameter, str(param_value))

            Formulae.compute_derived_parameters(tp, mode, bldta, verbose)
            parameter_final_value = None
            exec('parameter_final_value = tp.%s' % parameter)

            if parameter_final_value != param_value:
                raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                     'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                     % parameter)

            percentage_done = i * 100.0 / len(param_values)
            print "> %.1f%% done: Evaluating %s for %s = %s" % (percentage_done, expression,
                                                                parameter, str(param_value))

            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            result_expression = eval('tp.%s' % expression)
            results.append(SKAAPI.evaluate_expression(result_expression, tp, tsnap, nfacet))

        print 'done with parameter sweep!'
        return (param_values, results)

    @staticmethod
    def eval_param_sweep_2d(telescope, mode, band=None, hpso=None, bldta=False,
                            max_baseline=None, nr_frequency_channels=None, expression='Rflop',
                            parameters=None, params_ranges=None, number_steps=2, verbose=False):
        """
        Evaluates an expression for a 2D grid of different values for two parameters, by varying each parameter
        linearly in a specified range in a number of steps. Similar to eval_param_sweep_1d, except that it sweeps
        a 2D parameter space, returning a matrix of values.

        @param telescope:
        @param mode:
        @param band:
        @param hpso:
        @param bldta:
        @param max_baseline:
        @param nr_frequency_channels:
        @param expression:
        @param parameters:
        @param params_ranges:
        @param number_steps:
        @param verbose:
        @return:
        """
        assert (parameters is not None) and (len(parameters) == 2)
        assert (params_ranges is not None) and (len(params_ranges) == 2)
        for prange in params_ranges:
            assert len(prange) == 2
            assert prange[1] > prange[0]

        print "Evaluating expression %s while\nsweeping parameters %s and %s over 2D domain [%s, %s] x [%s, %s] in %d " \
              "steps each,\nfor a total of %d data evaluation points" % \
              (expression, parameters[0], parameters[1], str(params_ranges[0][0]), str(params_ranges[0][1]),
               str(params_ranges[1][0]), str(params_ranges[1][1]), number_steps, (number_steps+1)**2)

        telescope_params = imp.calc_tel_params(telescope, mode, band, hpso, bldta, max_baseline,
                                               nr_frequency_channels, verbose)

        param1_values = np.linspace(params_ranges[0][0], params_ranges[0][1], num=number_steps + 1)
        param2_values = np.linspace(params_ranges[1][0], params_ranges[1][1], num=number_steps + 1)
        results = np.zeros((len(param1_values), len(param2_values)))  # Create an empty numpy matrix to hold results

        nr_evaluations = len(param1_values) * len(param2_values)

        # Nested 2D loop over all values for param1 and param2
        for i in range(len(param1_values)):
            param1_value = param1_values[i]
            for j in range(len(param2_values)):
                param2_value = param2_values[j]

                tp = copy.deepcopy(telescope_params)

                exec ('tp.%s = %g' % (parameters[0], param1_value))
                exec ('tp.%s = %g' % (parameters[1], param2_value))

                # Look at value directly after assignment
                param1_value = eval('tp.%s' % parameters[0])
                param2_value = eval('tp.%s' % parameters[1])

                parameter1_final_value = None
                parameter2_final_value = None
                exec('parameter1_final_value = tp.%s' % parameters[0])
                exec('parameter2_final_value = tp.%s' % parameters[1])

                if parameter1_final_value != param1_value:
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[0])
                if parameter2_final_value != param2_value:
                    print parameter2_final_value
                    print param2_value
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[1])

                percentage_done = (i*len(param2_values) + j) * 100.0 / nr_evaluations
                print "> %.1f%% done: Evaluating %s for (%s, %s) = (%s, %s)" % (percentage_done, expression,
                                                                                 parameters[0], parameters[1],
                                                                                 str(param1_value), str(param2_value))

                Formulae.compute_derived_parameters(tp, mode, bldta, verbose)
                (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                result_expression = eval('tp.%s' % expression)

                results[i,j] = SKAAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)

        print 'done with parameter sweep!'
        return (param1_values, param2_values, results)

    @staticmethod
    def evaluate_expressions(expressions, tp, tsnap, nfacet):
        """
        Evaluate a sequence of expressions by substituting the telescope_parameters into them. Returns the result
        """
        results = []
        for expression in expressions:
            result = SKAAPI.evaluate_expression(expression, tp, tsnap, nfacet)
            results.append(result)
        return results

    @staticmethod
    def compute_results(telescope, band, mode, max_baseline=None, nr_frequency_channels=None, BL_dep_time_av=False,
                        verbose=False):
        """
        Computes a set of results for a given telescope in a given band and mode. The max baseline and number of
        frequency channels may be specified as well, if the defaults are not to be used.
        @param telescope
        @param band:
        @param mode:
        @param max_baseline: (optional) the maximum baseline to be used
        @param nr_frequency_channels: (optional) the maximum number of frequency channels
        @return: a dictionary of result values
        """
        assert imp.telescope_and_band_are_compatible(telescope, band)

        # And now the results:
        tp_basic = ParameterContainer()
        ParameterDefinitions.apply_telescope_parameters(tp_basic, telescope)
        max_allowed_baseline = tp_basic.baseline_bins[-1]
        if max_baseline is not None:
            assert max_baseline <= max_allowed_baseline
        else:
            max_baseline = max_allowed_baseline

        tp = imp.calc_tel_params(telescope=telescope, mode=mode, band=band, bldta=BL_dep_time_av,
                                 max_baseline=max_baseline, nr_frequency_channels=nr_frequency_channels,
                                 verbose=verbose)  # Calculate the telescope parameters

        (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)

        # The following variables will be evaluated:
        result_variable_strings = ('Mbuf_vis', 'Mw_cache',
                                   'Npix_linear', 'Rio', 'Rflop',
                                   'Rflop_grid', 'Rflop_fft', 'Rflop_proj', 'Rflop_conv', 'Rflop_phrot')

        # These are the descriptions of the variables defined above
        result_titles = ('Visibility Buffer', 'Working (cache) memory',
                         'Image side length', 'I/O Rate', 'Total Compute Requirement',
                         'rflop_grid', 'rflop_fft', 'rflop_proj', 'rflop_conv', 'rflop_phrot')

        # And the units of the variables, as they are displayed on screen
        result_units = ('PetaBytes', 'TeraBytes', 'pixels', 'TeraBytes/s', 'PetaFLOPS',
                        'PetaFLOPS', 'PetaFLOPS', 'PetaFLOPS', 'PetaFLOPS', 'PetaFLOPS')

        assert len(result_variable_strings) == len(result_titles)  # sanity check
        assert len(result_variable_strings) == len(result_units)  # sanity check

        # The results are returned as a dictionary
        result_dict = {'Tsnap': tsnap, 'NFacet': nfacet}
        for variable_string in result_variable_strings:
            result_expression = eval('tp.%s' % variable_string)
            result = SKAAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)
            result_dict[variable_string] = result

        return result_dict


