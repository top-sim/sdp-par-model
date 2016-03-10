"""
This file contains methods for programmatically interacting with the SKA SDP Parametric Model using Python.
"""
import copy

from parameter_definitions import Telescopes, ImagingModes, Bands, ParameterDefinitions
from equations import Equations
from implementation import Implementation as imp
import numpy as np
from sympy import Lambda, FiniteSet
import warnings

class SkaPythonAPI:
    """
    This class (SKA API) represents an API by which the SKA Parametric Model can be called programmatically, without
    making use of the iPython Notebook infrastructure.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate_expression(expression, tp, tsnap, nfacet):
        """Evaluate an expression by substituting the telescopec parameters
        into them. Depending on the type of expression, the result
        might be a value, a list of values (in case it is
        baseline-dependent) or a string (if evaluation failed).

        @param expression: the expression, expressed as a function of the telescope parameters, Tsnap and Nfacet
        @param tp: the telescope parameters (ParameterContainer object containing all relevant parameters)
        @param tsnap: The snapshot time to use
        @param nfacet: The number of facets to use
        @param key: (optional) the string name of the expression that is being evaluated. Used in error reporting.
        @return:

        """

        # Already a plain value?
        if isinstance(expression, int) or isinstance(expression, float) or \
           isinstance(expression, str) or isinstance(expression, list):
            return expression

        # Otherwise try to evaluate using sympy
            try:

            # First substitute parameters
            expression_subst = expression.subs({tp.Tsnap: tsnap, tp.Nfacet: nfacet})

            # Lambda? Assume it depends on baseline lengths
            if isinstance(expression_subst, Lambda):
                result = []
                counts = (tp.nbaselines / sum(tp.baseline_bin_distribution)) * tp.baseline_bin_distribution
                for (bcount, bmax) in zip(counts, tp.baseline_bins):
                    if expression_subst.nargs == FiniteSet(1):
                        result.append(float(expression_subst(bmax)))
                    else:
                        result.append(float(expression_subst(bcount, bmax)))
            else:
                # Otherwise just evaluate directly
                result = float(expression_subst)

            except Exception as e:
                if key is None:
                    msg = "Failed to evaluate %s with msg: %s" % (str(expression), str(e))
                else:
                    msg = "Subsitution of %s aborted with msg: %s" % (key, str(e))
                warnings.warn(msg)
        return result

    @staticmethod
    def eval_expression_default(pipelineConfig, expression='Rflop', verbose=False):
        """
        Evaluating a parameter for its default parameter value
        @param pipelineConfig:
        @param expression_string:
        @param verbose:
        """

        result = 0
        for submode in pipelineConfig.relevant_modes:
            tp = imp.calc_tel_params(pipelineConfig, verbose)

            result_expression = eval('tp.%s' % expression_string)
            (tsnap_opt, nfacet_opt) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            result += SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)

        return result

    @staticmethod
    def eval_param_sweep_1d(pipelineConfig, expression='Rflop',
                            parameter='Rccf', param_val_min=10,
                            param_val_max=10, number_steps=1,
                            verbose=False):
        """
        Evaluates an expression for a range of different parameter values, by varying the parameter linearly in
        a specified range in a number of steps

        @param pipelineConfig:
        @param expression_string: The expression that needs to be evaluated, as string (e.g. "Rflop")
        @param parameter_string: the parameter that will be swept - written as text (e.g. "Bmax")
        @param param_val_min: minimum value for the parameter's value sweep
        @param param_val_max: maximum value for the parameter's value sweep
        @param number_steps: the number of *intervals* that will be used to sweep the parameter from min to max

        @param verbose:
        @return: @raise AssertionError:
        """
        assert param_val_max > param_val_min

        print "Starting sweep of parameter %s, evaluating expression %s over range (%s, %s) in %d steps " \
              "(i.e. %d data points)" % \
              (parameter_string, expression_string, str(param_val_min), str(param_val_max), number_steps, number_steps + 1)

        param_values = np.linspace(param_val_min, param_val_max, num=number_steps + 1)

        results = []
        for i in range(len(param_values)):

            # Calculate telescope parameter with adjusted parameter
            adjusts = { parameter: str(param_values[i]) }
            tp = imp.calc_tel_params(pipelineConfig, verbose, adjusts=adjusts)

            percentage_done = i * 100.0 / len(param_values)
            print "> %.1f%% done: Evaluating %s for %s = %g" % (percentage_done, expression_string,
                                                                parameter_string, param_values[i])

            # Perform a check to see that the value of the assigned parameter wasn't changed by the imaging equations,
            # otherwise the assigned value would have been lost (i.e. not a free parameter)
            parameter_final_value = None
            exec('parameter_final_value = tp.%s' % parameter_string)
            eta = 1e-10
            if abs((parameter_final_value - param_values[i])/param_values[i]) > eta:
                raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                     'by the method compute_derived_parameters(). (%g -> %g). '
                                     'Cannot peform parameter sweep.'
                                     % (parameter_string, param_values[i], parameter_final_value))

            result_expression = eval('tp.%s' % expression_string)
            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            results.append(SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet))

        print 'done with parameter sweep!'
        return (param_values, results)

    @staticmethod
    def eval_param_sweep_2d(pipelineConfig, expression='Rflop',
                            parameters=None, params_ranges=None,
                            number_steps=2, verbose=False):
        """
        Evaluates an expression for a 2D grid of different values for two parameters, by varying each parameter
        linearly in a specified range in a number of steps. Similar to eval_param_sweep_1d, except that it sweeps
        a 2D parameter space, returning a matrix of values.

        @param telescope:
        @param mode:
        @param band:
        @param hpso:
        @param blcoal: Baseline dependent coalescing (before gridding)
        @param on_the_fly:
        @param max_baseline:
        @param nr_frequency_channels:
        @param expression_string: The expression that needs to be evalued, as string (e.g. "Rflop")
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

        n_param_x_values = number_steps + 1
        n_param_y_values = number_steps + 1
        nr_evaluations = n_param_x_values * n_param_y_values  # The number of function evaluations that will be required

        print "Evaluating expression %s while\nsweeping parameters %s and %s over 2D domain [%s, %s] x [%s, %s] in %d " \
              "steps each,\nfor a total of %d data evaluation points" % \
              (expression_string, parameters[0], parameters[1], str(params_ranges[0][0]), str(params_ranges[0][1]),
               str(params_ranges[1][0]), str(params_ranges[1][1]), number_steps, nr_evaluations)

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
                    parameters[0]: str(param_x_value),
                    parameters[1]: str(param_y_value),
                }
                tp = imp.calc_tel_params(pipelineConfig, verbose, adjusts=adjusts)

                percentage_done = (ix * n_param_y_values + iy) * 100.0 / nr_evaluations
                print "> %.1f%% done: Evaluating %s for (%s, %s) = (%s, %s)" % (percentage_done, expression_string,
                                                                                parameters[0], parameters[1],
                                                                                str(param_x_value), str(param_y_value))

                # Perform a check to see that the value of the assigned parameters weren't changed by the imaging
                # equations, otherwise the assigned values would have been lost (i.e. not free parameters)
                parameter1_final_value = None
                parameter2_final_value = None
                exec('parameter1_final_value = tp.%s' % parameters[0])
                exec('parameter2_final_value = tp.%s' % parameters[1])
                eta = 1e-10
                if abs((parameter1_final_value - param_x_value) / param_x_value) > eta:
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[0])
                if abs((parameter2_final_value - param_y_value) / param_y_value) > eta:
                    print parameter2_final_value
                    print param_y_value
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[1])

                result_expression = eval('tp.%s' % expression_string)
                (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                results[iy, ix] = SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)

        print 'done with parameter sweep!'
        return (param_x_values, param_y_values, results)

    @staticmethod
    def evaluate_expressions(expressions, tp, tsnap, nfacet):
        """
        Evaluate a sequence of expressions by substituting the telescope_parameters into them. Returns the result
        @param expressions: An array of expressions to be evaluated
        @param tp: The set of telescope parameters that should be used to evaluate each expression
        @param tsnap: The relevant (typically optimal) snapshot time
        @param nfacet: The relevant (typically optimal) number of facets to use
        """
        results = []
        for i in range(len(expressions)):
            expression = expressions[i]
            result = SkaPythonAPI.evaluate_expression(expression, tp, tsnap, nfacet)
            results.append(result)
        return results
