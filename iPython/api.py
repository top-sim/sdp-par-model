"""
This file contains methods for programmatically interacting with the SKA SDP Parametric Model using Python.
"""
import copy

from parameter_definitions import Telescopes, ImagingModes, Bands, ParameterDefinitions
from equations import Equations
from implementation import Implementation as imp
from parameter_definitions import ParameterContainer
import numpy as np


class SkaPythonAPI:
    """
    This class (SKA API) represents an API by which the SKA Parametric Model can be called programmatically, without
    making use of the iPython Notebook infrastructure.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate_expression(expression, tp, tsnap, nfacet, take_max):
        """
        Evaluate an expression by substituting the telescopecparameters into them. Returns the result
        @param expression: the expression, expressed as a function of the telescope parameters, Tsnap and Nfacet
        @param tp: the telescope parameters (ParameterContainer object containing all relevant parameters)
        @param tsnap: The snapshot time to use
        @param nfacet: The number of facets to use
        @param take_max: True iff the expression's maximum value across bins is returned instead of its sum
        @return:
        """
        try:
            expression_subst = expression.subs({tp.Tsnap: tsnap, tp.Nfacet: nfacet})
            result = imp.evaluate_binned_expression(expression_subst, tp, take_max=take_max)
        except Exception as e:
            result = expression
        return result

    @staticmethod
    def eval_param_sweep_1d(telescope, mode, band=None, hpso=None, bldta=False, on_the_fly=False,
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
        @param on_the_fly:
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
        take_max = expression in imp.EXPR_NOT_SUMMED  # For these the bins' values are not summed; max taken instead

        print "Starting sweep of parameter %s, evaluating expression %s over range (%s, %s) in %d steps " \
              "(i.e. %d data points)" % \
              (parameter, expression, str(param_val_min), str(param_val_max), number_steps, number_steps+1)

        telescope_params = imp.calc_tel_params(telescope, mode, band, hpso, bldta, on_the_fly, max_baseline,
                                               nr_frequency_channels, verbose)

        param_values = np.linspace(param_val_min, param_val_max, num=number_steps + 1)

        results = []
        for i in range(len(param_values)):
            tp = copy.deepcopy(telescope_params)
            exec('tp.%s = param_values[i]' % parameter)  # Assigns the parameter to its temporary evaluation-value

            percentage_done = i * 100.0 / len(param_values)
            print "> %.1f%% done: Evaluating %s for %s = %g" % (percentage_done, expression,
                                                                parameter, param_values[i])

            Equations.apply_imaging_equations(tp, mode, bldta, on_the_fly, verbose)  # modifies tp in-place

            # Perform a check to see that the value of the assigned parameter wasn't changed by the imaging equations,
            # otherwise the assigned value would have been lost (i.e. not a free parameter)
            parameter_final_value = None
            exec('parameter_final_value = tp.%s' % parameter)
            if parameter_final_value != param_values[i]:
                raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                     'by the method compute_derived_parameters(). (%g -> %g). '
                                     'Cannot peform parameter sweep.'
                                     % (parameter, param_values[i], parameter_final_value))

            result_expression = eval('tp.%s' % expression)
            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            results.append(SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet, take_max))

        print 'done with parameter sweep!'
        return (param_values, results)

    @staticmethod
    def eval_param_sweep_2d(telescope, mode, band=None, hpso=None, bldta=False, on_the_fly=False,
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
        @param on_the_fly:
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

        take_max = expression in imp.EXPR_NOT_SUMMED  # For these the bins' values are not summed; max taken instead

        print "Evaluating expression %s while\nsweeping parameters %s and %s over 2D domain [%s, %s] x [%s, %s] in %d " \
              "steps each,\nfor a total of %d data evaluation points" % \
              (expression, parameters[0], parameters[1], str(params_ranges[0][0]), str(params_ranges[0][1]),
               str(params_ranges[1][0]), str(params_ranges[1][1]), number_steps, (number_steps+1)**2)

        telescope_params = imp.calc_tel_params(telescope, mode, band, hpso, bldta, otfk=on_the_fly,
                                               max_baseline=max_baseline, nr_frequency_channels=nr_frequency_channels,
                                               verbose=verbose)

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

                # Assigns the parameters to their temporary evaluation-values
                exec('tp.%s = param1_value' % parameters[0])
                exec('tp.%s = param2_value' % parameters[1])

                percentage_done = (i*len(param2_values) + j) * 100.0 / nr_evaluations
                print "> %.1f%% done: Evaluating %s for (%s, %s) = (%s, %s)" % (percentage_done, expression,
                                                                                 parameters[0], parameters[1],
                                                                                 str(param1_value), str(param2_value))

                Equations.apply_imaging_equations(tp, mode, bldta, on_the_fly, verbose)   # modifies tp in-place

                # Perform a check to see that the value of the assigned parameters weren't changed by the imaging
                # equations, otherwise the assigned values would have been lost (i.e. not free parameters)
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

                result_expression = eval('tp.%s' % expression)
                (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                results[i, j] = SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet, take_max)

        print 'done with parameter sweep!'
        return (param1_values, param2_values, results)

    @staticmethod
    def evaluate_expressions(expressions, tp, tsnap, nfacet, take_maxima):
        """
        Evaluate a sequence of expressions by substituting the telescope_parameters into them. Returns the result
        @param expressions: An array of expressions to be evaluated
        @param tp: The set of telescope parameters that should be used to evaluate each expression
        @param tsnap: The relevant (typically optimal) snapshot time
        @param nfacet: The relevant (typically optimal) number of facets to use
        @param take_maxima: An array of boolean values, true if the corresponding variable's max across all bine
                            should be used instead of its sum
        """
        results = []
        assert len(expressions) == len(take_maxima)
        for i in range(len(expressions)):
            expression = expressions[i]
            take_max = take_maxima[i]
            result = SkaPythonAPI.evaluate_expression(expression, tp, tsnap, nfacet, take_max)
            results.append(result)
        return results
