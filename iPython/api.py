"""
This file contains methods for programmatically interacting with the SKA SDP Parametric Model using Python.
"""
import copy

from parameter_definitions import Telescopes, ImagingModes, Bands, ParameterDefinitions
from equations import Equations
from implementation import Implementation as imp
import numpy as np
import warnings


class SkaPythonAPI:
    """
    This class (SKA API) represents an API by which the SKA Parametric Model can be called programmatically, without
    making use of the iPython Notebook infrastructure.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate_all_expressions(tp, tsnap, nfacet):
        """
        Because a lot of time is consumed by computing optimal tsnap and nfacet values, we might as well evaluate
        *all* the parameters of tp by substituting these values into the fields of tp.
        For this usecase we sum across all bins for each parameter (will be the wrong thing to do for some; use
        evaluate_expression separately for those)
        @param tp: the telescope parameters (ParameterContainer object containing all relevant parameters)
        @param tsnap: The snapshot time to use
        @param nfacet: The number of facets to use
        @return:
        """
        fields = tp.__dict__
        tp_eval = copy.deepcopy(tp)
        for key in fields.keys():
            expression = fields[key]
            # For some reason we have to use a 'clean' tp copy to evaluate each expression, otherwise we run into
            # all sorts of recursion depth issues in the evaluate_expression method. Not sure why...
            tp_temp = copy.deepcopy(tp)
            expression_eval = SkaPythonAPI.evaluate_expression(expression, tp_temp, tsnap, nfacet, False, key)
            # We now write the evaluated expression into tp_eval
            exec("tp_eval.%s = expression_eval" % key)
        return tp_eval

    @staticmethod
    def evaluate_expression(expression, tp, tsnap, nfacet, take_max, key=None):
        """
        Evaluate an expression by substituting the telescopecparameters into them. Returns the result
        @param expression: the expression, expressed as a function of the telescope parameters, Tsnap and Nfacet
        @param tp: the telescope parameters (ParameterContainer object containing all relevant parameters)
        @param tsnap: The snapshot time to use
        @param nfacet: The number of facets to use
        @param take_max: True iff the expression's maximum value across bins is returned instead of its sum
        @param key: (optional) the string name of the expression that is being evaluated. Used in error reporting.
        @return:
        """
        # Literal expressions need not be evaluated, because they are already final
        if imp.is_literal(expression):
            result = expression
        else:
            try:
                expression_subst = expression.subs({tp.Tsnap: tsnap, tp.Nfacet: nfacet})
                result = imp.evaluate_binned_expression(expression_subst, tp, take_max=take_max)
            except Exception as e:
                if key is None:
                    msg = "Subsitution aborted with msg: %s" % str(e)
                else:
                    msg = "Subsitution of %s aborted with msg: %s" % (key, str(e))
                warnings.warn(msg)
                result = expression
        return result

    @staticmethod
    def eval_expression_default(telescope, mode, band=None, hpso=None, bldta=True, on_the_fly=False,
                                   max_baseline=None, nr_frequency_channels=None, expression='Rflop', verbose=False):
        """
        Evaluating a parameter for its default parameter value
        @param telescope:
        @param mode:
        @param band:
        @param hpso:
        @param bldta:
        @param on_the_fly:
        @param max_baseline:
        @param nr_frequency_channels:
        @param expression:
        @param verbose:
        """
        relevant_modes = (mode,)  # A list with one element
        if mode not in ImagingModes.pure_modes:
            if mode == ImagingModes.All:
                relevant_modes = ImagingModes.pure_modes # all three of them, to be summed
            elif mode == ImagingModes.ContAndSpectral:
                relevant_modes = (ImagingModes.Continuum, ImagingModes.Spectral) # to be summed
            else:
                raise Exception("The '%s' imaging mode is currently not supported" % str(mode))

        result = 0
        for submode in relevant_modes:
            tp = imp.calc_tel_params(telescope, submode, band, hpso, bldta, on_the_fly, max_baseline,
                                                   nr_frequency_channels, verbose)
            Equations.apply_imaging_equations(tp, submode, bldta, on_the_fly, verbose)  # modifies tp in-place

            result_expression = eval('tp.%s' % expression)
            (tsnap_opt, nfacet_opt) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            result += SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap_opt, nfacet_opt, False)

        return result

    @staticmethod
    def eval_param_sweep_1d(telescope, mode, band=None, hpso=None, bldta=True, on_the_fly=False,
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
    def eval_param_sweep_2d(telescope, mode, band=None, hpso=None, bldta=True, on_the_fly=False,
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

        take_max = expression in imp.EXPR_NOT_SUMMED  # Iff true, bins' values are not summed; max taken instead
        n_param_x_values = number_steps + 1
        n_param_y_values = number_steps + 1
        nr_evaluations = n_param_x_values * n_param_y_values  # The number of function evaluations that will be required

        print "Evaluating expression %s while\nsweeping parameters %s and %s over 2D domain [%s, %s] x [%s, %s] in %d " \
              "steps each,\nfor a total of %d data evaluation points" % \
              (expression, parameters[0], parameters[1], str(params_ranges[0][0]), str(params_ranges[0][1]),
               str(params_ranges[1][0]), str(params_ranges[1][1]), number_steps, nr_evaluations)

        telescope_params = imp.calc_tel_params(telescope, mode, band, hpso, bldta, otfk=on_the_fly,
                                               max_baseline=max_baseline, nr_frequency_channels=nr_frequency_channels,
                                               verbose=verbose)

        param_x_values = np.linspace(params_ranges[0][0], params_ranges[0][1], num=n_param_x_values)
        param_y_values = np.linspace(params_ranges[1][0], params_ranges[1][1], num=n_param_y_values)
        results = np.zeros((n_param_x_values, n_param_y_values))  # Create an empty numpy matrix to hold results

        # Nested 2D loop over all values for param1 and param2. Indexes iterate over y (inner loop), then x (outer loop)
        for ix in range(n_param_x_values):
            param_x_value = param_x_values[ix]
            for iy in range(n_param_y_values):
                param_y_value = param_y_values[iy]

                tp = copy.deepcopy(telescope_params)
                # Overwrite the corresponding fields of tp with the to-be-evaluated values
                exec('tp.%s = param_x_value' % parameters[0])
                exec('tp.%s = param_y_value' % parameters[1])

                percentage_done = (ix * n_param_y_values + iy) * 100.0 / nr_evaluations
                print "> %.1f%% done: Evaluating %s for (%s, %s) = (%s, %s)" % (percentage_done, expression,
                                                                                 parameters[0], parameters[1],
                                                                                 str(param_x_value), str(param_y_value))

                Equations.apply_imaging_equations(tp, mode, bldta, on_the_fly, verbose)   # modifies tp in-place

                # Perform a check to see that the value of the assigned parameters weren't changed by the imaging
                # equations, otherwise the assigned values would have been lost (i.e. not free parameters)
                parameter1_final_value = None
                parameter2_final_value = None
                exec('parameter1_final_value = tp.%s' % parameters[0])
                exec('parameter2_final_value = tp.%s' % parameters[1])
                if parameter1_final_value != param_x_value:
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[0])
                if parameter2_final_value != param_y_value:
                    print parameter2_final_value
                    print param_y_value
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[1])

                result_expression = eval('tp.%s' % expression)
                (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                results[iy, ix] = SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet, take_max)

        print 'done with parameter sweep!'
        return (param_x_values, param_y_values, results)

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
