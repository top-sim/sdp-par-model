"""
This file contains methods for programmatically interacting with the SKA SDP Parametric Model using Python.
"""
import copy

from parameter_definitions import Telescopes, ImagingModes, Bands, ParameterDefinitions
from formulae import Formulae
from implementation import Implementation as imp
from implementation import ParameterContainer
import sympy.physics.units as u
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
    def eval_exp_param_sweep_1d(telescope, mode, band=None, hpso=None, bldta=False,
                                max_baseline=None, nr_frequency_channels=None, expression='Rflop',
                                parameter='Rccf', param_val_min=10, param_val_max=10, number_steps=1, unit_string=None,
                                verbose=False):
        """
        Evaluates an expression for a range of different parameter values, by varying the parameter linearly in
        a specified range in a number of steps

        :param expression: The expression that needs to be evaluated, written as text (e.g. "Rflop")
        :param telescope_params: ParameterContainer (class) containing the initial set of telescope parameters
        :param parameter: the parameter that will be swept - written as text (e.g. "Bmax")
        :param param_val_min: minimum value for the parameter's value sweep
        :param param_val_max: maximum value for the parameter's value sweep
        :param number_steps: the number of *intervals* that will be used to sweep the parameter from min to max
        :param unit_string: If the swept param has units (e.g. kilometres) this *must* be supplied as text, e.g. "u.km"
                            otherwise, if there are no units, the unit-string must be set to "None"
        :return:
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

            # In some cases the parameter has an SI unit (like meters). We then need to multiply that in
            if unit_string is None:
                exec ('tp.%s = %g' % (parameter, param_value))
            else:
                if verbose:
                    print '\nSetting param tp.%s = %g * %s' % (parameter, param_value, unit_string)
                exec ('tp.%s = %g * %s' % (parameter, param_value, unit_string))

            if verbose:
                param_val_string = eval('tp.%s' % parameter)
                print ">> Evaluating %s for %s = %s | %s" % (expression, parameter, param_val_string, str(param_value))

            Formulae.compute_derived_parameters(tp, mode, bldta, verbose)
            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            result_expression = eval('tp.%s' % expression)
            results.append(SKAAPI.evaluate_expression(result_expression, tp, tsnap, nfacet))

        print 'done with parameter sweep!'
        return (param_values, results)

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
        max_allowed_baseline = tp_basic.baseline_bins[-1] / u.km
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


