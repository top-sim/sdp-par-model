__author__ = 'Francois'

from parameter_definitions import Telescopes, ImagingModes, Bands
from implementation import Implementation as imp
import sympy.physics.units as u


class SKAAPI:
    """
    This class (SKA API) represents an API by which the SKA Parametric Model can be called programmatically, without
    making use of the iPython Notebook infrastructure.
    """

    def __init__(self):
        pass

    telescopes_pretty_print = {Telescopes.SKA1_Low_old: 'SKA1-Low',
                               Telescopes.SKA1_Mid_old: 'SKA1-Mid (Band 1)',
                               Telescopes.SKA1_Sur_old: 'SKA1-Survey (Band 1)'
                               }

    modes_pretty_print = {ImagingModes.Continuum: 'Continuum',
                          ImagingModes.Spectral: 'Spectral',
                          ImagingModes.SlowTrans: 'SlowTrans'
                          }

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
    def compute_results(telescope, band, mode, max_baseline=None, nr_frequency_channels=None, verbose=False):
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
        SKAAPI.telescope_and_band_are_compatible(telescope, band)

        mode_lookup = {}
        for key in SKAAPI.modes_pretty_print:
            mode_lookup[key] = key

        # And now the results:
        tp = imp.calc_tel_params(telescope=telescope, mode=mode, band=band, verbose=verbose)  # Calculate the telescope parameters
        max_allowed_baseline = tp.baseline_bins[-1] / u.km

        if max_baseline is not None:
            max_baseline = max_allowed_baseline
        else:
            if max_baseline > max_allowed_baseline:
                raise AssertionError('max_baseline exceeds the maximum allowed baseline of %g km for this telescope.'
                                     % max_allowed_baseline)

        tp.Bmax = max_baseline * u.km
        tp.Nf_max = nr_frequency_channels
        imp.update_derived_parameters(tp, mode=mode_lookup[mode])
        (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=True)

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
        assert len(result_variable_strings) == len(result_units)   # sanity check

        # The results are returned as a dictionary
        result_dict = {'Tsnap': tsnap, 'NFacet': nfacet}
        for variable_string in result_variable_strings:
            result_expression = eval('tp.%s' % variable_string)
            result = SKAAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)
            result_dict[variable_string] = result

        return result_dict


