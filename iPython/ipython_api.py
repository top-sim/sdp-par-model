__author__ = 'Francois'

from IPython.display import clear_output, display, HTML

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

from parameter_definitions import *  # definitions of variables, primary telescope parameters
from formulae import *  # formulae that derive secondary telescope-specific parameters from input parameters
from implementation import Implementation as imp  # methods for performing computations (i.e. crunching the numbers)
from api import SKAAPI as api  # methods for interfacing with the parametric model

class IPythonAPI:
    """
    This class (IPython API) offer a set of methods for interacting witht he SKA SDP Parametric Model in the
    IPython Notebook (Jupyter) environment. The reason the code is implemented here is to keep the notebook itself
    free from clutter, and to make coding easier
    """
    @staticmethod
    def show_table(header, titles, values, units):
        """
        Plots a table of values
        @param header:
        @param titles:
        @param values:
        @param units:
        @return:
        """
        s = '<h3>%s:</h3><table>\n' % header
        assert len(titles) == len(values)
        assert len(titles) == len(units)
        for i in range(len(titles)):
            s += '<tr><td>{0}</td><td><font color="blue">{1}</font> {2}</td></tr>\n'.format(titles[i], values[i], units[i])
        s += '</table>'
        display(HTML(s))

    @staticmethod
    def show_table_compare(header, titles, values_1, values_2, units):
        """
        Plots a table that compares two sets of values with each other
        @param header:
        @param titles:
        @param values_1:
        @param values_2:
        @param units:
        @return:
        """
        s = '<h4>%s:</h4><table>\n' % header
        assert len(titles) == len(values_1)
        assert len(titles) == len(values_2)
        assert len(titles) == len(units)
        for i in range(len(titles)):
            s += '<tr><td>{0}</td><td><font color="blue">{1}</font></td><td><font color="darkcyan">{2}</font>' \
                 '</td><td>{3}</td></tr>\n'.format(titles[i], values_1[i], values_2[i], units[i])
        s += '</table>'
        display(HTML(s))

    @staticmethod
    def plot_flops_pie(header, titles, values, colours=None):
        """
        Plot FLOPS as a pie chart
        @param header:
        @param rflop_grid:
        @param rflop_fft:
        @param rflop_proj:
        @param rflop_conv:
        @param rflop_phrot:
        @return:
        """
        assert len(titles) == len(values)
        if colours is not None:
            assert len(colours) == len(values)
        nr_slices = len(values)

        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session

        # The slices will be ordered and plotted counter-clockwise.
        explode = np.ones(nr_slices) * 0.05  # The radial offset of the slices

        plt.pie(values, explode=explode, labels=titles, colors=colours,
                autopct='%1.1f%%', shadow=True, startangle=90)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.title('%s\n' % header)

        plt.show()

    @staticmethod
    def plot_flops_stacked(header, bar_titles, dictionary_of_value_arrays, colours=None):
        """
        Plots a stacked bar chart, with any number of columns and components per stack (must be equal for all bars)
        @param header:
        @param bar_titles: The title belonging to each bar
        @param dictionary_of_value_arrays: A dictionary that maps each bar title to an array of values (to be stacked).
        @return:
        """
        # Do some sanity checks
        number_of_elements = len(dictionary_of_value_arrays)
        if colours is not None:
            assert number_of_elements == len(colours)
        for key in dictionary_of_value_arrays:
            assert len(dictionary_of_value_arrays[key]) == len(bar_titles)

        #Plot a stacked bar chart
        width = 0.35
        nr_bars = len(bar_titles)
        indices = np.arange(nr_bars)  # The indices of the bars
        bottoms = np.zeros(nr_bars)   # The height of each bar, i.e. the bottom of the next stacked block

        index = 0
        for key in dictionary_of_value_arrays:
            values = np.array(dictionary_of_value_arrays[key])
            if colours is not None:
                plt.bar(indices, values, width, color=colours[index], bottom=bottoms)
            else:
                plt.bar(indices, values, width, bottom=bottoms)
            bottoms += values
            index += 1

        plt.xticks(indices+width/2., bar_titles)
        plt.title(header)
        plt.legend(dictionary_of_value_arrays.keys(), loc=2) # legend upper-left

    @staticmethod
    def compare_telescopes_default(Telescope_1, Telescope_2, Band, Mode, verbose=False):
        """
        Evaluates two telescopes, both operating in a given band and mode, using their default parameters.
        E.g.: The two telescopes may have different (default) maximum baselines. Plots the results side by side.
        @param Telescope_1:
        @param Telescope_2:
        @param Band:
        @param Mode:
        @return:
        """
        telescopes = (Telescope_1, Telescope_2)

        if not (api.telescope_and_band_are_compatible(Telescope_1, Band) and
                api.telescope_and_band_are_compatible(Telescope_2, Band)):
            msg = 'ERROR: At least one of the Telescopes is incompatible with the selected Band'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
        else:
            # And now the results:
            display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.</font>'))
            tps = {}  # Maps each telescope to its parameter set
            for telescope in telescopes:
                tp = imp.calc_tel_params(telescope, Mode, band=Band, verbose=verbose)  # Calculate the telescope parameters
                imp.update_derived_parameters(tp, Mode)
                (Tsnap, Nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                tp.Tsnap_opt = Tsnap
                tp.Nfacet_opt = Nfacet
                tps[telescope] = tp

            tel_results = {}  # Maps each telescope to its numerical results, expressed as strings
            values_per_telescope = {}
            for telescope in telescopes:
                tp = tps[telescope]

                # The result expressions need to be defined here as they depend on tp (updated in the line above)
                result_expressions = (tp.Mbuf_vis/u.peta, tp.Mw_cache/u.tera, tp.Npix_linear, tp.Rio/u.tera,
                                      tp.Rflop/u.peta, tp.Rflop_grid/u.peta, tp.Rflop_fft/u.peta, tp.Rflop_proj/u.peta,
                                      tp.Rflop_conv/u.peta, tp.Rflop_phrot/u.peta)
                result_titles = ('Telescope', 'Max Baseline', 'Max # channels', 'Optimal Number of Facets',
                                 'Optimal Snapshot Time', 'Visibility Buffer',
                                 'Working (cache) memory', 'Image side length', 'I/O Rate', 'Total Compute Requirement',
                                 '-> Gridding', '-> FFT', '-> Projection', '-> Convolution', '-> Phase Rotation')
                result_units = ('km','','', '', 'sec.', 'PetaBytes', 'TeraBytes', 'pixels', 'TeraBytes/s','PetaFLOPS',
                                'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')

                result_value_string = [telescope]  # Start with the telescope's name
                result_value_string.append('%d' % (tp.Bmax / u.km))
                result_value_string.append('%d' % tp.Nf_max)
                result_value_string.append('%d' % tp.Nfacet_opt)
                result_value_string.append('%.3g' % tp.Tsnap_opt)
                result_values = api.evaluate_expressions(result_expressions, tp, tp.Tsnap_opt, tp.Nfacet_opt)
                for i in range(len(result_values)):
                    expression = result_expressions[i]
                    if expression is not tp.Npix_linear:
                        result_value_string.append('%.3g' % result_values[i])
                    else:
                        result_value_string.append('%d' % result_values[i])

                tel_results[telescope] = result_value_string
                values_per_telescope[telescope] = result_values[-5:]  # the last five values
            display(HTML('<font color="blue">Done computing. Results follow:</font>'))

            IPythonAPI.show_table_compare('Computed Values', result_titles, tel_results[Telescope_1],
                                          tel_results[Telescope_2], result_units)

            labels = ('(de)Gridding', '(i)FFT', '(Re)Projection', 'Convolution', 'Phrot')
            telescope_labels = (Telescope_1, Telescope_2)
            colours = ('yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'green')
            values = {}
            i = -1
            for label in labels:
                i += 1
                values[label] = (values_per_telescope[Telescope_1][i], values_per_telescope[Telescope_2][i])

            IPythonAPI.plot_flops_stacked('Computational Requirements (PetaFLOPS)', telescope_labels, values, colours)

    @staticmethod
    def evaluate_telescope_manual(Telescope, Band, Mode, max_baseline, Nf_max, Nfacet, Tsnap, verbose=False):
        """
        Evaluates a telescope with manually supplied parameters, including NFacet and Tsnap
        @param Telescope:
        @param Band:
        @param Mode:
        @param max_baseline:
        @param Nf_max:
        @param Nfacet:
        @param Tsnap:
        @return:
        """
        # First we plot a table with all the provided parameters
        param_titles = ('Max Baseline', 'Max # of channels', 'Telescope', 'Band', 'Mode', 'Tsnap', 'Nfacet')
        param_values = (max_baseline, Nf_max, Telescope, Band, Mode, Tsnap, Nfacet)
        param_units = ('km', '', '', '', '', 'sec', '')
        IPythonAPI.show_table('Arguments', param_titles, param_values, param_units)

        if not api.telescope_and_band_are_compatible(Telescope, Band):
            msg = 'ERROR: Telescope and Band are not compatible'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
        else:
            # And now the results:
            tp = imp.calc_tel_params(Telescope, Mode, band=Band, verbose=verbose)  # Calculate the telescope parameters
            max_allowed_baseline = tp.baseline_bins[-1] / u.km
            if max_baseline <= max_allowed_baseline:
                tp.Bmax = max_baseline * u.km
                tp.Nf_max = Nf_max
                imp.update_derived_parameters(tp, mode=Mode)
                # The result expressions need to be defined here as they depend on tp (updated in the line above)
                result_expressions = (tp.Mbuf_vis/u.peta, tp.Mw_cache/u.tera, tp.Npix_linear, tp.Rio/u.tera,
                                      tp.Rflop/u.peta, tp.Rflop_grid/u.peta, tp.Rflop_fft/u.peta, tp.Rflop_proj/u.peta,
                                      tp.Rflop_conv/u.peta, tp.Rflop_phrot/u.peta)
                result_titles = ('Visibility Buffer', 'Working (cache) memory', 'Image side length', 'I/O Rate',
                                 'Total Compute Requirement',
                                 '-> Gridding', '-> FFT', '-> Projection', '-> Convolution', '-> Phase Rotation')
                result_units = ('PetaBytes', 'TeraBytes', 'pixels', 'TeraBytes/s','PetaFLOPS',
                                'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')
                result_values = api.evaluate_expressions(result_expressions, tp, Tsnap, Nfacet)
                result_value_string = []
                for i in range(len(result_values)):
                    expression = result_expressions[i]
                    if expression is not tp.Npix_linear:
                        result_value_string.append('%.3g' % result_values[i])
                    else:
                        result_value_string.append('%d' % result_values[i])

                IPythonAPI.show_table('Computed Values', result_titles, result_value_string, result_units)
            else :
                msg = 'ERROR: max_baseline exceeds the maximum allowed baseline of %g km for this telescope.' \
                    % max_allowed_baseline
                s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
                display(HTML(s))

    @staticmethod
    def evaluate_telescope_optimized(Telescope, Band, Mode, max_baseline, Nf_max, verbose=False):
        """
        Evaluates a telescope with manually supplied parameters, but then automatically optimizes NFacet and Tsnap
        to minimize the total FLOP rate for the supplied parameters
        @param Telescope:
        @param Band:
        @param Mode:
        @param max_baseline:
        @param Nf_max:
        @return:
        """
        # First we plot a table with all the provided parameters
        param_titles = ('Max Baseline', 'Max # of channels', 'Telescope', 'Band', 'Mode')
        param_values = (max_baseline, Nf_max, Telescope, Band, Mode)
        param_units = ('km', '', '', '', '')
        IPythonAPI.show_table('Arguments', param_titles, param_values, param_units)

        if not api.telescope_and_band_are_compatible(Telescope, Band):
            msg = 'ERROR: Telescope and Band are not compatible'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
        else:
            # And now the results:
            display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.</font>'))
            tp = imp.calc_tel_params(Telescope, Mode, band=Band, verbose=verbose)  # Calculate the telescope parameters
            max_allowed_baseline = tp.baseline_bins[-1] / u.km
            if max_baseline <= max_allowed_baseline:
                tp.Bmax = max_baseline * u.km
                tp.Nf_max = Nf_max
                imp.update_derived_parameters(tp, Mode)
                (Tsnap, Nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)

                # The result expressions need to be defined here as they depend on tp (updated in the line above)
                result_expressions = (tp.Mbuf_vis/u.peta, tp.Mw_cache/u.tera, tp.Npix_linear, tp.Rio/u.tera,
                                      tp.Rflop/u.peta, tp.Rflop_grid/u.peta, tp.Rflop_fft/u.peta, tp.Rflop_proj/u.peta,
                                      tp.Rflop_conv/u.peta, tp.Rflop_phrot/u.peta)
                result_titles = ('Optimal Number of Facets', 'Optimal Snapshot Time', 'Visibility Buffer', 'Working (cache) memory',
                                 'Image side length', 'I/O Rate', 'Total Compute Requirement',
                                 '-> Gridding', '-> FFT', '-> Projection', '-> Convolution', '-> Phase Rotation')
                result_units = ('', 'sec.', 'PetaBytes', 'TeraBytes', 'pixels', 'TeraBytes/s','PetaFLOPS',
                                'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')

                result_value_string = ['%d' % Nfacet, '%.3g' % Tsnap]
                result_values = api.evaluate_expressions(result_expressions, tp, Tsnap, Nfacet)
                for i in range(len(result_values)):
                    expression = result_expressions[i]
                    if expression is not tp.Npix_linear:
                        result_value_string.append('%.3g' % result_values[i])
                    else:
                        result_value_string.append('%d' % result_values[i])

                IPythonAPI.show_table('Computed Values', result_titles, result_value_string, result_units)
                labels = ('(de)Gridding', '(i)FFT', '(Re)Projection', 'Convolution', 'Phrot')
                colours = ('yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'green')
                values = result_values[-5:]  # the last five values

                IPythonAPI.plot_flops_pie('FLOP breakdown for %s' % Telescope, labels, values, colours)
            else:
                msg = 'ERROR: max_baseline exceeds the maximum allowed baseline of %g km for this telescope.' % max_allowed_baseline
                s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
                display(HTML(s))