"""
This file contains methods for interacting with the SKA SDP Parametric Model using Python from the IPython Notebook
(Jupyter) environment. It extends the methods defined in API.py
The reason the code is implemented here is to keep notebooks themselves free from clutter, and to make using the
notebooks easier.
"""
from api import SkaPythonAPI as api  # This class' (SkaIPythonAPI's) parent class

from IPython.display import clear_output, display, HTML

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings

from parameter_definitions import *  # definitions of variables, primary telescope parameters
from parameter_definitions import Constants as c
from equations import *  # formulae that derive secondary telescope-specific parameters from input parameters
from implementation import Implementation as imp  # methods for performing computations (i.e. crunching the numbers)
from parameter_definitions import ParameterContainer


class SkaIPythonAPI(api):
    """
    This class (IPython API) is a subclass of its parent, SKA-API. It offers a set of methods for interacting with the
    SKA SDP Parametric Model in the IPython Notebook (Jupyter) environment. The reason the code is implemented here is
    to keep the notebook itself free from clutter, and to make coding easier.
    """
    def __init__(self):
        api.__init__(self)
        pass

    @staticmethod
    def defualt_rflop_plotting_colours():
        """
        Defines a default colour order used in plotting Rflop components
        @return:
        """
        return ('green', 'gold', 'yellowgreen', 'lightskyblue', 'lightcoral')

    @staticmethod
    def show_table(title, labels, values, units):
        """
        Plots a table of label-value pairs
        @param title: string
        @param labels: string list / tuple
        @param values: string list / tuple
        @param units: string list / tuple
        @return:
        """
        s = '<h3>%s:</h3><table>\n' % title
        assert len(labels) == len(values)
        assert len(labels) == len(units)
        for i in range(len(labels)):
            s += '<tr><td>{0}</td><td><font color="blue">{1}</font> {2}</td></tr>\n'.format(labels[i], values[i], units[i])
        s += '</table>'
        display(HTML(s))

    @staticmethod
    def show_table_compare(title, labels, values_1, values_2, units):
        """
        Plots a table that for a set of labels, compares each' value with the other
        @param title:
        @param labels:
        @param values_1:
        @param values_2:
        @param units:
        @return:
        """
        s = '<h4>%s:</h4><table>\n' % title
        assert len(labels) == len(values_1)
        assert len(labels) == len(values_2)
        assert len(labels) == len(units)
        for i in range(len(labels)):
            s += '<tr><td>{0}</td><td><font color="darkcyan">{1}</font></td><td><font color="blue">{2}</font>' \
                 '</td><td>{3}</td></tr>\n'.format(labels[i], values_1[i], values_2[i], units[i])
        s += '</table>'
        display(HTML(s))

    @staticmethod
    def show_table_compare3(title, labels, values_1, values_2, values_3, units):
        """
        Plots a table that for a set of 3 values pe label compares each' value with the other
        @param title:
        @param labels:
        @param values_1:
        @param values_2:
        @param values_3:
        @param units:
        @return:
        """
        s = '<h5>%s:</h5><table>\n' % title
        assert len(labels) == len(values_1)
        assert len(labels) == len(values_2)
        assert len(labels) == len(values_3)
        assert len(labels) == len(units)
        for i in range(len(labels)):
            s += '<tr><td>{0}</td><td><font color="darkcyan">{1}</font></td><td><font color="blue">{2}</font>' \
                 '</td><td><font color="purple">{3}</font>''</td><td>{4}</td></tr>\n'.format(
                labels[i], values_1[i], values_2[i], values_3[i], units[i])
        s += '</table>'
        display(HTML(s))

    @staticmethod
    def plot_line_datapoints(title, x_values, y_values, xlabel=None, ylabel=None):
        """
        Plots a series of (x,y) values using a line and data-point visualization.
        @param title:
        @param x_values:
        @param y_values:
        @return:
        """
        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session
        assert len(x_values) == len(y_values)
        plt.plot(x_values, y_values, 'ro', x_values, y_values, 'b')
        plt.title('%s\n' % title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def plot_2D_surface(title, x_values, y_values, z_values, contours = None, xlabel=None, ylabel=None, zlabel=None):
        """
        Plots a series of (x,y) values using a line and data-point visualization.
        @param title: The plot's title
        @param x_values: a 1D numpy array
        @param y_values: a 1D numpy array
        @param z_values: a 2D numpy array, indexed as (x,y)
        @param contours: optional array of values at which contours should be drawn
        @return:
        """
        colourmap = 'coolwarm'  # options include: 'afmhot', 'coolwarm'
        contour_colour = [(1., 0., 0., 1.)]  # red

        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session
        assert len(x_values) == len(y_values)

        sizex = len(x_values)
        sizey = len(y_values)
        assert np.shape(z_values)[0] == sizex
        assert np.shape(z_values)[1] == sizey
        xx = np.tile(x_values, (sizey, 1))
        yy = np.transpose(np.tile(y_values, (sizex, 1)))

        C = pylab.contourf(xx, yy, z_values, 15, alpha=.75, cmap=colourmap)
        pylab.colorbar(shrink=.92)
        if contours is not None:
            C = pylab.contour(xx, yy, z_values, levels = contours, colors=contour_colour,
                              linewidths=[2], linestyles='dashed')
            plt.clabel(C, inline=1, fontsize=10)

        C.ax.set_xlabel(xlabel)
        C.ax.set_ylabel(ylabel)
        C.ax.set_title(title, fontsize=16)
        pylab.show()

    @staticmethod
    def plot_3D_surface(title, x_values, y_values, z_values, contours = None, xlabel=None, ylabel=None, zlabel=None):
        """
        Plots a series of (x,y) values using a line and data-point visualization.
        @param title: The plot's title
        @param x_values: a 1D numpy array
        @param y_values: a 1D numpy array
        @param z_values: a 2D numpy array, indexed as (x,y)
        @param contours: optional array of values at which contours should be drawn
        @return:
        """
        colourmap = cm.coolwarm  # options include: 'afmhot', 'coolwarm'
        contour_colour = [(1., 0., 0., 1.)]  # red

        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session
        assert len(x_values) == len(y_values)

        sizex = len(x_values)
        sizey = len(y_values)
        assert np.shape(z_values)[0] == sizex
        assert np.shape(z_values)[1] == sizey
        xx = np.tile(x_values, (sizey, 1))
        yy = np.transpose(np.tile(y_values, (sizex, 1)))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xx, yy, z_values, rstride=1, cstride=1, cmap=colourmap, linewidth=0.2, alpha=0.6,
                               antialiased=True, shade=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if contours is not None:
            cset = ax.contour(xx, yy, z_values, contours, zdir='z', linewidths = (2.0), colors=contour_colour)
            plt.clabel(cset, inline=1, fontsize=10)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title, fontsize=16)
        plt.show()

    @staticmethod
    def plot_pie(title, labels, values, colours=None):
        """
        Plots a pie chart
        @param title:
        @param labels:
        @param values: a numpy array
        @param colous:
        """
        assert len(labels) == len(values)
        if colours is not None:
            assert len(colours) == len(values)
        nr_slices = len(values)

        # The values need to sum to one, for a pie plot. Let's enforce that.
        values_norm = values / np.linalg.norm(values)

        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session

        # The slices will be ordered and plotted counter-clockwise.
        explode = np.ones(nr_slices) * 0.05  # The radial offset of the slices

        plt.pie(values_norm, explode=explode, labels=labels, colors=colours,
                autopct='%1.1f%%', shadow=True, startangle=90)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.title('%s\n' % title)

        plt.show()

    @staticmethod
    def save_pie(title, labels, values, filename, colours=None):
        """
        Works exactly same way as plot_pie(), but instead of plotting, saves a pie chart to SVG output file.
        Useful for exporting results to documents and such
        @param title:
        @param labels:
        @param values: a numpy array
        @param colous:
        """

        assert len(labels) == len(values)
        if colours is not None:
            assert len(colours) == len(values)
        nr_slices = len(values)

        # The values need to sum to one, for a pie plot. Let's enforce that.
        values_norm = values / np.linalg.norm(values)

        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session

        # The slices will be ordered and plotted counter-clockwise.
        explode = np.ones(nr_slices) * 0.05  # The radial offset of the slices

        plt.pie(values_norm, explode=explode, labels=labels, colors=colours,
                autopct='%1.1f%%', shadow=True, startangle=90)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.title('%s\n' % title)

        plt.savefig(filename, format='svg', dpi=1200)

    @staticmethod
    def plot_stacked_bars(title, labels, dictionary_of_value_arrays, colours=None):
        """
        Plots a stacked bar chart, with any number of columns and components per stack (must be equal for all bars)
        @param title:
        @param labels: The label belonging to each bar
        @param dictionary_of_value_arrays: A dictionary that maps each label to an array of values (to be stacked).
        @return:
        """
        # Do some sanity checks
        number_of_elements = len(dictionary_of_value_arrays)
        if colours is not None:
            assert number_of_elements == len(colours)
        for key in dictionary_of_value_arrays:
            assert len(dictionary_of_value_arrays[key]) == len(labels)

        #Plot a stacked bar chart
        width = 0.35
        nr_bars = len(labels)
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

        plt.xticks(indices+width/2., labels)
        plt.title(title)
        plt.legend(dictionary_of_value_arrays.keys(), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend(dictionary_of_value_arrays.keys(), loc=1) # loc=2 -> legend upper-left

    @staticmethod
    def compare_telescopes_default(telescope_1, telescope_2, band_1, band_2, mode_1, mode_2,
                                   tel1_bldta=True, tel2_bldta=True,
                                   tel1_otf=False, tel2_otf=False, verbose=False):
        """
        Evaluates two telescopes, both operating in a given band and mode, using their default parameters.
        A bit of an ugly bit of code, because it contains both computations and display code. But it does make for
        pretty interactive results. Plots the results side by side.
        @param telescope_1:
        @param telescope_2:
        @param band_1:
        @param band_2:
        @param mode_1:
        @param mode_2:
        @param tel1_otf: On the fly kernels for telescope 1
        @param tel2_otf: On the fly kernels for telescope 2
        @param tel1_bldta: Use baseline dependent time averaging for Telescope1
        @param tel2_bldta: Use baseline dependent time averaging for Telescope2
        @param verbose: print verbose output during execution
        @return:
        """
        telescopes = (telescope_1, telescope_2)
        bldtas = (tel1_bldta, tel2_bldta)
        modes = (mode_1, mode_2)
        bands = (band_1, band_2)
        on_the_flys = (tel1_otf, tel2_otf)
        tels_result_strings = []  # Maps each telescope to its results expressed as text, for display in HTML table
        tels_stackable_result_values = []   # Maps each telescope to its numerical results, to be plotted in bar chart

        if not (imp.telescope_and_band_are_compatible(telescope_1, band_1) and
                imp.telescope_and_band_are_compatible(telescope_2, band_2)):
            msg = 'ERROR: At least one of the Telescopes is incompatible with its selected Band'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
            return

        # And now the results:
        display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.</font>'))
        tels_params = []  # Maps each telescope to its parameter set (one parameter set for each mode)

        result_titles = ('Telescope', 'Band', 'Mode', 'Baseline Dependent Time Avg.', 'Max Baseline',
                         'Max # channels', 'Optimal Number of Facets', 'Optimal Snapshot Time',
                         'Image side length', 'Visibility Buffer (no overheads)', 'Working (cache) memory', 'I/O Rate',
                         'Total Compute Requirement',
                         '-> Gridding', '-> FFT', '-> Projection', '-> Convolution', '-> Phase Rotation')
        result_units = ('', '', '', '', 'm', '', '', 'sec.', 'pixels', 'PetaBytes', 'TeraBytes', 'TeraBytes/s',
                        'PetaFLOPS', 'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')

        for i in range(len(telescopes)):
            tp_default = ParameterContainer()  # temp parameter container to get default values for this telescope
            telescope = telescopes[i]
            ParameterDefinitions.apply_telescope_parameters(tp_default, telescope)

            # Use default values of max_baseline and Nf_max
            max_baseline = tp_default.Bmax
            Nf_max = tp_default.Nf_max

            bldta = bldtas[i]
            mode = modes[i]
            band = bands[i]
            on_the_fly = on_the_flys[i]

            # We now make a distinction between "pure" and composite modes
            relevant_modes = (mode,)  # A list with one element
            if mode not in ImagingModes.pure_modes:
                if mode == ImagingModes.All:
                    relevant_modes = ImagingModes.pure_modes # all three of them, to be summed
                else:
                    raise Exception("The '%s' imaging mode is currently not supported" % str(mode))
            (result_values, result_value_string_end) = SkaIPythonAPI._compute_results(telescope, band, relevant_modes,
                                                                                      bldta, on_the_fly, max_baseline,
                                                                                      Nf_max, verbose)
            result_value_string = [telescope, band, mode, bldta, '%g' % max_baseline , '%d' % Nf_max]
            result_value_string.extend(result_value_string_end)

            tels_result_strings.append(result_value_string)
            tels_stackable_result_values.append(result_values[-5:])  # the last five values

        display(HTML('<font color="blue">Done computing. Results follow:</font>'))

        SkaIPythonAPI.show_table_compare('Computed Values', result_titles, tels_result_strings[0],
                                          tels_result_strings[1], result_units)

        labels = ('Gridding', 'FFT', 'Projection', 'Convolution', 'Phase rot.')
        colours = SkaIPythonAPI.defualt_rflop_plotting_colours()
        bldta_text = {True: ' (BLDTA)', False: ' (no BLDTA)'}
        otf_text = {True: ' (otf kernels)', False: ''}

        telescope_labels = ('%s\n%s\n%s' % (telescope_1, bldta_text[tel1_bldta], otf_text[tel1_otf]),
                            '%s\n%s\n%s' % (telescope_2, bldta_text[tel2_bldta], otf_text[tel2_otf]))
        values = {}
        i = -1
        for label in labels:
            i += 1
            values[label] = (tels_stackable_result_values[0][i], tels_stackable_result_values[1][i])

        SkaIPythonAPI.plot_stacked_bars('Computational Requirements (PetaFLOPS)', telescope_labels, values, colours)

    @staticmethod
    def evaluate_telescope_manual(telescope, band, mode, max_baseline="default", Nf_max="default", Nfacet=-1, Tsnap=-1,
                                  bldta=True, on_the_fly=False, verbose=False):
        """
        Evaluates a telescope with manually supplied parameters.
        These manually supplied parameters specifically include NFacet; values that can otherwise automtically be
        optimized to minimize an expression (e.g. using the method evaluate_telescope_optimized)
        @param telescope:
        @param band:
        @param mode:
        @param Nfacet:
        @param Tsnap:
        @param max_baseline:
        @param Nf_max:
        @param bldta:
        @param on_the_fly:
        @param verbose:
        @return:
        """
        # We calculate a "temporary" parameter container to get default values for this telescope. This container will
        # afterwards be discarded. The actual parameters used in the simulation are contained in "tp" further below
        tp_default = ParameterContainer()
        ParameterDefinitions.apply_telescope_parameters(tp_default, telescope)

        # We allow the baseline and/or Nf_max to be undefined, in which case the default values are used.
        if max_baseline == 'default':
            max_baseline = tp_default.Bmax
        if Nf_max == 'default':
            Nf_max = tp_default.Nf_max

        assert Nfacet > 0
        assert Tsnap > 0

        # First we plot a table with all the provided parameters
        param_titles = ('Max Baseline', 'Max # of channels', 'Telescope', 'Band', 'Mode', 'Tsnap', 'Nfacet')
        param_values = (max_baseline, Nf_max, telescope, band, mode, Tsnap, Nfacet)
        param_units = ('m', '', '', '', '', 'sec', '')
        SkaIPythonAPI.show_table('Arguments', param_titles, param_values, param_units)

        # End for-loop. We have now computed the telescope parameters for each mode
        result_titles = ('Image side length(s)', 'Visibility Buffer (no overheads)', 'Working (cache) memory',
                         'I/O Rate', 'Total Compute Requirement',
                         '-> Gridding', '-> FFT', '-> Phase Rotation', '-> Projection', '-> Convolution')
        result_units = ('pixels', 'PetaBytes', 'TeraBytes', 'TeraBytes/s',
                        'PetaFLOPS', 'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')

        assert len(result_titles) == len(result_units)

        if not imp.telescope_and_band_are_compatible(telescope, band):
            msg = 'ERROR: Telescope and Band are not compatible'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
            return

        max_allowed_baseline = tp_default.baseline_bins[-1]
        if max_baseline > max_allowed_baseline:
            msg = 'ERROR: max_baseline exceeds the maximum allowed baseline of %g m for this telescope.' \
                % max_allowed_baseline
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
            return

        display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.'
                     '</font>'))

        # We now make a distinction between "pure" and composite modes
        relevant_modes = (mode,)  # A list with one element
        if mode not in ImagingModes.pure_modes:
            if mode == ImagingModes.All:
                relevant_modes = ImagingModes.pure_modes  # all three of them, to be summed
            else:
                raise Exception("The '%s' imaging mode is currently not supported" % str(mode))

        result_values = np.zeros(10)  # The number of computed values in the result_expressions array
        result_value_string = ['', ]  # The non-summed value: Npix_linear
        # TODO, change the way that take_maxima are determined, as we do in in eval_param_sweep_1d
        take_maxima = [False] * 10
        take_maxima[0] = True  # The first entry corresponds to Npix_linear, see below
        take_maxima[1] = True  # The second entry corresponds to Mbuf_vis, see below
        for submode in relevant_modes:
            # Calculate the telescope parameters
            tp = imp.calc_tel_params(telescope, submode, band=band, bldta=bldta, otfk=on_the_fly,
                                     max_baseline=max_baseline, nr_frequency_channels=Nf_max,
                                     verbose=verbose)

            result_expressions = (tp.Npix_linear, tp.Mbuf_vis/c.peta, tp.Mw_cache/c.tera, tp.Rio/c.tera,
                                  tp.Rflop/c.peta, tp.Rflop_grid/c.peta, tp.Rflop_fft/c.peta,
                                  tp.Rflop_phrot/c.peta, tp.Rflop_proj/c.peta, tp.Rflop_conv/c.peta)

            results_for_submode = api.evaluate_expressions(result_expressions, tp, Tsnap, Nfacet, take_maxima)
            result_value_string[0] += str('%.d, ') % results_for_submode[0]  # Npix_linear
            result_values[1:] += results_for_submode[1:]  # Sum the rest of the values

        # String formatting of the first result (Npix_linear)
        result_value_string[0] = result_value_string[0][:-2]
        if len(relevant_modes) > 1:
            result_value_string[0] = '(%s)' % result_value_string[0]

        for i in range(len(result_values)):
            if i == 0:
                pass
            else:  # floating point expression
                result_value_string.append('%.3g' % result_values[i])

        display(HTML('<font color="blue">Done computing. Results follow:</font>'))

        SkaIPythonAPI.show_table('Computed Values', result_titles, result_value_string, result_units)
        labels = ('Gridding', 'FFT', 'Phase rot.', 'Projection', 'Convolution')
        colours = SkaIPythonAPI.defualt_rflop_plotting_colours()
        values = result_values[-5:]  # the last five values
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, labels, values, colours)

    @staticmethod
    def evaluate_hpso_optimized(hpso_key, bldta=True, on_the_fly=False, verbose=False):
        """
        Evaluates a High Priority Science Objective by optimizing NFacet and Tsnap to minimize the total FLOP rate
        @param hpso:
        @param bldta:
        @param on_the_fly:
        @param verbose:
        @return:
        """
        tp_default = ParameterContainer()
        ParameterDefinitions.apply_global_parameters(tp_default)
        ParameterDefinitions.apply_hpso_parameters(tp_default, hpso_key)
        telescope = tp_default.telescope
        hpso_mode = tp_default.mode

        # First we plot a table with all the provided parameters
        param_titles = ('HPSO Number', 'Telescope', 'Mode', 'Max Baseline', 'Max # of channels', 'Observation time', 'Texp (not used in calc)', 'Tpoint (not used in calc)')
        (hours, minutes, seconds) = imp.seconds_to_hms(tp_default.Tobs)
        Tobs_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
        (hours, minutes, seconds) = imp.seconds_to_hms(tp_default.Texp)
        Texp_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
        (hours, minutes, seconds) = imp.seconds_to_hms(tp_default.Tpoint)
        Tpoint_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
        param_values = (hpso_key, telescope, hpso_mode, tp_default.Bmax, tp_default.Nf_max, Tobs_string, Texp_string, Tpoint_string)
        param_units = ('', '', '', 'm', '', '', '', '')
        SkaIPythonAPI.show_table('Parameters', param_titles, param_values, param_units)

        modes_expanded = (hpso_mode,)
        if hpso_mode == ImagingModes.ContAndSpectral:
            modes_expanded = (ImagingModes.Continuum, ImagingModes.Spectral)

        tps = {}
        Tsnap_opt = {}
        Nfacet_opt = {}
        substitutions = {}
        expressions = {}
        expression_strings = ('Rflop_conv', 'Rflop_fft', 'Rflop_grid', 'Rflop_proj', 'Rflop_phrot', 'Rflop', 'Mbuf_vis',
                              'Mw_cache', 'Npix_linear', 'Rio')
        nr_key_expr_per_mode = len(expression_strings)
        key_expressions = {}
        for key in expression_strings:
            key_expressions[key] = {}

        display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.</font>'))

        for mode in modes_expanded:
            if verbose:
                print 'Computing HPSO %s for %s mode' % (hpso_key, mode)

            tp = imp.calc_tel_params(telescope, mode, hpso=hpso_key, bldta=bldta, otfk=on_the_fly, verbose=verbose)
            tps[mode] = tp
            (Tsnap_opt[mode], Nfacet_opt[mode]) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            substitutions[mode] = {tp.Tsnap : Tsnap_opt[mode], tp.Nfacet : Nfacet_opt[mode]}
            expressions[mode] = (tp.Rflop_conv, tp.Rflop_fft, tp.Rflop_grid, tp.Rflop_proj, tp.Rflop_phrot, tp.Rflop, tp.Mbuf_vis, tp.Mw_cache, tp.Npix_linear, tp.Rio)

            for index in range(nr_key_expr_per_mode):
                key_expressions[expression_strings[index]][mode] = expressions[mode][index]

        key_results = {}
        for key in expression_strings:
            take_max = (key in imp.EXPR_NOT_SUMMED)
            results = np.array([])
            for mode in modes_expanded:
                tp = tps[mode]
                expression = key_expressions[key][mode]
                expression_subst = expression  # Will stay unchanged if a literal number (see if statement below)
                if imp.is_literal(expression):
                    substitution = substitutions[mode]
                    expression_subst = expression.subs(substitution)
                results = np.append(results, imp.evaluate_binned_expression(expression_subst, tp, take_max=take_max))

            if take_max:
                result = np.max(results)
            else:
                result = np.sum(results)

            key_results[key] = result

        display(HTML('<font color="blue">Done computing. Results follow:</font>'))

        # Next, the computed results:
        # TODO won't this be cleaner when we use _Compute_results ?
        ############################

        result_titles = ('Optimal Number(s) of Facets', 'Optimal Snapshot Time(s)',
                         'Image side length(s)', 'Visibility Buffer (no overheads)', 'Working (cache) memory',
                         'I/O Rate', 'Total Compute Requirement',
                         '-> Gridding', '-> FFT', '-> Phase Rotation', '-> Projection', '-> Convolution')
        result_units = ('', 'sec.', 'pixels', 'PetaBytes', 'TeraBytes', 'TeraBytes/s',
                        'PetaFLOPS', 'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')

        assert len(result_titles) == len(result_units)

        result_value_string = ['', '', '']  # The non-summed values: nfacet_opt, tsnap_opt and Npix_linear
        if len(modes_expanded) == 1:
            result_value_string[0] += '%d' % Nfacet_opt[modes_expanded[0]]
            result_value_string[1] += '%.1f' % Tsnap_opt[modes_expanded[0]]
        else:
            for mode in modes_expanded:
                result_value_string[0] += '%s : %d' % (mode, Nfacet_opt[mode])
                result_value_string[1] += '%s : %.1f' % (mode, Tsnap_opt[mode])

        result_value_string[2] = ('%d' % int(np.ceil(key_results['Npix_linear'])))  # Npix_linear

        result_value_string.append('%.3g' % (key_results['Mbuf_vis'] / 1e15))
        result_value_string.append('%.3g' % (key_results['Mw_cache'] / 1e12))
        result_value_string.append('%.3g' % (key_results['Rio'] / 1e12))
        result_value_string.append('%.3g' % (key_results['Rflop'] / 1e15))
        result_value_string.append('%.3g' % (key_results['Rflop_grid'] / 1e15))
        result_value_string.append('%.3g' % (key_results['Rflop_fft'] / 1e15))
        result_value_string.append('%.3g' % (key_results['Rflop_phrot'] / 1e15))
        result_value_string.append('%.3g' % (key_results['Rflop_proj'] / 1e15))
        result_value_string.append('%.3g' % (key_results['Rflop_conv'] / 1e15))

        SkaIPythonAPI.show_table('Computed Values', result_titles, result_value_string, result_units)

        values = (key_results['Rflop_grid'], key_results['Rflop_fft'], key_results['Rflop_phrot'], key_results['Rflop_proj'], key_results['Rflop_conv'])
        labels = ('Gridding', 'FFT', 'Phase rot.', 'Projection', 'Convolution')
        colours = SkaIPythonAPI.defualt_rflop_plotting_colours()
        SkaIPythonAPI.plot_pie('FLOP breakdown for HPSO %s' % hpso_key, labels, values, colours)

    @staticmethod
    def evaluate_telescope_optimized(telescope, band, mode, max_baseline="default", Nf_max="default",
                                     bldta=True, on_the_fly=False, verbose=False):
        """
        Evaluates a telescope with manually supplied parameters, but then automatically optimizes NFacet and Tsnap
        to minimize the total FLOP rate for the supplied parameters
        @param telescope:
        @param band:
        @param mode:
        @param max_baseline:
        @param Nf_max:
        @param bldta:
        @param on_the_fly:
        @param verbose:
        @return:
        """
        # We calculate a "temporary" parameter container to get default values for this telescope. This container will
        # afterwards be discarded. The actual parameters used in the simulation are contained in "tp" further below
        tp_default = ParameterContainer()
        ParameterDefinitions.apply_telescope_parameters(tp_default, telescope)

        # We allow the baseline and/or Nf_max to be undefined, in which case the default values are used.
        if max_baseline == 'default':
            max_baseline = tp_default.Bmax
        if Nf_max == 'default':
            Nf_max = tp_default.Nf_max

        # First we plot a table with all the provided parameters
        param_titles = ('Max Baseline', 'Max # of channels', 'Telescope', 'Band', 'Mode')
        param_values = (max_baseline, Nf_max, telescope, band, mode)
        param_units = ('m', '', '', '', '')
        SkaIPythonAPI.show_table('Arguments', param_titles, param_values, param_units)

        result_titles = ('Optimal Number(s) of Facets', 'Optimal Snapshot Time(s)',
                         'Image side length(s)', 'Visibility Buffer (no overheads)', 'Working (cache) memory',
                         'I/O Rate', 'Total Compute Requirement',
                         '-> Gridding', '-> FFT', '-> Phase Rotation', '-> Projection', '-> Convolution')
        result_units = ('', 'sec.', 'pixels', 'PetaBytes', 'TeraBytes', 'TeraBytes/s',
                        'PetaFLOPS', 'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')

        assert len(result_titles) == len(result_units)

        if not imp.telescope_and_band_are_compatible(telescope, band):
            msg = 'ERROR: Telescope and Band are not compatible'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
            return

        max_allowed_baseline = tp_default.baseline_bins[-1]
        if max_baseline > max_allowed_baseline:
            msg = 'ERROR: max_baseline exceeds the maximum allowed baseline of %g m for this telescope.' \
                % max_allowed_baseline
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
            return

        display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.'
                     '</font>'))

        # We now make a distinction between "pure" and composite modes
        relevant_modes = (mode,)  # A list with one element
        if mode not in ImagingModes.pure_modes:
            if mode == ImagingModes.All:
                relevant_modes = ImagingModes.pure_modes  # all three of them, to be summed
            else:
                raise Exception("The '%s' imaging mode is currently not supported" % str(mode))
        (result_values, result_value_string) = SkaIPythonAPI._compute_results(telescope, band, relevant_modes,
                                                                              bldta, on_the_fly, max_baseline,
                                                                              Nf_max, verbose)

        display(HTML('<font color="blue">Done computing. Results follow:</font>'))

        SkaIPythonAPI.show_table('Computed Values', result_titles, result_value_string, result_units)
        labels = ('Gridding', 'FFT', 'Phase rot.', 'Projection', 'Convolution')
        colours = SkaIPythonAPI.defualt_rflop_plotting_colours()
        values = result_values[-5:]  # the last five values
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, labels, values, colours)

    @staticmethod
    def compute_results(telescope, band, mode, bldta=True, otfk=False, max_baseline=None, nr_frequency_channels=None,
                        verbose=False):
        """
        A specialized utility for computing results. This is a slightly easier-to-interface-with version of
        the private method _compute_results (below)
        @param telescope:
        @param band:
        @param mode:
        @param bldta:
        @param otfk:
        @param verbose:
        @return: @raise Exception:
        """
        relevant_modes = (mode,)  # A list with one element
        if mode not in ImagingModes.pure_modes:
            if mode == ImagingModes.All:
                relevant_modes = ImagingModes.pure_modes # all three of them, to be summed
            else:
                raise Exception("The '%s' imaging mode is currently not supported" % str(mode))
        (result_values, result_values_strings) \
            = SkaIPythonAPI._compute_results(telescope, band, relevant_modes, bldta, otfk, max_baseline=None,
                                             nr_frequency_channels=None, verbose=verbose)

        result_titles = ['Optimal Number of Facets', 'Optimal Snapshot Time',
                         'Image side length', 'Visibility Buffer (no overheads, PB)', 'Working (cache) memory (TB)',
                         'I/O Rate (TBps)', 'Total Compute Requirement (PetaFLOPS)',
                         'Gridding', 'FFT', 'Projection', 'Convolution', 'Phase Rotation']

        assert len(result_titles) == len(result_values)
        assert len(result_titles) == len(result_values_strings)

        return (result_values, result_values_strings, result_titles)

    @staticmethod
    def _compute_results(telescope, band, relevant_modes, bldta, otfk, max_baseline, nr_frequency_channels, verbose):
        """
        A specialized utility for computing a hard-coded set of results. This is a private method.
        Computes a fixed array of ten numerical results and twelve string results; these two result arrays are
        returned as a tuple, and used for display purposes in graphs. Both are needed, because results of composite
        modes are either summed (such as FLOPS) or concatenanted (such as optimal Tsnap values).
        @param telescope:
        @param band:
        @param bldta:
        @param otfk: on the fly kernels
        @param relevant_modes:
        @param max_baseline: The maximum baseline to use
        @param nr_frequency_channels:
        @param verbose:
        @return: (result_values, result_value_string) arrays of length 12. The first two numerical values always = zero.
        """
        result_values = np.zeros(12)  # The number of computed values in the result_expressions array + tsnap + nfacet
        result_value_string = ['', '', '']  # The non-summed values: nfacet_opt, tsnap_opt and Npix_linear
        # The parameters that care computed as binned expressions are mostly summed across bins. The exception to this
        # rule is Npix_linear, where the maximum is taken instead
        take_maxima = [False] * 10
        #TODO: improved the way that take_maxima is determined. Rather do this automatically based on the expression
        take_maxima[0] = True  # The first entry corresponds to Npix_linear, see below
        take_maxima[1] = True  # The second entry corresponds to Mbuf_vis, see below
        for submode in relevant_modes:
            # Calculate the telescope parameters

            #TODO: temporarily limit max baseline to 75km in fast imaging mode for MID (for plotting)
            #if (telescope == Telescopes.SKA1_Mid) and (submode == ImagingModes.FastImg):
            #warnings.warn('MID is being limited to use only 75km max baseline in Fase Imaging mode - see api_ipython.py line 767!')
            #max_baseline = 75000

            tp = imp.calc_tel_params(telescope, submode, band=band, bldta=bldta, otfk=otfk,
                                     max_baseline=max_baseline, nr_frequency_channels=nr_frequency_channels,
                                     verbose=verbose)

            print 'Max baseline = %d' % tp.Bmax #TODO: remove

            (tsnap_opt, nfacet_opt) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)

            result_expressions = (tp.Npix_linear, tp.Mbuf_vis/c.peta, tp.Mw_cache/c.tera, tp.Rio/c.tera,
                                  tp.Rflop/c.peta, tp.Rflop_grid/c.peta, tp.Rflop_fft/c.peta,
                                  tp.Rflop_phrot/c.peta, tp.Rflop_proj/c.peta, tp.Rflop_conv/c.peta)

            result_value_string[0] += str('%d, ') % nfacet_opt
            result_value_string[1] += str('%.1f, ') % tsnap_opt
            results_for_submode = api.evaluate_expressions(result_expressions, tp, tsnap_opt, nfacet_opt, take_maxima)
            result_value_string[2] += str('%.d, ') % results_for_submode[0]  # Npix_linear
            result_values[3:] += results_for_submode[1:]  # Sum the rest of the values

        # String formatting of the first two results (Tsnap_opt and NFacet_opt)
        result_value_string[0] = result_value_string[0][:-2]
        result_value_string[1] = result_value_string[1][:-2]
        result_value_string[2] = result_value_string[2][:-2]

        composite_result = len(relevant_modes) > 1
        if composite_result:
            result_value_string[0] = '(%s)' % result_value_string[0]
            result_value_string[1] = '(%s)' % result_value_string[1]
            result_value_string[2] = '(%s)' % result_value_string[2]

        for i in range(len(result_values)):
            if i < 3:
                pass
            else:  # floating point expression
                result_value_string.append('%.3g' % result_values[i])

        return (result_values, result_value_string)
