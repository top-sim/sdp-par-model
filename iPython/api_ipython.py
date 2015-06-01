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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    def show_table(title, labels, values, units):
        """
        Plots a table of label-value pairs
        @param title:
        @param labels:
        @param values:
        @param units:
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
    def compare_telescopes_default(Telescope_1, Telescope_2, Band_1, Band_2, Mode_1, Mode_2,
                                   Tel1_BLDTA=False, Tel2_BLDTA=False,
                                   Tel1_OTF_kernels=False, Tel2_OTF_kernels=False, verbose=False):
        """
        Evaluates two telescopes, both operating in a given band and mode, using their default parameters.
        A bit of an ugly bit of code, because it contains both computations and display code. But it does make for
        pretty interactive results.
        E.g.: The two telescopes may have different (default) maximum baselines. Plots the results side by side.
        @param Telescope_1:
        @param Telescope_2:
        @param Band:
        @param Mode:
        @param Tel1_BLDTA: Use baseline dependent time averaging for Telescope1 ?
        @param Tel2_BLDTA: Use baseline dependent time averaging for Telescope2 ?
        @param verbose: print verbose output during execution
        @return:
        """
        telescopes = (Telescope_1, Telescope_2)
        bdtas = (Tel1_BLDTA, Tel2_BLDTA)
        modes = (Mode_1, Mode_2)
        bands = (Band_1, Band_2)
        on_the_fly = (Tel1_OTF_kernels, Tel2_OTF_kernels)
        tels_result_strings = []  # Maps each telescope to its results expressed as text, for display in HTML table
        tels_result_values = []   # Maps each telescope to its numerical results, to be plotted in bar chart

        if not (imp.telescope_and_band_are_compatible(Telescope_1, Band_1) and
                imp.telescope_and_band_are_compatible(Telescope_2, Band_2)):
            msg = 'ERROR: At least one of the Telescopes is incompatible with its selected Band'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
        else:
            # And now the results:
            display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.</font>'))
            tels_params = []  # Maps each telescope to its parameter set (one parameter set for each mode)

            for i in range(len(telescopes)):
                telescope = telescopes[i]
                bldta = bdtas[i]
                mode = modes[i]
                band = bands[i]
                otf = on_the_fly[i]
                tps = {}
                # We make a distinction against the "pure" modes, and summed modes
                if mode in (ImagingModes.Continuum, ImagingModes.SlowTrans, ImagingModes.Spectral):
                    tp = imp.calc_tel_params(telescope, mode, band=band, bldta=bldta, on_the_fly=otf,
                                             verbose=verbose)  # Calculate the telescope parameters

                    (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                    tp.Tsnap_opt = tsnap
                    tp.Nfacet_opt = nfacet
                    tps = {mode : tp}

                elif mode == ImagingModes.All:
                    # This mode consists of the *sum* of the Continuum, SlowTrans and Spectral modes,
                    # of which each has a separate set of telescope parameters

                    for submode in (ImagingModes.Continuum, ImagingModes.SlowTrans, ImagingModes.Spectral):
                        tp = imp.calc_tel_params(telescope, submode, band=band, bldta=bldta, on_the_fly=otf,
                                                 verbose=verbose)  # Calculate the telescope parameters

                        (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                        tp.Tsnap_opt = tsnap
                        tp.Nfacet_opt = nfacet
                        tps[submode] = tp
                else:
                    raise Exception("The imaging mode %s is currently not supported" % str(mode))

                tels_params.append(tps)

            # End for-loop. We have now computed the telescope parameters for each mode
            result_titles = ('Telescope', 'Band', 'Mode', 'Baseline Dependent Time Avg.', 'Max Baseline',
                             'Max # channels', 'Optimal Number of Facets', 'Optimal Snapshot Time',
                             'Visibility Buffer', 'Working (cache) memory', 'Image side length', 'I/O Rate',
                             'Total Compute Requirement',
                             '-> Gridding', '-> FFT', '-> Projection', '-> Convolution', '-> Phase Rotation')
            result_units = ('', '', '', '', 'm', '', '', 'sec.', 'PetaBytes', 'TeraBytes', 'pixels', 'TeraBytes/s',
                            'PetaFLOPS', 'PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS','PetaFLOPS')

            nr_result_expressions = 10  # The number of computed values in the result_values array (see below)

            for i in range(len(telescopes)):
                mode = modes[i]
                band = bands[i]
                bdta = bdtas[i]
                telescope = telescopes[i]
                result_values = np.zeros(nr_result_expressions)

                if mode in (ImagingModes.Continuum, ImagingModes.SlowTrans, ImagingModes.Spectral):
                    tp = tels_params[i][mode]
                    # The result expressions need to be defined here as they depend on tp (read above)
                    result_expressions = (tp.Mbuf_vis/c.peta, tp.Mw_cache/c.tera, tp.Npix_linear, tp.Rio/c.tera,
                                          tp.Rflop/c.peta, tp.Rflop_grid/c.peta, tp.Rflop_fft/c.peta,
                                          tp.Rflop_proj/c.peta, tp.Rflop_conv/c.peta, tp.Rflop_phrot/c.peta)
                    # Start building result string
                    result_value_string = [telescope, band, mode, bdta, '%d' % tp.Bmax,
                                           '%d' % tp.Nf_max, '%d' % tp.Nfacet_opt, '%.3g' % tp.Tsnap_opt]
                    result_values = api.evaluate_expressions(result_expressions, tp, tp.Tsnap_opt, tp.Nfacet_opt)

                elif mode == ImagingModes.All:
                    for submode in (ImagingModes.Continuum, ImagingModes.SlowTrans, ImagingModes.Spectral):
                        tp = tels_params[i][submode]
                        # The result expressions need to be defined here as they depend on tp (read above)
                        result_expressions = (tp.Mbuf_vis/c.peta, tp.Mw_cache/c.tera, tp.Npix_linear, tp.Rio/c.tera,
                                              tp.Rflop/c.peta, tp.Rflop_grid/c.peta, tp.Rflop_fft/c.peta,
                                              tp.Rflop_proj/c.peta, tp.Rflop_conv/c.peta, tp.Rflop_phrot/c.peta)
                        result_value_string = [telescope, band, mode, bdta, '%d' % tp.Bmax,
                                               'n/a', 'n/a', 'n/a']

                        result_values += api.evaluate_expressions(result_expressions, tp, tp.Tsnap_opt, tp.Nfacet_opt)
                else:
                    raise Exception("The imaging mode %s is currently not supported" % str(mode))

                for i in range(nr_result_expressions):
                    if i == 2: # The third expression is Npix_linear, that we want to print as an integer value
                        result_value_string.append('%d' % result_values[i])
                    else:
                        result_value_string.append('%.3g' % result_values[i])
                tels_result_strings.append(result_value_string)
                tels_result_values.append(result_values[-5:])  # the last five values

            display(HTML('<font color="blue">Done computing. Results follow:</font>'))

            SkaIPythonAPI.show_table_compare('Computed Values', result_titles, tels_result_strings[0],
                                          tels_result_strings[1], result_units)

            labels = ('Gridding', 'FFT', 'Projection', 'Convolution', 'Phase rot.')
            bldta_text = {True: ' (BLDTA)', False: ' (no BLDTA)'}
            otf_text = {True: ' (otf kernels)', False: ''}


            telescope_labels = ('%s\n%s\n%s' % (Telescope_1, bldta_text[Tel1_BLDTA], otf_text[Tel1_OTF_kernels]),
                                '%s\n%s\n%s' % (Telescope_2, bldta_text[Tel2_BLDTA], otf_text[Tel2_OTF_kernels]))
            colours = ('yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'green')
            values = {}
            i = -1
            for label in labels:
                i += 1
                values[label] = (tels_result_values[0][i], tels_result_values[1][i])

            SkaIPythonAPI.plot_stacked_bars('Computational Requirements (PetaFLOPS)', telescope_labels, values, colours)

    @staticmethod
    def evaluate_telescope_manual(Telescope, Band, Mode, max_baseline, Nf_max, Nfacet, Tsnap, BL_dep_time_av=False, On_the_fly=0, verbose=False):
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
        param_units = ('m', '', '', '', '', 'sec', '')
        SkaIPythonAPI.show_table('Arguments', param_titles, param_values, param_units)

        if not imp.telescope_and_band_are_compatible(Telescope, Band):
            msg = 'ERROR: Telescope and Band are not compatible'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
        else:
            tp_basic = ParameterContainer()
            ParameterDefinitions.apply_telescope_parameters(tp_basic, Telescope)
            max_allowed_baseline = tp_basic.baseline_bins[-1]
            if max_baseline > max_allowed_baseline:
                msg = 'ERROR: max_baseline exceeds the maximum allowed baseline of %g m for this telescope.' \
                    % max_allowed_baseline
                s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
                display(HTML(s))
            else:
                tp = imp.calc_tel_params(telescope=Telescope, mode=Mode, band=Band, bldta=BL_dep_time_av,
                                         max_baseline=max_baseline, nr_frequency_channels=Nf_max, on_the_fly=On_the_fly,
                                         verbose=verbose)  # Calculate the telescope parameters

                # The result expressions need to be defined here as they depend on tp (updated in the line above)
                result_expressions = (tp.Mbuf_vis/c.peta, tp.Mw_cache/c.tera, tp.Npix_linear, tp.Rio/c.tera,
                                      tp.Rflop/c.peta, tp.Rflop_grid/c.peta, tp.Rflop_fft/c.peta, tp.Rflop_proj/c.peta,
                                      tp.Rflop_conv/c.peta, tp.Rflop_phrot/c.peta)
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

                SkaIPythonAPI.show_table('Computed Values', result_titles, result_value_string, result_units)

    @staticmethod
    def evaluate_telescope_optimized(Telescope, Band, Mode, max_baseline, Nf_max, BL_dep_time_av=False, otf=0, verbose=False):
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
        param_units = ('m', '', '', '', '')
        SkaIPythonAPI.show_table('Arguments', param_titles, param_values, param_units)

        if not imp.telescope_and_band_are_compatible(Telescope, Band):
            msg = 'ERROR: Telescope and Band are not compatible'
            s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
            display(HTML(s))
        else:
            tp_basic = ParameterContainer()
            ParameterDefinitions.apply_telescope_parameters(tp_basic, Telescope)
            max_allowed_baseline = tp_basic.baseline_bins[-1]
            if max_baseline > max_allowed_baseline:
                msg = 'ERROR: max_baseline exceeds the maximum allowed baseline of %g m for this telescope.' \
                    % max_allowed_baseline
                s = '<font color="red"><b>{0}</b>.<br>Adjust to recompute.</font>'.format(msg)
                display(HTML(s))
            else:
                display(HTML('<font color="blue">Computing the result -- this may take several (tens of) seconds.'
                             '</font>'))
                tp = imp.calc_tel_params(telescope=Telescope, mode=Mode, band=Band, bldta=BL_dep_time_av,
                                         max_baseline=max_baseline, nr_frequency_channels=Nf_max, on_the_fly=otf,
                                         verbose=verbose)  # Calculate the telescope parameters
                (Tsnap, Nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)

                # The result expressions need to be defined here as they depend on tp (updated in the line above)
                result_expressions = (tp.Mbuf_vis/c.peta, tp.Mw_cache/c.tera, tp.Npix_linear, tp.Rio/c.tera,
                                      tp.Rflop/c.peta, tp.Rflop_grid/c.peta, tp.Rflop_fft/c.peta, tp.Rflop_proj/c.peta,
                                      tp.Rflop_conv/c.peta, tp.Rflop_phrot/c.peta)
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

                SkaIPythonAPI.show_table('Computed Values', result_titles, result_value_string, result_units)
                labels = ('Gridding', 'FFT', 'Projection', 'Convolution', 'Phase rot.')
                colours = ('yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'green')
                values = result_values[-5:]  # the last five values
                SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % Telescope, labels, values, colours)
