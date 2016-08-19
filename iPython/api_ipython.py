"""
This file contains methods for interacting with the SKA SDP Parametric model using Python from the IPython Notebook
(Jupyter) environment. It extends the methods defined in API.py
The reason the code is implemented here is to keep notebooks themselves free from clutter, and to make using the
notebooks easier.
"""
from __future__ import print_function
from api import SkaPythonAPI as api  # This class' (SkaIPythonAPI's) parent class

from IPython.display import clear_output, display, HTML, FileLink

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings

from parameter_definitions import *  # definitions of variables, primary telescope parameters
from parameter_definitions import Constants as c
from equations import *  # formulae that derive secondary telescope-specific parameters from input parameters
from implementation import Implementation as imp  # methods for performing computations (i.e. crunching the numbers)
from implementation import PipelineConfig
from parameter_definitions import ParameterContainer

import csv

class SkaIPythonAPI(api):
    """
    This class (IPython API) is a subclass of its parent, SKA-API. It offers a set of methods for interacting with the
    SKA SDP Parametric model in the IPython Notebook (Jupyter) environment. The reason the code is implemented here is
    to keep the notebook itself free from clutter, and to make coding easier.
    """
    def __init__(self):
        api.__init__(self)
        pass

    # Possible calculated results to display in the notebook
    RESULT_MAP = [
        # Table Row Title              Unit          Default? Sum?   Expression
        ('-- Parameters --',           '',           True,    False, lambda tp: ''                    ),
        ('Telescope',                  '',           True,    False, lambda tp: tp.telescope          ),
        ('Band',                       '',           True,    False, lambda tp: str(tp.band) if tp.band is not None else ''),
        ('Frequency Min',              'GHz',        False,   False, lambda tp: tp.freq_min/c.giga    ),
        ('Frequency Max',              'GHz',        False,   False, lambda tp: tp.freq_max/c.giga    ),
        ('Pipeline',                   '',           True,    False, lambda tp: str(tp.pipeline)      ),
        ('Baseline coalescing',        '',           True,    False, lambda tp: tp.blcoal             ),
        ('On-the-fly kernels',         '',           True,    False, lambda tp: tp.on_the_fly         ),
        ('Scale predict by facet',     '',           True,    False, lambda tp: tp.scale_predict_by_facet),
        ('Max # of channels',          '',           True,    False, lambda tp: tp.Nf_max             ),
        ('Max Baseline',               'm',          True,    False, lambda tp: tp.Bmax               ),
        ('Dump time',                  's',          False,   False, lambda tp: tp.Tint_used,         ),
        ('Observation Time',           's',          False,   False, lambda tp: tp.Tobs,              ),
        ('Snapshot Time',              's',          True,    False, lambda tp: tp.Tsnap,             ),
        ('Facets',                     '',           True,    False, lambda tp: tp.Nfacet,            ),
        ('Stations/antennas',          '',           False,   False, lambda tp: tp.Na,                ),
        ('Max Baseline [per bin]',     'm',          False,   False, lambda tp: tp.Bmax_bins,         ),
        ('Baseline fraction [per bin]','',           False,   False, lambda tp: tp.frac_bins,         ),

        ('-- Image --',                '',           True,    False, lambda tp: ''                    ),
        ('Facet FoV size',             'deg',        False,   False, lambda tp: tp.Theta_fov/c.degree,),
        ('Total FoV size',             'deg',        False,   False, lambda tp: tp.Theta_fov_total/c.degree,),
        ('PSF size',                   'arcs',       False,   False, lambda tp: tp.Theta_beam/c.arcsecond,),
        ('Pixel size',                 'arcs',       False,   False, lambda tp: tp.Theta_pix/c.arcsecond,),
        ('Facet side length',          'pixels',     True,    False, lambda tp: tp.Npix_linear,       ),
        ('Image side length',          'pixels',     True,    False, lambda tp: tp.Npix_linear_fov_total,),
        ('Epsilon (approx)',           '',           False,   False, lambda tp: tp.epsilon_f_approx,  ),
        ('Qbw',                        '',           False,   False, lambda tp: tp.Qbw,               ),
        ('Max subband ratio',          '',           False,   False, lambda tp: tp.max_subband_freq_ratio,),
        ('Number subbands',            '',           False,   False, lambda tp: tp.Nsubbands,),
        ('Station/antenna diameter',   '',           False,   False, lambda tp: tp.Ds,),

        ('-- Channelization --',       '',           False,   False, lambda tp: ''                    ),
        ('Ionospheric timescale',      's',          False,   False, lambda tp: tp.Tion,              ),
        ('Coalesce time pred',         's',          False,   False, lambda tp: tp.Tcoal_predict,     ),
        ('Coalesce time bw',           's',          False,   False, lambda tp: tp.Tcoal_backward,    ),
        ('Combined Samples',           '',           False,   False, lambda tp: tp.combine_time_samples,),
        ('Channels total, no-smear',   '',           False,   False, lambda tp: tp.Nf_no_smear        ),
        ('Channels predict, no-smear', '',           False,   False, lambda tp: tp.Nf_no_smear_predict,),
        ('Channels backward, no-smear','',           False,   False, lambda tp: tp.Nf_no_smear_backward,),
        ('Frequencies predict ifft',   '',           False,   False, lambda tp: tp.Nf_FFT_predict,    ),
        ('Frequencies predict kernels','',           False,   False, lambda tp: tp.Nf_gcf_predict,    ),
        ('Frequencies predict de-grid','',           False,   False, lambda tp: tp.Nf_vis_predict,    ),
        ('Frequencies total',          '',           False,   False, lambda tp: tp.Nf_vis,            ),
        ('Frequencies backward kernels','',          False,   False, lambda tp: tp.Nf_gcf_backward,   ),
        ('Frequencies backward grid',  '',           False,   False, lambda tp: tp.Nf_vis_backward,   ),
        ('Frequencies backward fft',   '',           False,   False, lambda tp: tp.Nf_FFT_backward,   ),
        ('Channels out',               '',           False,   False, lambda tp: tp.Nf_out,            ),
        ('Visibilities ingest',        '1/s',        False,   False, lambda tp: tp.Rvis_ingest,       ),
        ('Visibilities averaged',      '1/s',        False,   False, lambda tp: tp.Rvis,              ),
        ('Visibilities pred',          '1/s',        False,   False, lambda tp: tp.Rvis_predict,      ),
        ('Visibilities bw',            '1/s',        False,   False, lambda tp: tp.Rvis_backward,     ),

        ('-- Geometry --',             '',           False,   False, lambda tp: ''                    ),
        ('Delta W earth',              'lambda',     False,   False, lambda tp: tp.DeltaW_Earth,      ),
        ('Delta W snapshot',           'lambda',     False,   False, lambda tp: tp.DeltaW_SShot,      ),
        ('Delta W max',                'lambda',     False,   False, lambda tp: tp.DeltaW_max,        ),

        ('-- Kernel Sizes --',         '',           False,   False, lambda tp: ''                    ),
        ('W kernel support pred',      'uv-pixels',  False,   False, lambda tp: tp.Ngw_predict,       ),
        ('AW kernel support pred',     'uv-pixels',  False,   False, lambda tp: tp.Nkernel_AW_predict,),
        ('W kernel support pred, ff',  'pixels',     False,   False, lambda tp: tp.Ncvff_predict,     ),
        ('W kernel support bw',        'uv-pixels',  False,   False, lambda tp: tp.Ngw_backward,      ),
        ('AW kernel support bw',       'uv-pixels',  False,   False, lambda tp: tp.Nkernel_AW_backward,),
        ('W kernel support bw, ff',    'pixels',     False,   False, lambda tp: tp.Ncvff_backward,    ),

        ('-- I/O --',                  '',           True,    False, lambda tp: ''                    ),
        ('Visibility Buffer',          'PetaBytes',  True,    True,  lambda tp: tp.Mbuf_vis/c.peta,   ),
        ('Total buffer ingest rate',   'TeraBytes/s',True,    False, lambda tp: tp.Rvis_ingest*tp.Nbeam*tp.Npp*tp.Mvis/c.tera),
        #('Rosies buffer size',   'PetaBytes',       True,       False, lambda tp: tp.Tobs*tp.buffer_factor*tp.Rvis_ingest*tp.Nbeam*tp.Npp*tp.Mvis/c.peta),

        ('Working (cache) memory',     'TeraBytes',  True,    True,  lambda tp: tp.Mw_cache/c.tera,   ),
        ('-> ',                        'TeraBytes',  True,    True,  lambda tp: tp.get_products('Mwcache', scale=c.tera), ),
        ('Visibility I/O Rate',        'TeraBytes/s',True,    True,  lambda tp: tp.Rio/c.tera,        ),
        ('-> ',                        'TeraBytes/s',True,    True,  lambda tp: tp.get_products('Rio', scale=c.tera), ),
        ('Inter-Facet I/O Rate',       'TeraBytes/s',True,    True,  lambda tp: tp.Rinterfacet/c.tera,),
        ('-> ',                        'TeraBytes/s',True,    True,  lambda tp: tp.get_products('Rinterfacet', scale=c.tera), ),

        ('-- Compute --',              '',           True,    False, lambda tp: ''                    ),
        ('Total Compute Requirement',  'PetaFLOPS',  True,    True,  lambda tp: tp.Rflop/c.peta,      ),
        ('-> ',                        'PetaFLOPS',  True,    True,  lambda tp: tp.get_products('Rflop', scale=c.peta), ),
    ]

    @staticmethod
    def get_result_sum(resultMap):
        """
        Returns the corresponding entries of whether expressions should be summed or concatenated in a list.
        @param resultMap:
        @return:
        """
        return list(map(lambda row: row[3], resultMap))

    @staticmethod
    def get_result_expressions(resultMap,tp):
        """
        Returns the expression that needs to be evaluated
        @param resultMap:
        @param tp:
        @return:
        """
        def expr(row):
            try:
                return row[4](tp)
            except AttributeError:
                return "(undefined)"
        return list(map(expr, resultMap))

    # Rows needed for graphs
    GRAPH_ROWS = list(map(lambda row: row[0], RESULT_MAP[-9:]))

    @staticmethod
    def mk_result_map_rows(verbosity = 'Overview'):
        '''Collects result map information for a given row set
        @rows: Row set to show. If None, we will use default rows.
        @return: A tuple of the result map, the sorted list of the row
        names and a list of the row units.
        '''

        if verbosity == 'Overview':
            result_map = list(filter(lambda row: row[2], SkaIPythonAPI.RESULT_MAP))
        else:
            result_map = SkaIPythonAPI.RESULT_MAP

        return (result_map,
                list(map(lambda row: row[0], result_map)),
                list(map(lambda row: row[1], result_map)))

    @staticmethod
    def default_rflop_plotting_colours(rows):
        """
        Defines a default colour order used in plotting Rflop components
        @return:
        """

        # Stolen from D3's category20
        cat20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                 '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                 '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                 '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
        return (cat20 + cat20)[:len(rows)]

    @staticmethod
    def format_result(value):
        """
        Format a result value for viewing. As we expect that most numbers
        should be in a "nice" range this means we truncate number
        accuracy by default.
        """

        # Floating point values up to 3 digits
        if isinstance(value, float):
            return '%.3g' % value
        # Lists: Apply formating recursively
        if isinstance(value, list):
            s = '['
            for v in value:
                if len(s) > 1: s += ', '
                s += SkaIPythonAPI.format_result(v)
            return s + ']'
        # Otherwise: Trust default formatting
        return '%s' % value

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
            if labels[i].startswith('--'):
                s += '<tr><th colspan="2">{0}</th></tr>'.format(labels[i])
                continue
            def row(label, val):
                return '<tr><td>{0}</td><td><font color="blue">{1}</font> {2}</td></tr>\n'.format(
                         label, SkaIPythonAPI.format_result(val), units[i])
            if not isinstance(values[i], dict):
                s += row(labels[i], values[i])
            else:
                for name in sorted(values[i].keys()):
                    s += row(labels[i] + name, values[i][name])
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
            if labels[i].startswith('--'):
                s += '<tr><th colspan="4">{0}</th></tr>'.format(labels[i])
                continue
            def row(label, val1, val2):
                return '<tr><td>{0}</td><td><font color="darkcyan">{1}</font></td><td><font color="blue">{2}</font>' \
                       '</td><td>{3}</td></tr>\n'.format(
                         label,
                         SkaIPythonAPI.format_result(val1),
                         SkaIPythonAPI.format_result(val2),
                         units[i])
            if not isinstance(values_1[i], dict) and not isinstance(values_2[i], dict):
                s += row(labels[i], values_1[i], values_2[i])
            else:
                for name in sorted(set(values_1[i]).union(values_2[i])):
                    s += row(labels[i] + name, values_1[i].get(name, 0), values_2[i].get(name, 0))

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
            if labels[i].startswith('--'):
                s += '<tr><th colspan="5">{0}</th></tr>'.format(labels[i])
                continue
            s += '<tr><td>{0}</td><td><font color="darkcyan">{1}</font></td><td><font color="blue">{2}</font>' \
                 '</td><td><font color="purple">{3}</font>''</td><td>{4}</td></tr>\n'.format(
                     labels[i],
                     SkaIPythonAPI.format_result(values_1[i]),
                     SkaIPythonAPI.format_result(values_2[i]),
                     SkaIPythonAPI.format_result(values_3[i]),
                     units[i])
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
    def plot_2D_surface(title, x_values, y_values, z_values, contours=None, xlabel=None, ylabel=None, zlabel=None, nlevels=15):
        """
        Plots a series of (x,y) values using a line and data-point visualization.
        @param title: The plot's title
        @param x_values: a 1D numpy array
        @param y_values: a 1D numpy array
        @param z_values: a 2D numpy array, indexed as (x,y)
        @param contours: optional array of values at which contours should be drawn
        @param zlabel:
        @param ylabel:
        @param xlabel:
        @return:
        """
        colourmap = 'coolwarm'  # options include: 'afmhot', 'coolwarm'
        contour_colour = [(1., 0., 0., 1.)]  # red

        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session
#        assert len(x_values) == len(y_values)

        sizex = len(x_values)
        sizey = len(y_values)
        assert np.shape(z_values)[0] == sizey
        assert np.shape(z_values)[1] == sizex
        xx = np.tile(x_values, (sizey, 1))
        yy = np.transpose(np.tile(y_values, (sizex, 1)))

        C = pylab.contourf(xx, yy, z_values, nlevels, alpha=.75, cmap=colourmap)
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
    def plot_3D_surface(title, x_values, y_values, z_values,
                        contours=None, xlabel=None, ylabel=None, zlabel=None, nlevels=15):
        """
        Plots a series of (x,y) values using a line and data-point visualization.
        @param title: The plot's title
        @param x_values: a 1D numpy array
        @param y_values: a 1D numpy array
        @param z_values: a 2D numpy array, indexed as (x,y)
        @param contours: optional array of values at which contours should be drawn
        @param zlabel:
        @param ylabel:
        @param xlabel:
        @return:
        """
        colourmap = cm.coolwarm  # options include: 'afmhot', 'coolwarm'
        contour_colour = [(1., 0., 0., 1.)]  # red

        pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session
#        assert len(x_values) == len(y_values)

        sizex = len(x_values)
        sizey = len(y_values)
        assert np.shape(z_values)[0] == sizey
        assert np.shape(z_values)[1] == sizex
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
        @param colours:
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
        @param filename
        @param colours:
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
    def plot_stacked_bars(title, labels, value_labels, dictionary_of_value_arrays,
                          colours=None, width=0.35,
                          save=None):
        """
        Plots a stacked bar chart, with any number of columns and components per stack (must be equal for all bars)
        @param title:
        @param labels: The label belonging to each bar
        @param dictionary_of_value_arrays: A dictionary that maps each label to an array of values (to be stacked).
        @param colours:
        @return:
        """
        # Do some sanity checks
        number_of_elements = len(dictionary_of_value_arrays)
        if colours is not None:
            assert number_of_elements == len(colours)
        for key in dictionary_of_value_arrays:
            assert len(list(dictionary_of_value_arrays[key])) == len(list(labels))

        #Plot a stacked bar chart
        nr_bars = len(labels)
        indices = np.arange(nr_bars)  # The indices of the bars
        bottoms = {} # The height of each bar, by key

        # Collect bars to generate. We want the first bar to end up at
        # the top, therefore we determine their position starting from
        # the back.
        valueSum = np.zeros(nr_bars)
        for key in reversed(list(value_labels)):
            bottoms[key] = valueSum
            valueSum = valueSum + np.array(dictionary_of_value_arrays[key])
        for index, key in enumerate(value_labels):
            values = np.array(dictionary_of_value_arrays[key])
            if colours is not None:
                plt.bar(indices, values, width, color=colours[index], bottom=bottoms[key])
            else:
                plt.bar(indices, values, width, bottom=bottom[key])
            for x, v, b in zip(indices, values, bottoms[key]):
                if v >= np.amax(np.array(valueSum)) / 40:
                    plt.text(x+width/2, b+v/2, "%.1f%%" % (100 * v / valueSum[x]),
                             horizontalalignment='center', verticalalignment='center')

        plt.xticks(indices+width/2., labels)
        plt.title(title)
        plt.legend(value_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend(dictionary_of_value_arrays.keys(), loc=1) # loc=2 -> legend upper-left

        if not save is None:
            plt.savefig(save, format='pdf', dpi=1200, bbox_inches = 'tight')
        pylab.show()

    @staticmethod
    def check_pipeline_config(cfg, pure_pipelines):
        """
        Check pipeline configuration, displaying a message in the Notebook
        for every problem found. Returns whether the configuration is
        usable at all.
        """
        (okay, messages) = cfg.is_valid(pure_pipelines=pure_pipelines)
        for msg in messages:
            display(HTML('<p><font color="red"><b>{0}</b></font></p>'.format(msg)))
        if not okay:
            display(HTML('<p><font color="red">Adjust to recompute.</font></p>'))
        return okay

    @staticmethod
    def compare_telescopes_default(telescope_1, band_1, pipeline_1,
                                   telescope_2, band_2, pipeline_2,
                                   tel1_blcoal=True, tel2_blcoal=True,
                                   tel1_otf=False, tel2_otf=False,
                                   scale_predict_by_facet=True,
                                   verbosity='Overview'):
        """
        Evaluates two telescopes, both operating in a given band and pipeline, using their default parameters.
        A bit of an ugly bit of code, because it contains both computations and display code. But it does make for
        pretty interactive results. Plots the results side by side.
        @param telescope_1:
        @param telescope_2:
        @param band_1:
        @param band_2:
        @param pipeline_1:
        @param pipeline_2:
        @param tel1_otf: On the fly kernels for telescope 1
        @param tel2_otf: On the fly kernels for telescope 2
        @param tel1_blcoal: Use Baseline dependent coalescing (before gridding) for Telescope1
        @param tel2_blcoal: Use Baseline dependent coalescing (before gridding) for Telescope2
        @param verbosity: amount of output to generate
        @return:
        """

        # Make configurations and check
        cfg_1 = PipelineConfig(telescope=telescope_1, band=band_1,
                               pipeline=pipeline_1, blcoal=tel1_blcoal,
                               on_the_fly=tel1_otf,
                               scale_predict_by_facet=scale_predict_by_facet)
        cfg_2 = PipelineConfig(telescope=telescope_2, band=band_2,
                               pipeline=pipeline_2, blcoal=tel2_blcoal,
                               on_the_fly=tel2_otf,
                               scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.check_pipeline_config(cfg_1, pure_pipelines=True) or \
           not SkaIPythonAPI.check_pipeline_config(cfg_2, pure_pipelines=True):
            return

        # Determine which rows to show
        (result_map, result_titles, result_units) = SkaIPythonAPI.mk_result_map_rows(verbosity)

        # Loop through telescope configurations, collect results
        display(HTML('<font color="blue">Computing the result -- this may take several seconds.</font>'))
        tels_result_values = [
            SkaIPythonAPI._compute_results(cfg_1, verbosity=='Debug', result_map),
            SkaIPythonAPI._compute_results(cfg_2, verbosity=='Debug', result_map),
        ]
        display(HTML('<font color="blue">Done computing.</font>'))

        # Show comparison table
        SkaIPythonAPI.show_table_compare('Computed Values', result_titles, tels_result_values[0],
                                         tels_result_values[1], result_units)

        # Show comparison stacked bars
        products_1 = tels_result_values[0][-1]
        products_2 = tels_result_values[1][-1]
        labels = sorted(set(products_1).union(products_2))
        colours = SkaIPythonAPI.default_rflop_plotting_colours(labels)
        telescope_labels = (cfg_1.describe(), cfg_2.describe())

        values = {
            label: (products_1.get(label,0),products_2.get(label,0))
            for label in labels
        }

        SkaIPythonAPI.plot_stacked_bars('Computational Requirements (PetaFLOPS)', telescope_labels, labels, values,
                                        colours)

    @staticmethod
    def evaluate_telescope_manual(telescope, band, pipeline,
                                  max_baseline="default",
                                  Nf_max="default", Nfacet=-1,
                                  Tsnap=-1, blcoal=True,
                                  on_the_fly=False, scale_predict_by_facet=True,
                                  verbosity='Overview'):
        """
        Evaluates a telescope with manually supplied parameters.
        These manually supplied parameters specifically include NFacet; values that can otherwise automtically be
        optimized to minimize an expression (e.g. using the method evaluate_telescope_optimized)
        @param telescope:
        @param band:
        @param pipeline:
        @param Nfacet:
        @param Tsnap:
        @param max_baseline:
        @param Nf_max:
        @param Nfacet:
        @param Tsnap:
        @param blcoal: Baseline dependent coalescing (before gridding)
        @param on_the_fly:
        @param verbosity:
        @param rows:
        @return:
        """

        assert Nfacet > 0
        assert Tsnap > 0

        # Make configuration
        cfg = PipelineConfig(telescope=telescope, pipeline=pipeline, band=band,
                             max_baseline=max_baseline, Nf_max=Nf_max, blcoal=blcoal,
                             on_the_fly=on_the_fly,
                             scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.check_pipeline_config(cfg, pure_pipelines=True): return

        display(HTML('<font color="blue">Computing the result -- this may take several seconds.'
                     '</font>'))

        # Determine which rows to calculate & show
        (result_map, result_titles, result_units) = SkaIPythonAPI.mk_result_map_rows(verbosity)

        # Loop through pipelines
        result_values = SkaIPythonAPI._compute_results(cfg, verbosity=='Debug', result_map,
                                                       Tsnap=Tsnap, Nfacet=Nfacet)

        # Show table of results
        display(HTML('<font color="blue">Done computing. Results follow:</font>'))
        SkaIPythonAPI.show_table('Computed Values', result_titles, result_values, result_units)

        # Show pie graph of FLOP counts
        values = result_values[-1]  # the last value
        colours = SkaIPythonAPI.default_rflop_plotting_colours(set(values))
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, values.keys(), list(values.values()), colours)

    @staticmethod
    def evaluate_hpso_optimized(hpso_key, blcoal=True,
                                on_the_fly=False,
                                scale_predict_by_facet=True,
                                verbosity='Overview'):
        """
        Evaluates a High Priority Science Objective by optimizing NFacet and Tsnap to minimize the total FLOP rate
        @param hpso:
        @param blcoal: Baseline dependent coalescing (before gridding)
        @param on_the_fly:
        @param verbosity:
        @return:
        """

        tp_default = ParameterContainer()
        ParameterDefinitions.apply_global_parameters(tp_default)
        ParameterDefinitions.apply_hpso_parameters(tp_default, hpso_key)
        telescope = tp_default.telescope
        hpso_pipeline = tp_default.pipeline

        # First we plot a table with all the provided parameters
        param_titles = ('HPSO Number', 'Telescope', 'Pipeline', 'Max Baseline', 'Max # of channels', 'Observation time',
                        'Texp (not used in calc)', 'Tpoint (not used in calc)')
        (hours, minutes, seconds) = imp.seconds_to_hms(tp_default.Tobs)
        Tobs_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
        (hours, minutes, seconds) = imp.seconds_to_hms(tp_default.Texp)
        Texp_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
        (hours, minutes, seconds) = imp.seconds_to_hms(tp_default.Tpoint)
        Tpoint_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
        param_values = (hpso_key, telescope, hpso_pipeline, tp_default.Bmax, tp_default.Nf_max, Tobs_string, Texp_string,
                        Tpoint_string)
        param_units = ('', '', '', 'm', '', '', '', '')
        SkaIPythonAPI.show_table('Parameters', param_titles, param_values, param_units)

        # Make and check pipeline configuration
        cfg = PipelineConfig(hpso=hpso_key, blcoal=blcoal, on_the_fly=on_the_fly,
                             scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.check_pipeline_config(cfg, pure_pipelines=True): return

        # Determine which rows to calculate & show
        (result_map, result_titles, result_units) = SkaIPythonAPI.mk_result_map_rows(verbosity)

        # Compute
        result_values = SkaIPythonAPI._compute_results(cfg, verbosity=='Debug', result_map)
        display(HTML('<font color="blue">Done computing. Results follow:</font>'))

        # Show table of results
        SkaIPythonAPI.show_table('Computed Values', result_titles, result_values, result_units)

        # Show pie graph of FLOP counts
        values = result_values[-1]  # the last value
        colours = SkaIPythonAPI.default_rflop_plotting_colours(set(values))
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, values.keys(), list(values.values()), colours)

    @staticmethod
    def evaluate_telescope_optimized(telescope, band, pipeline, max_baseline="default", Nf_max="default",
                                     blcoal=True, on_the_fly=False, scale_predict_by_facet=True, verbosity='Overview'):
        """
        Evaluates a telescope with manually supplied parameters, but then automatically optimizes NFacet and Tsnap
        to minimize the total FLOP rate for the supplied parameters
        @param telescope:
        @param band:
        @param pipeline:
        @param max_baseline:
        @param Nf_max:
        @param blcoal: Baseline dependent coalescing (before gridding)
        @param on_the_fly:
        @param verbosity:
        @return:
        """

        # Make configuration
        cfg = PipelineConfig(telescope=telescope, pipeline=pipeline,
                             band=band, max_baseline=max_baseline,
                             Nf_max=Nf_max, blcoal=blcoal,
                             on_the_fly=on_the_fly,
                             scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.pipeline_is_valid(cfg, pure_pipelines=True): return

        # Determine rows to show
        (result_map, result_titles, result_units) = SkaIPythonAPI.mk_result_map_rows(verbosity)

        # Compute
        display(HTML('<font color="blue">Computing the result -- this may take several seconds.'
                     '</font>'))
        result_values = SkaIPythonAPI._compute_results(cfg, verbosity=='Debug', result_map)
        display(HTML('<font color="blue">Done computing. Results follow:</font>'))

        # Make table
        SkaIPythonAPI.show_table('Computed Values', result_titles, result_values, result_units)

        # Make pie plot
        values = result_values[-1]  # the last value
        colours = SkaIPythonAPI.default_rflop_plotting_colours(set(values))
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, values.keys(), values.values(), colours)

    @staticmethod
    def _pipeline_configurations(telescopes, bands, pipelines,
                                 blcoal=True, on_the_fly=False, scale_predict_by_facet=True):
        """Make a list of all valid configuration combinations in the list."""

        configs = []
        for telescope in telescopes:
            for band in bands:
                for pipeline in pipelines:
                    cfg = PipelineConfig(telescope=telescope, band=band,
                                         pipeline=pipeline, blcoal=blcoal,
                                         on_the_fly=on_the_fly,
                                         scale_predict_by_facet=scale_predict_by_facet)

                    # Check whether the configuration is valid
                    (okay, msgs) = cfg.is_valid()
                    if okay:
                        configs.append(cfg)

        return configs

    @staticmethod
    def write_csv_pipelines(filename, telescopes, bands, pipelines,
                            blcoal=True, on_the_fly=False, scale_predict_by_facet=True,
                            verbose=False):
        """
        Evaluates all valid configurations of this telescope and dumps the
        result as a CSV file.
        """

        # Make configuration list
        configs = SkaIPythonAPI._pipeline_configurations(telescopes, bands, pipelines,
                                                         blcoal, on_the_fly, scale_predict_by_facet)

        # Calculate
        rows = SkaIPythonAPI.RESULT_MAP # Everything - hardcoded for now
        results = SkaIPythonAPI._batch_compute_results(configs, verbose, rows)

        # Write CSV
        SkaIPythonAPI._write_csv(filename, results, rows)

    @staticmethod
    def write_csv_hpsos(filename, hpsos,
                        blcoal=True, on_the_fly=False, scale_predict_by_facet=True,
                        verbose=False):
        """
        Evaluates all valid configurations of this telescope and dumps the
        result as a CSV file.
        """

        # Make configuration list
        configs = []
        for hpso in hpsos:
            cfg = PipelineConfig(hpso=hpso, blcoal=blcoal, on_the_fly=on_the_fly,
                                 scale_predict_by_facet=scale_predict_by_facet)
            configs.append(cfg)

        # Calculate
        rows = SkaIPythonAPI.RESULT_MAP # Everything - hardcoded for now
        results = SkaIPythonAPI._batch_compute_results(configs, verbose, rows)

        # Write CSV
        SkaIPythonAPI._write_csv(filename, results, rows)

    @staticmethod
    def _compute_results(pipelineConfig, verbose, result_map, Tsnap=None, Nfacet=None):
        """A private method for computing a set of results.

        @param pipelineConfig: Complete pipeline configuration
        @param verbose:
        @param result_map: results to produce
        @param Tsnap: Snapshot time. If None it will get determined by optimisation.
        @param Nfacet: Facet count. If None it will get determined by optimisation.
        @return: result value array
        """

        # Loop through pipeliness to collect result values
        result_value_array = []
        for pipeline in pipelineConfig.relevant_pipelines:

            # Calculate the telescope parameters
            pipelineConfig.pipeline = pipeline
            tp = imp.calc_tel_params(pipelineConfig, verbose=verbose)

            # Optimise Tsnap & Nfacet
            tsnap_opt = Tsnap
            nfacet_opt = Nfacet
            if tsnap_opt is None and nfacet_opt is None:
                (tsnap_opt, nfacet_opt) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            elif tsnap_opt is None or nfacet_opt is None:
                raise Exception("We can only optimise Tsnap and Nfacet together so far!")

            # Evaluate expressions from map
            result_expressions = SkaIPythonAPI.get_result_expressions(result_map, tp)
            results_for_pipeline = api.evaluate_expressions(result_expressions, tp, tsnap_opt, nfacet_opt)
            result_value_array.append(results_for_pipeline)

        # Now transpose, then sum up results from pipelines per row
        result_values = []
        transposed_results = zip(*result_value_array)
        sum_results = SkaIPythonAPI.get_result_sum(result_map)
        for (row_values, sum_it) in zip(transposed_results, sum_results):
            if sum_it:
                try:
                    result_values.append(sum(row_values))
                    continue
                except TypeError:
                    pass
            if len(row_values) == 1:
                result_values.append(row_values[0])
            else:
                result_values.append(list(row_values))
        return result_values

    @staticmethod
    def _batch_compute_results(configs, verbose, result_map):
        """Calculate a whole bunch of pipeline configurations. """

        display(HTML('<font color="blue">Calculating pipelines -- this may take quite a while.</font>'))
        results = []
        for cfg in configs:

            # Check that the configuration is valid, skip if it isn't
            (okay, msgs) = cfg.is_valid()
            if not okay:
                # display(HTML('<p>Skipping %s (%s)</p>' % (cfg.describe(), ", ".join(msgs))))
                continue

            # Compute, add to results
            if verbose:
                display(HTML('<p>Calculating %s...</p>' % cfg.describe()))
            results.append((cfg, SkaIPythonAPI._compute_results(cfg, verbose, result_map)))
        return results

    @staticmethod
    def _write_csv(filename, results, rows):
        """
        Writes pipeline calculation results as a CSV file
        """

        with open(filename, 'w') as csvfile:
            w = csv.writer(csvfile)

            # Output row with configurations
            w.writerow([''] + list(map(lambda r: r[0].describe(), results)))

            # Output actual results
            for i, row in enumerate(rows):

                rowTitle = row[0]
                rowUnit = row[1]
                if rowUnit != '': rowUnit = ' [' + rowUnit + ']'

                # Convert lists to dictionaries
                resultRow = map(lambda r: r[1][i], results)
                resultRow = list(map(lambda r: dict(enumerate(r)) if isinstance(r,list) else r,
                                     resultRow))

                # Dictionary? Expand
                dicts = list(filter(lambda r: isinstance(r, dict), resultRow))
                if len(list(dicts)) > 0:

                    # Collect labels
                    labels = set()
                    for d in dicts:
                        labels = labels.union(d.keys())

                    # Show all of them, properly sorted. Non-dicts
                    # (errors) are simply shoved into the first row.
                    first = True
                    for label in labels:
                        def printRow(r):
                            if isinstance(r, dict):
                                return r.get(label, '')
                            elif first:
                                return r
                            return ''
                        w.writerow([rowTitle + str(label) + rowUnit] + list(map(printRow, resultRow)))
                        first = False

                else:

                    # Simple write out as-is
                    w.writerow([rowTitle + rowUnit] + resultRow)


        display(HTML('<font color="blue">Results written to %s.</font>' % filename))

    @staticmethod
    def _read_csv(filename):
        """
        Reads pipeline calculation results from a CSV file as written by _write_csv.
        """

        display(HTML('<font color="blue">Reading %s...</font>' % filename))
        with open(filename, 'r') as csvfile:
            r = csv.reader(csvfile)
            it = iter(r)

            # First row must be headings (i.e. the configurations)
            headings = next(it)
            results = []

            # Data in the rest of them
            for row in it:
                resultRow = []
                for h, v in zip(headings[1:], row[1:]):
                    resultRow.append((h, v))
                results.append((row[0], resultRow))

            return results

    @staticmethod
    def compare_csv(result_file, ref_file, ignore_modifiers=True, export_html=''):
        """
        Read and compare two CSV files with telescope parameters

        @param result_file: CVS file with telescope parameters
        @param ref_file: CVS file with reference parameters
        @param ignore_modifiers: Ignore modifiers when matching columns (say, [blcoal])
        """

        # Read results and reference. Make lookup dictionary for the latter.
        results = SkaIPythonAPI._read_csv(result_file)
        ref = dict(SkaIPythonAPI._read_csv(ref_file))

        def strip_modifiers(head):
            if ignore_modifiers:
                p = head.find(' [')
                if p != -1: return head[:p]
            return head

        # Headings
        s = '<table><tr><td></td>'
        for head, _ in results[0][1]:
            s += '<th>%s</th>' % head
        s += '</tr>\n'

        # Sum up differences
        diff_total = 0
        total_count = 0

        for name, row in results:
            s += '<tr><td>%s</td>' % name

            # Locate reference results
            refRow = ref.get(name, [])
            refRow = map(lambda h_v: (strip_modifiers(h_v[0]), h_v[1]), refRow)
            refRow = dict(refRow)

            # Loop through values
            for head, val in row:
                head = strip_modifiers(head)

                # Number?
                try:

                    # Non-null value
                    num = float(val)
                    if num == 0: raise ValueError

                    # Try to get reference as number, too
                    ref_num = None
                    if head in refRow:
                        try: ref_num = float(refRow[head])
                        except ValueError: ref_num = None

                    # Determine difference
                    diff = None
                    if not (ref_num is None or ref_num == 0):
                        diff = 100*(num-ref_num)/ref_num
                        # Relative difference - use number as
                        # reference for negative changes, as -50% is
                        # about as bad as +200%.
                        diff_rel = max(diff, 100*(ref_num-num)/num)
                        diff_total += abs(diff_rel)
                        total_count += 1

                    # Output
                    if not diff is None:
                        s += '<td bgcolor="#%2x%2x00">%s (%+d%%)</td>' % (
                            int(min(diff_rel/50*255, 255)),
                            int(255-min(max(0, diff_rel-50)/50*255, 255)),
                            SkaIPythonAPI.format_result(num),
                            diff)
                    else:
                        s += '<td>%s</td>' % SkaIPythonAPI.format_result(num)

                except ValueError:

                    # Get reference as string
                    ref_str = refRow.get(head)

                    # No number, output as is
                    if not ref_str is None:
                        total_count += 1
                        if val == ref_str:
                            if val == '':
                                s += '<td></td>'
                            else:
                                s += '<td bgcolor="#00ff00">%s (same)</td>' % val
                        else:
                            s += '<td bgcolor="#ffff00">%s (!= %s)</td>' % (val, ref_str)
                            diff_total += 100
                    else:
                        s += '<td>%s</td>' % val

            s += '</tr>\n'
        s += '</table>'

        if export_html != '':
            f = open(export_html, 'w')
            print("<!doctype html>", file=f)
            print("<html>", file=f)
            print("  <title>SDP Parametric Model Result Comparison</title>", file=f)
            print("  <body><p>Comparing %s against %s</p>" % (result_file, ref_file), file=f)
            print(s, file=f)
            print("  </body>", file=f)
            print("</html>", file=f)
            f.close()
            display(FileLink(export_html))
        else:
            display(HTML('<h3>Comparison:</h3>'))
            display(HTML(s))
        display(HTML('<font color="blue">Done. %.2f %% average relative difference.</font>' % (diff_total / total_count)))

        return diff_total

    @staticmethod
    def stack_bars_pipelines(title, telescopes, bands, pipelines,
                             blcoal=True, on_the_fly=False, scale_predict_by_facet=True,
                             save=None):
        """
        Evaluates all valid configurations of this telescope and shows
        results as stacked bars.
        """

        # Make configurations
        configs = SkaIPythonAPI._pipeline_configurations(telescopes, bands, pipelines,
                                                         blcoal, on_the_fly, scale_predict_by_facet)

        # Calculate
        rows = [SkaIPythonAPI.RESULT_MAP[-1]] # Products only
        results = SkaIPythonAPI._batch_compute_results(configs, False, rows)

        products = list(map(lambda r: r[1][-1], results))
        labels = sorted(set().union(*list(map(lambda p: p.keys(), products))))
        colours = SkaIPythonAPI.default_rflop_plotting_colours(labels)
        tel_labels = list(map(lambda cfg: cfg.describe().replace(" ", "\n"), configs))
        values = {
            label: list(map(lambda p: p.get(label, 0), products))
            for label in labels
        }

        # Show stacked bar graph
        SkaIPythonAPI.plot_stacked_bars(title, tel_labels, labels, values, colours, width=0.7, save=save)

