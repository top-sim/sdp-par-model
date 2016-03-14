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

from parameter_definitions import *  # definitions of variables, primary telescope parameters
from parameter_definitions import Constants as c
from equations import *  # formulae that derive secondary telescope-specific parameters from input parameters
from implementation import Implementation as imp  # methods for performing computations (i.e. crunching the numbers)
from implementation import PipelineConfig
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

    # Possible calculated results to display in the notebook
    RESULT_MAP = [
        # Table Row Title              Unit          Default? Sum?   Expression
        ('-- Parameters --',           '',           True,    False, lambda tp: ''                    ),
        ('Telescope',                  '',           True,    False, lambda tp: tp.telescope          ),
        ('Band',                       '',           True,    False, lambda tp: str(tp.band) if tp.band is not None else ''),
        ('Mode',                       '',           True,    False, lambda tp: str(tp.imaging_mode)  ),
        ('BL-dependent averaging',     '',           True,    False, lambda tp: tp.bl_dep_time_av     ),
        ('On-the-fly kernels',         '',           True,    False, lambda tp: tp.on_the_fly         ),
        ('Scale predict by facet',     '',           True,    False, lambda tp: tp.scale_predict_by_facet),
        ('Max # of channels',          '',           True,    False, lambda tp: tp.Nf_max             ),
        ('Max Baseline',               'm',          True,    False, lambda tp: tp.Bmax               ),
        ('Observation Time',           's',          False,   False, lambda tp: tp.Tobs,              ),
        ('Snapshot Time',              's',          True,    False, lambda tp: tp.Tsnap,             ),
        ('Facets',                     '',           True,    False, lambda tp: tp.Nfacet,            ),
        ('Stations/antennas',          '',           False,   False, lambda tp: tp.Na,                ),
        ('Max Baseline [per bin]',     'm',          False,   False, lambda tp: tp.Bmax_bins,         ),
        ('Baseline fraction [per bin]','',           False,   False, lambda tp: tp.frac_bins,         ),

        ('-- Image --',                '',           True,    False, lambda tp: ''                    ),
        ('Facet FoV size',             'deg',        False,   False, lambda tp: tp.Theta_fov/c.degree,),
        ('Total FoV size',             'deg',        False,   False, lambda tp: tp.Total_fov/c.degree,),
        ('PSF size',                   'arcs',       False,   False, lambda tp: tp.Theta_beam/c.arcsecond,),
        ('Pixel size',                 'arcs',       False,   False, lambda tp: tp.Theta_pix/c.arcsecond,),
        ('Facet side length',          'pixels',     True,    False, lambda tp: tp.Npix_linear,       ),
        ('Image side length',          'pixels',     True,    False, lambda tp: tp.Npix_linear_total_fov,     ),
        ('Epsilon (approx)',           '',           False,   False, lambda tp: tp.epsilon_f_approx,  ),
        ('Qbw',                        '',           False,   False, lambda tp: tp.Qbw,               ),
        ('Max subband ratio',          '',           False,   False, lambda tp: tp.max_subband_freq_ratio,),
        ('Number subbands',            '',           False,   False, lambda tp: tp.Number_imaging_subbands,),
        ('Station/antenna diameter',   '',           False,   False, lambda tp: tp.Ds,),

        ('-- Channelization --',       '',           False,   False, lambda tp: ''                    ),
        ('Ionospheric timescale',      's',          False,   False, lambda tp: tp.Tion,              ),
        ('Coalesce time pred',         's',          False,   False, lambda tp: tp.Tcoal_predict,     ),
        ('Coalesce time bw',           's',          False,   False, lambda tp: tp.Tcoal_backward,    ),
        ('Combined Samples',           '',           False,   False, lambda tp: tp.combine_time_samples,),
        ('Channels predict, no-smear', '',           False,   False, lambda tp: tp.Nf_no_smear_predict,),
        ('Channels backward, no-smear','',           False,   False, lambda tp: tp.Nf_no_smear_backward,),
        ('No. of freqs predict ifft',  '',           False,   False, lambda tp: tp.Nf_FFT_predict,    ),
        ('No. of freqs predict conv kernels',   '',  False,   False, lambda tp: tp.Nf_gcf_predict,    ),
        ('Channels to de-grid (predict)',       '',  False,   False, lambda tp: tp.Nf_vis_predict,    ),
        ('No. of freqs for b-ward conv kernels', '', False,   False, lambda tp: tp.Nf_gcf_backward,   ),
        ('Channels to grid (backward)',          '', False,   False, lambda tp: tp.Nf_vis_backward,   ),
        ('No. of frequencies backward fft',      '', False,   False, lambda tp: tp.Nf_FFT_backward,   ),
        ('Channels out',               '',           False,   False, lambda tp: tp.Nf_out,            ),
        ('Visibilities pred',          '',           False,   False, lambda tp: tp.Nvis_predict,      ),
        ('Visibilities bw',            '',           False,   False, lambda tp: tp.Nvis_backward,     ),

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
        ('Working (cache) memory',     'TeraBytes',  True,    True,  lambda tp: tp.Mw_cache/c.tera,   ),
        ('Visibility I/O Rate',        'TeraBytes/s',True,    True,  lambda tp: tp.Rio/c.tera,        ),
        ('Inter-Facet I/O Rate',       'TeraBytes/s',True,    True,  lambda tp: tp.Rinterfacet/c.tera,),

        ('-- Compute --',              '',           True,    False, lambda tp: ''                    ),
        ('Total Compute Requirement',  'PetaFLOPS',  True,    True,  lambda tp: tp.Rflop/c.peta,      ),
        ('-> Gridding total',          'PetaFLOPS',  False,   True,  lambda tp: tp.Rflop_grid/c.peta, ),
        ('-> FFT total',               'PetaFLOPS',  False,   True,  lambda tp: tp.Rflop_fft/c.peta,  ),
        ('-> Phase Rotation',          'PetaFLOPS',  False,   True,  lambda tp: tp.Rflop_phrot/c.peta,),
        ('-> Projection',              'PetaFLOPS',  False,   True,  lambda tp: tp.Rflop_proj/c.peta, ),
        ('-> Conv Kernel Calc Total',  'PetaFLOPS',  False,   True,  lambda tp: tp.Rflop_conv/c.peta, ),
        ('-> Gridding Backward',       'PetaFLOPS',  True,    True,  lambda tp: tp.Rgrid_backward/c.peta, ),
        ('-> Gridding Predict',        'PetaFLOPS',  True,    True,  lambda tp: tp.Rgrid_predict/c.peta, ),
        ('-> FFT Backward',            'PetaFLOPS',  True,    True,  lambda tp: tp.Rflop_fft_bw/c.peta,  ),
        ('-> iFFT Predict',            'PetaFLOPS',  True,    True,  lambda tp: tp.Rflop_fft_predict/c.peta,  ),
        ('-> Phase Rotation',          'PetaFLOPS',  True,    True,  lambda tp: tp.Rflop_phrot/c.peta,),
        ('-> ReProjection',            'PetaFLOPS',  True,    True,  lambda tp: tp.Rflop_proj/c.peta, ),
        ('-> Conv Kernel Calc B-ward', 'PetaFLOPS',  True,    True,  lambda tp: tp.Rccf_backward/c.peta, ),
        ('-> Conv Kernel Calc Predict','PetaFLOPS',  True,    True,  lambda tp: tp.Rccf_predict/c.peta, )
    ]

    @staticmethod
    def get_result_sum(resultMap):
        return map(lambda row: row[3], resultMap)
    @staticmethod
    def get_result_expressions(resultMap,tp):
        return map(lambda row: row[4](tp), resultMap)

    # Rows needed for graphs
    GRAPH_ROWS = map(lambda row: row[0], RESULT_MAP[-8:])

    @staticmethod
    def mk_result_map_rows(verbosity = 'Overview'):
        '''Collects result map information for a given row set
        @rows: Row set to show. If None, we will use default rows.
        @return: A tuple of the result map, the sorted list of the row
        names and a list of the row units.
        '''

        if verbosity == 'Overview':
            result_map = filter(lambda row: row[2], SkaIPythonAPI.RESULT_MAP)
        else:
            result_map = SkaIPythonAPI.RESULT_MAP

        return (result_map,
                map(lambda row: row[0], result_map),
                map(lambda row: row[1], result_map))

    @staticmethod
    def defualt_rflop_plotting_colours():
        """
        Defines a default colour order used in plotting Rflop components
        @return:
        """
        return ('green', 'gold', 'yellowgreen', 'lightskyblue', 'lightcoral', 'red', 'lavender', 'chartreuse')

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
            s += '<tr><td>{0}</td><td><font color="blue">{1}</font> {2}</td></tr>\n'.format(
                labels[i], SkaIPythonAPI.format_result(values[i]), units[i])
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
            s += '<tr><td>{0}</td><td><font color="darkcyan">{1}</font></td><td><font color="blue">{2}</font>' \
                 '</td><td>{3}</td></tr>\n'.format(labels[i],
                                                   SkaIPythonAPI.format_result(values_1[i]),
                                                   SkaIPythonAPI.format_result(values_2[i]),
                                                   units[i])
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
    def plot_2D_surface(title, x_values, y_values, z_values, contours = None, xlabel=None, ylabel=None, zlabel=None, nlevels=15):
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
#        assert len(x_values) == len(y_values)

        sizex = len(x_values)
        sizey = len(y_values)
        assert np.shape(z_values)[0] == sizey
        assert np.shape(z_values)[1] == sizex
        xx = np.tile(x_values, (sizey, 1))
        yy = np.transpose(np.tile(y_values, (sizex, 1)))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xx, yy, z_values, nlevels, rstride=1, cstride=1, cmap=colourmap, linewidth=0.2, alpha=0.6,
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
    def plot_stacked_bars(title, labels, value_labels, dictionary_of_value_arrays, colours=None):
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
        for key in value_labels:
            values = np.array(dictionary_of_value_arrays[key])
            if colours is not None:
                plt.bar(indices, values, width, color=colours[index], bottom=bottoms)
            else:
                plt.bar(indices, values, width, bottom=bottoms)
            bottoms += values
            index += 1

        plt.xticks(indices+width/2., labels)
        plt.title(title)
        plt.legend(value_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend(dictionary_of_value_arrays.keys(), loc=1) # loc=2 -> legend upper-left
        pylab.show()

    @staticmethod
    def check_pipeline_config(cfg, pure_modes):
        """
        Check pipeline configuration, displaying a message in the Notebook
        for every problem found. Returns whether the configuration is
        usable at all.
        """
        (okay, messages) = cfg.check(pure_modes=pure_modes)
        for msg in messages:
            display(HTML('<p><font color="red"><b>{0}</b></font></p>'.format(msg)))
        if not okay:
            display(HTML('<p><font color="red">Adjust to recompute.</font></p>'))
        return okay

    @staticmethod
    def compare_telescopes_default(telescope_1, band_1, mode_1,
                                   telescope_2, band_2, mode_2,
                                   tel1_bldta=True, tel2_bldta=True,
                                   tel1_otf=False, tel2_otf=False,
                                   scale_predict_by_facet=False,
                                   verbosity='Overview'):
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
        @param verbosity: amount of output to generate
        @return:
        """

        # Make configurations and check
        cfg_1 = PipelineConfig(telescope=telescope_1, band=band_1,
                               mode=mode_1, bldta=tel1_bldta,
                               on_the_fly=tel1_otf,
                               scale_predict_by_facet=scale_predict_by_facet)
        cfg_2 = PipelineConfig(telescope=telescope_2, band=band_2,
                               mode=mode_2, bldta=tel2_bldta,
                               on_the_fly=tel2_otf,
                               scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.check_pipeline_config(cfg_1, pure_modes=True) or \
           not SkaIPythonAPI.check_pipeline_config(cfg_2, pure_modes=True):
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
        labels = SkaIPythonAPI.GRAPH_ROWS
        colours = SkaIPythonAPI.defualt_rflop_plotting_colours()
        bldta_text = {True: ' (BLDTA)', False: ' (no BLDTA)'}
        otf_text = {True: ' (otf kernels)', False: ''}

        telescope_labels = ('%s\n%s\n%s' % (telescope_1, bldta_text[tel1_bldta], otf_text[tel1_otf]),
                            '%s\n%s\n%s' % (telescope_2, bldta_text[tel2_bldta], otf_text[tel2_otf]))

        values = {
            label: (tels_result_values[0][-len(labels)+i],
                    tels_result_values[1][-len(labels)+i])
            for i, label in enumerate(labels)
        }

        SkaIPythonAPI.plot_stacked_bars('Computational Requirements (PetaFLOPS)', telescope_labels, labels, values, colours)

    @staticmethod
    def evaluate_telescope_manual(telescope, band, mode,
                                  max_baseline="default",
                                  Nf_max="default", Nfacet=-1,
                                  Tsnap=-1, bldta=True,
                                  on_the_fly=False, scale_predict_by_facet=False,
                                  verbosity='Overview'):
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
        @param verbosity:
        @return:
        """

        assert Nfacet > 0
        assert Tsnap > 0

        # Make configuration
        cfg = PipelineConfig(telescope=telescope, mode=mode, band=band,
                             max_baseline=max_baseline, Nf_max=Nf_max, bldta=bldta,
                             on_the_fly=on_the_fly,
                             scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.check_pipeline_config(cfg, pure_modes=True): return

        display(HTML('<font color="blue">Computing the result -- this may take several seconds.'
                     '</font>'))

        # Determine which rows to calculate & show
        (result_map, result_titles, result_units) = SkaIPythonAPI.mk_result_map_rows(verbosity)

        # Loop through modes
        result_values = SkaIPythonAPI._compute_results(cfg, verbosity=='Debug', result_map,
                                                       Tsnap=Tsnap, Nfacet=Nfacet)

        # Show table of results
        display(HTML('<font color="blue">Done computing. Results follow:</font>'))
        SkaIPythonAPI.show_table('Computed Values', result_titles, result_values, result_units)

        # Show pie graph of FLOP counts
        labels = SkaIPythonAPI.GRAPH_ROWS
        colours = SkaIPythonAPI.defualt_rflop_plotting_colours()
        values = result_values[-8:]  # the last 8 values
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, labels, values, colours)

    @staticmethod
    def evaluate_hpso_optimized(hpso_key, bldta=True,
                                on_the_fly=False,
                                scale_predict_by_facet=False,
                                verbosity='Overview'):
        """
        Evaluates a High Priority Science Objective by optimizing NFacet and Tsnap to minimize the total FLOP rate
        @param hpso:
        @param bldta:
        @param on_the_fly:
        @param verbosity:
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

        # Make and check pipeline configuration
        cfg = PipelineConfig(hpso=hpso_key,bldta=bldta,on_the_fly=on_the_fly,
                             scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.check_pipeline_config(cfg, pure_modes=True): return

        # Determine which rows to calculate & show
        (result_map, result_titles, result_units) = SkaIPythonAPI.mk_result_map_rows(verbosity)

        # Compute
        result_values = SkaIPythonAPI._compute_results(cfg, verbosity=='Debug', result_map)
        display(HTML('<font color="blue">Done computing. Results follow:</font>'))

        # Show table of results
        SkaIPythonAPI.show_table('Computed Values', result_titles, result_values, result_units)

        # Show pie graph of FLOP counts
        labels = SkaIPythonAPI.GRAPH_ROWS
        colours = SkaIPythonAPI.default_rflop_plotting_colours()
        values = result_values[-8:]  # the last 8 values
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, labels, values, colours)

    @staticmethod
    def evaluate_telescope_optimized(telescope, band, mode, max_baseline="default", Nf_max="default",
                                     bldta=True, on_the_fly=False, scale_predict_by_facet=False, verbosity='Overview'):
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
        @param verbosity:
        @return:
        """

        # Make configuration
        cfg = PipelineConfig(telescope=telescope, mode=mode,
                             band=band, max_baseline=max_baseline,
                             Nf_max=Nf_max, bldta=bldta,
                             on_the_fly=on_the_fly,
                             scale_predict_by_facet=scale_predict_by_facet)
        if not SkaIPythonAPI.check_pipeline_config(cfg, pure_modes=True): return

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
        labels = SkaIPythonAPI.GRAPH_ROWS
        colours = SkaIPythonAPI.defualt_rflop_plotting_colours()
        values = result_values[-8:]  # the last 8 values
        SkaIPythonAPI.plot_pie('FLOP breakdown for %s' % telescope, labels, values, colours)

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

        # Loop through modes to collect result values
        result_value_array = []
        for submode in pipelineConfig.relevant_modes:

            # Calculate the telescope parameters
            pipelineConfig.mode = submode
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
            results_for_submode = api.evaluate_expressions(result_expressions, tp, tsnap_opt, nfacet_opt)
            result_value_array.append(results_for_submode)

        # Now transpose, then sum up results from submodes per row
        result_values = []
        transposed_results = zip(*result_value_array)
        sum_results = SkaIPythonAPI.get_result_sum(result_map)
        for (row_values, sum_it) in zip(transposed_results, sum_results):
            if sum_it:
                result_values.append(sum(row_values))
            elif len(row_values) == 1:
                result_values.append(row_values[0])
            else:
                result_values.append(list(row_values))
        return result_values
