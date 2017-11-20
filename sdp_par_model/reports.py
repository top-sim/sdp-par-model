"""
This file contains methods for generating reports for SKA SDP
parametric model data using especially matplotlib and Jupyter. Having
these functions separate allows us to keep notebooks free of clutter.
"""
from __future__ import print_function

import re
import warnings

import csv
from IPython.display import clear_output, display, HTML, FileLink
from ipywidgets import FloatProgress
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pymp
import sympy

from .parameters import definitions as ParameterDefinitions
from .parameters.definitions import Constants as c
from .parameters.container import ParameterContainer
#from .parameters import equations as ParameterEquations # formulae that derive secondary telescope-specific parameters from input parameters
from . import evaluate as imp  # methods for performing computations (i.e. crunching the numbers)
from .config import PipelineConfig

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
    ('Max Baseline',               'km',         True,    False, lambda tp: tp.Bmax / c.kilo      ),
    ('Dump time',                  's',          False,   False, lambda tp: tp.Tint_used,         ),
    ('Observation Time',           's',          False,   False, lambda tp: tp.Tobs,              ),
    ('Snapshot Time',              's',          True,    False, lambda tp: tp.Tsnap,             ),
    ('Facets',                     '',           True,    False, lambda tp: tp.Nfacet,            ),
    ('w-stacking planes',          '',           True,    False, lambda tp: tp.Nwstack,           ),
    ('w-stacking planes predict',  '',           True,    False, lambda tp: tp.Nwstack_predict,   ),
    ('Stations/antennas',          '',           False,   False, lambda tp: tp.Na,                ),
    ('Max Baseline [per bin]',     'km',         False,   False, lambda tp: [ bin['b'] / c.kilo for  bin in tp.bl_bins ] ),
    ('Baseline fraction [per bin]','%',          False,   False, lambda tp: [ 100*bin['bfrac'] for  bin in tp.bl_bins ]  ),

    ('-- Image --',                '',           True,    False, lambda tp: ''                    ),
    ('Facet FoV size',             'deg',        False,   False, lambda tp: tp.Theta_fov/c.degree,),
    ('Total FoV size',             'deg',        False,   False, lambda tp: tp.Theta_fov_total/c.degree,),
    ('PSF size',                   'arcs',       False,   False, lambda tp: tp.Theta_beam/c.arcsecond,),
    ('Pixel size',                 'arcs',       False,   False, lambda tp: tp.Theta_pix/c.arcsecond,),
    ('Facet side length',          'pixels',     True,    False, lambda tp: tp.Npix_linear,       ),
    ('Image side length',          'pixels',     True,    False, lambda tp: tp.Npix_linear_fov_total,),
    ('Grid side dimension',        'lambda',     True,    False, lambda tp: tp.Lambda_grid,),
    ('Baselines dimension',        'lambda',     True,    False, lambda tp: tp.Lambda_bl,),
    ('Epsilon (approx)',           '',           False,   False, lambda tp: tp.epsilon_f_approx,  ),
    ('Qbw',                        '',           False,   False, lambda tp: tp.Qbw,               ),
    ('Max subband ratio',          '',           False,   False, lambda tp: tp.max_subband_freq_ratio,),
    ('Number subbands',            '',           False,   False, lambda tp: tp.Nsubbands,),
    ('Station/antenna diameter',   '',           False,   False, lambda tp: tp.Ds,),

    ('-- Time Coalescing --',      '',           False,   False, lambda tp: ''                    ),
    ('Ionospheric timescale',      's',          False,   False, lambda tp: tp.Tion,              ),
    ('Coalesce time full',         's',          False,   False, lambda tp: tp.Tcoal_skipper,     ),
    ('Coalesce time pred',         's',          False,   False, lambda tp: tp.Tcoal_predict,     ),
    ('Coalesce time bw',           's',          False,   False, lambda tp: tp.Tcoal_backward,    ),
    ('Combined samples full',      '',           False,   False, lambda tp: tp.combine_time_samples,),
    ('Combined samples facet',     '',           False,   False, lambda tp: tp.combine_time_samples_facet,),
    ('Kernel time pred',           's',          False,   False, lambda tp: tp.Tkernel_predict,   ),
    ('Kernel time backward',       's',          False,   False, lambda tp: tp.Tkernel_backward,  ),
    ('Visibilities kernel pred',   '',           False,   False, lambda tp: tp.Nvis_gcf_predict,  ),
    ('Visibilities kernel bw',     '',           False,   False, lambda tp: tp.Nvis_gcf_backward, ),
    ('Oversampling used pred',     '',           False,   False, lambda tp: tp.Ngcf_used_predict,  ),
    ('Oversampling used bw',       '',           False,   False, lambda tp: tp.Ngcf_used_backward, ),

    ('-- Channelization --',       '',           False,   False, lambda tp: ''                    ),
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

    ('-- Visibility --',           '',           False,   False, lambda tp: ''                    ),
    ('Ingest rate',                'M/s',        False,   False, lambda tp: tp.Rvis_ingest / c.mega,),
    ('Averaged rate',              'M/s',        False,   False, lambda tp: tp.Rvis / c.mega,     ),
    ('Predict rate',               'M/s',        False,   False, lambda tp: tp.Rvis_predict  / c.mega, ),
    ('Backward rate',              'M/s',        False,   False, lambda tp: tp.Rvis_backward  / c.mega,),

    ('-- Kernel Sizes --',         '',           False,   False, lambda tp: ''                    ),
    ('Delta W earth',              'lambda',     False,   False, lambda tp: tp.DeltaW_Earth,      ),
    ('Delta W snapshot',           'lambda',     False,   False, lambda tp: tp.DeltaW_SShot,      ),
    ('Delta W max',                'lambda',     False,   False, lambda tp: tp.DeltaW_max,        ),
    ('Delta W stack',              'lambda',     False,   False, lambda tp: tp.DeltaW_stack,),
    ('Delta W theory',             'lambda',     False,   False, lambda tp: 1/(1-max(0,1-2*tp.Theta_fov**2)**0.5),),

    ('-- Kernel Sizes --',         '',           False,   False, lambda tp: ''                    ),
    ('W kernel support pred',      'uv-pixels',  False,   False, lambda tp: tp.Ngw_predict,       ),
    ('AW kernel support pred',     'uv-pixels',  False,   False, lambda tp: tp.Nkernel_AW_predict,),
    ('W kernel support bw',        'uv-pixels',  False,   False, lambda tp: tp.Ngw_backward,      ),
    ('AW kernel support bw',       'uv-pixels',  False,   False, lambda tp: tp.Nkernel_AW_backward,),

    ('-- Data --',                 '',           True,    False, lambda tp: ''                    ),
    ('Snapshot vis size',          'GB',         True,    True,  lambda tp: tp.Mvis * tp.Rvis * tp.Npp * tp.Tsnap / tp.Nf_FFT_backward / c.giga ),
    ('Facet vis size predict',     'GB',         True,    True,  lambda tp: tp.Mvis * tp.Rvis_predict * tp.Npp * tp.Tsnap / tp.Nf_FFT_predict / c.giga ),
    ('Facet size',                 'GB',         True,    False, lambda tp: tp.Mpx * tp.Npix_linear**2 / c.giga ),
    ('Facet size (all pol)',       'GB',         True,    False, lambda tp: tp.Mpx * tp.Npp * tp.Npix_linear**2 / c.giga ),
    ('Image size',                 'GB',         True,    False, lambda tp: tp.Mpx * tp.Npix_linear_fov_total**2 / c.giga ),
    ('Image size (all pol)',       'GB',         True,    False, lambda tp: tp.Mpx * tp.Npp * tp.Npix_linear_fov_total**2 / c.giga ),
    ('Image cube size',            'GB',         True,    False, lambda tp: tp.Nf_out * tp.Npp * tp.Mpx * tp.Npix_linear_fov_total**2 / c.giga ),

    ('Calibration input',          'MB',         True,    False, lambda tp: tp.Mcal_in / c.mega ),
    ('Calibration output',         'MB',         True,    False, lambda tp: tp.Mcal_out / c.mega ),
    ('Calibration process interval','s',         True,    False, lambda tp: tp.Tsolve ),

    ('Cleaning memory',            'PetaBytes',  True,    True,  lambda tp: tp.M_MSMFS/c.peta,   ),
    ('Working (cache) memory',     'TeraBytes',  True,    True,  lambda tp: tp.Mw_cache/c.tera,   ),
    ('-- I/O --',                  '',           True,    False, lambda tp: ''                    ),
    ('Visibility Buffer',          'PetaBytes',  True,    True,  lambda tp: tp.Mbuf_vis/c.peta,   ),
    ('Total buffer ingest rate',   'TeraBytes/s',True,    False, lambda tp: tp.Rvis_ingest*tp.Nbeam*tp.Npp*tp.Mvis/c.tera),
    #('Rosies buffer size',   'PetaBytes',       True,       False, lambda tp: tp.Tobs*tp.buffer_factor*tp.Rvis_ingest*tp.Nbeam*tp.Npp*tp.Mvis/c.peta),

    ('-> ',                        'TeraBytes',  True,    True,  lambda tp: tp.get_products('Mwcache', scale=c.tera), ),
    ('Visibility I/O Rate',        'TeraBytes/s',True,    True,  lambda tp: tp.Rio/c.tera,        ),
    ('Facet visibility rate',      'TeraBytes/s',True,    False, lambda tp: tp.Rfacet_vis/c.tera),
    ('Image Write Rate',           'TeraBytes/s',True,    True,  lambda tp: tp.Rimage/c.tera,        ),

    #('Inter-Facet I/O Rate',       'TeraBytes/s',True,    True,  lambda tp: tp.Rinterfacet/c.tera,),
    #('-> ',                        'TeraBytes/s',True,    True,  lambda tp: tp.get_products('Rinterfacet', scale=c.tera), ),

    ('-- Compute --',              '',           True,    False, lambda tp: ''                    ),
    ('Total Compute Requirement',  'PetaFLOP/s', True,    True,  lambda tp: tp.Rflop/c.peta,      ),
    ('-> ',                        'PetaFLOP/s', True,    True,  lambda tp: tp.get_products('Rflop', scale=c.peta), ),
]


def get_result_sum(resultMap):
    """
    Returns the corresponding entries of whether expressions should be summed or concatenated in a list.

    :param resultMap:
    :returns:
    """
    return list(map(lambda row: row[3], resultMap))


def get_result_expressions(resultMap,tp):
    """
    Returns the expression that needs to be evaluated

    :param resultMap:
    :param tp:
    :returns:
    """
    def expr(row):
        try:
            return row[4](tp)
        except AttributeError:
            return "(undefined)"
    return list(map(expr, resultMap))

def mk_result_map_rows(verbosity = 'Overview'):
    """
    Collects result map information for a given row set

    :param verbosity: Row set to show. If None, we will use default rows.
    :returns: A tuple of the result map, the sorted list of the row
       names and a list of the row units.
    """

    if verbosity == 'Overview':
        result_map = list(filter(lambda row: row[2], RESULT_MAP))
    else:
        result_map = RESULT_MAP

    return (result_map,
            list(map(lambda row: row[0], result_map)),
            list(map(lambda row: row[1], result_map)))


def default_rflop_plotting_colours(rows):
    """
    Defines a default colour order used in plotting Rflop components

    :returns: List of HTML colour codes as string
    """

    # Stolen from D3's category20
    cat20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
             '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
             '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
             '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    return (cat20 + cat20)[:len(rows)]


def format_result(value):
    """
    Format a result value for viewing. As we expect that most numbers
    should be in a "nice" range this means we truncate number
    accuracy by default.
    """

    # Attempt to make proper floats out of sympy expressions
    if isinstance(value, sympy.Expr):
        try:
            value = float(value.evalf())
        except e:
            pass
    # Floating point values up to 4 digits
    if isinstance(value, float):
        return min(['%.4g' % value, '%.5g' % value], key=len)
    # Lists: Apply formating recursively
    if isinstance(value, list):
        s = '['
        for v in value:
            if len(s) > 1: s += ', '
            s += format_result(v)
        return s + ']'
    # Otherwise: Trust default formatting
    return '%s' % value


def format_result_cell(val, color='black', colspan=1, typ='td'):
    """
    Format a result value for inclusing in a table.
    """
    return '<%s colspan="%d" style="text-align:left"><font color="%s">%s</font></%s>' % (
        typ, colspan, color, format_result(val), typ)


def format_result_cells(val, color='black', max_cols=1):
    """
    Format a result value for inclusing in a table. If the value is a
    list, we genrate multiple cells up to "max_cells".
    """

    row_html = ''
    if type(val) == list and len(val) <= max_cols:
        for v in val:
            row_html += format_result_cell(v, color, 1)
        if len(val) < max_cols:
            row_html += format_result_cell('', color, max_cols-len(val))
    else:
        row_html += format_result_cell(val, color, max_cols)
    return row_html


def show_table(title, labels, values, units, docs=None):
    """
    Plots a table of label-value pairs

    :param title: string
    :param labels: string list / tuple
    :param values: string list / tuple
    :param units: string list / tuple
    :param docs: Optional documentation per row
    :returns:
    """
    s = '<h3>%s:</h3><table>\n' % title
    assert len(labels) == len(values)
    assert len(labels) == len(units)
    max_cols = max([1] + [len(v) for v in values if type(v) == list])
    extra_cols = (2 if docs is None else 3)
    for i in range(len(labels)):
        if labels[i].startswith('--'):
            s += '<tr>%s</tr>' % format_result_cell(labels[i], colspan=max_cols+extra_cols, typ='th')
            continue
        def row(label, val):
            row_html = '<tr><td>%s</td>' % label
            row_html += format_result_cells(val, color='blue', max_cols=max_cols)
            row_html += '<td style="text-align:left">%s</td>' % units[i]
            if docs is not None:
                row_html += '<td style="text-align:left">%s</td>' % docs[i]
            return row_html + '</tr>\n'
        if not isinstance(values[i], dict):
            s += row(labels[i], values[i])
        else:
            for name in sorted(values[i].keys()):
                s += row(labels[i] + name, values[i][name])
    s += '</table>'
    display(HTML(s))


def show_table_compare(title, labels, values_1, values_2, units):
    """
    Plots a table that for a set of labels, compares each' value with the other

    :param title:
    :param labels:
    :param values_1:
    :param values_2:
    :param units:
    :returns:
    """
    s = '<h4>%s:</h4><table>\n' % title
    assert len(labels) == len(values_1)
    assert len(labels) == len(values_2)
    assert len(labels) == len(units)

    max_cols = max([1] + [len(v) for v in values_1+values_2 if type(v) == list])

    for i in range(len(labels)):
        if labels[i].startswith('--'):
            s += '<tr>%s</tr>' % format_result_cell(labels[i], colspan=2*max_cols+2, typ='th')
            continue
        def row(label, val1, val2):
            row_html = '<tr><td>%s</td>' % label
            row_html += format_result_cells(val1, color='darkcyan', max_cols=max_cols)
            row_html += format_result_cells(val2, color='blue', max_cols=max_cols)
            row_html += '<td style="text-align:left">%s</td></tr>\n' % units[i]
            return row_html
        if not isinstance(values_1[i], dict) and not isinstance(values_2[i], dict):
            s += row(labels[i], values_1[i], values_2[i])
        else:
            for name in sorted(set(values_1[i]).union(values_2[i])):
                s += row(labels[i] + name, values_1[i].get(name, 0), values_2[i].get(name, 0))

    s += '</table>'
    display(HTML(s))


def show_table_compare3(title, labels, values_1, values_2, values_3, units):
    """
    Plots a table that for a set of 3 values pe label compares each' value with the other

    :param title:
    :param labels:
    :param values_1:
    :param values_2:
    :param values_3:
    :param units:
    :returns:
    """
    s = '<h5>%s:</h5><table>\n' % title
    assert len(labels) == len(values_1)
    assert len(labels) == len(values_2)
    assert len(labels) == len(values_3)
    assert len(labels) == len(units)
    for i in range(len(labels)):
        if labels[i].startswith('--'):
            s += '<tr><th colspan="5" style="text-align:left">{0}</th></tr>'.format(labels[i])
            continue
        s += '<tr><td>{0}</td><td><font color="darkcyan">{1}</font></td><td><font color="blue">{2}</font>' \
             '</td><td><font color="purple">{3}</font>''</td><td>{4}</td></tr>\n'.format(
                 labels[i],
                 format_result(values_1[i]),
                 format_result(values_2[i]),
                 format_result(values_3[i]),
                 units[i])
    s += '</table>'
    display(HTML(s))


def plot_line_datapoints(title, x_values, y_values, xlabel=None, ylabel=None):
    """
    Plots a series of (x,y) values using a line and data-point visualization.

    :param title:
    :param x_values:
    :param y_values:
    :returns:
    """
    pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session
    assert len(x_values) == len(y_values)
    plt.plot(x_values, y_values, 'ro', x_values, y_values, 'b')
    plt.title('%s\n' % title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim((0, max(y_values)))
    plt.show()


def plot_2D_surface(title, x_values, y_values, z_values, contours=None, xlabel=None, ylabel=None, zlabel=None, nlevels=15):
    """
    Plots a series of (x,y) values using a line and data-point visualization.

    :param title: The plot's title
    :param x_values: a 1D numpy array
    :param y_values: a 1D numpy array
    :param z_values: a 2D numpy array, indexed as (x,y)
    :param contours: optional array of values at which contours should be drawn
    :param zlabel:
    :param ylabel:
    :param xlabel:
    :returns:
    """
    colourmap = 'coolwarm'  # options include: 'afmhot', 'coolwarm'
    contour_colour = [(1., 0., 0., 1.)]  # red

    pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session

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


def plot_3D_surface(title, x_values, y_values, z_values,
                    contours=None, xlabel=None, ylabel=None, zlabel=None, nlevels=15):
    """
    Plots a series of (x,y) values using a line and data-point visualization.

    :param title: The plot's title
    :param x_values: a 1D numpy array
    :param y_values: a 1D numpy array
    :param z_values: a 2D numpy array, indexed as (x,y)
    :param contours: optional array of values at which contours should be drawn
    :param zlabel:
    :param ylabel:
    :param xlabel:
    :returns:
    """
    colourmap = cm.coolwarm  # options include: 'afmhot', 'coolwarm'
    contour_colour = [(1., 0., 0., 1.)]  # red

    pylab.rcParams['figure.figsize'] = 8, 6  # that's default image size for this interactive session
    assert len(x_values) == len(y_values)

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


def plot_pie(title, labels, values, colours=None):
    """
    Plots a pie chart

    :param title:
    :param labels:
    :param values: a numpy array
    :param colours:
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


def save_pie(title, labels, values, filename, colours=None):
    """
    Works exactly same way as plot_pie(), but instead of plotting,
    saves a pie chart to SVG output file.  Useful for exporting
    results to documents and such

    :param title:
    :param labels:
    :param values: a numpy array
    :param filename:
    :param colours:
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


def plot_stacked_bars(title, labels, value_labels, dictionary_of_value_arrays,
                      colours=None, width=0.35,
                      save=None, xticks_rot='horizontal'):
    """
    Plots a stacked bar chart, with any number of columns and
    components per stack (must be equal for all bars)

    :param title:
    :param labels: The label belonging to each bar
    :param dictionary_of_value_arrays: A dictionary that maps each label to an array of values (to be stacked).
    :param colours:
    :returns:
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

    plt.xticks(indices+width/2., labels, rotation=xticks_rot)
    plt.title(title)
    plt.legend(value_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend(dictionary_of_value_arrays.keys(), loc=1) # loc=2 -> legend upper-left

    if not save is None:
        plt.savefig(save, format='pdf', dpi=1200, bbox_inches = 'tight')
    pylab.show()


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


def compare_telescopes_default(telescope_1, band_1, pipeline_1, adjusts_1,
                               telescope_2, band_2, pipeline_2, adjusts_2,
                               verbosity='Overview'):
    """
    Evaluates two telescopes, both operating in a given band and
    pipeline, using their default parameters.  A bit of an ugly bit of
    code, because it contains both computations and display code. But
    it does make for pretty interactive results. Plots the results
    side by side.

    :param telescope_1:
    :param telescope_2:
    :param band_1:
    :param band_2:
    :param pipeline_1:
    :param pipeline_2:
    :param adjusts_1: Configuration adjustments for telescope 1. See PipelineConfig.
    :param adjusts_2: Configuration adjustments for telescope 2. See PipelineConfig.
    :param verbosity: amount of output to generate
    """

    # Make configurations and check
    cfg_1 = PipelineConfig(telescope=telescope_1, band=band_1,
                           pipeline=pipeline_1, adjusts=adjusts_1)
    cfg_2 = PipelineConfig(telescope=telescope_2, band=band_2,
                           pipeline=pipeline_2, adjusts=adjusts_2)
    if not check_pipeline_config(cfg_1, pure_pipelines=True) or \
       not check_pipeline_config(cfg_2, pure_pipelines=True):
        return

    # Determine which rows to show
    (result_map, result_titles, result_units) = mk_result_map_rows(verbosity)

    # Loop through telescope configurations, collect results
    display(HTML('<font color="blue">Computing the result -- this may take several seconds.</font>'))
    detailed = (verbosity=='Debug')
    tels_result_values = [
        _compute_results(cfg_1, result_map, detailed, detailed),
        _compute_results(cfg_2, result_map, detailed, detailed),
    ]
    display(HTML('<font color="blue">Done computing.</font>'))

    # Show comparison table
    show_table_compare('Computed Values', result_titles, tels_result_values[0],
                                     tels_result_values[1], result_units)

    # Show comparison stacked bars
    products_1 = tels_result_values[0][-1]
    products_2 = tels_result_values[1][-1]
    labels = sorted(set(products_1).union(products_2))
    colours = default_rflop_plotting_colours(labels)
    telescope_labels = (cfg_1.describe(), cfg_2.describe())

    values = {
        label: (products_1.get(label,0),products_2.get(label,0))
        for label in labels
    }

    plot_stacked_bars('Computational Requirements (PetaFLOPS)', telescope_labels, labels, values,
                                    colours)


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

    :param telescope:
    :param band:
    :param pipeline:
    :param Nfacet:
    :param Tsnap:
    :param max_baseline:
    :param Nf_max:
    :param Nfacet:
    :param Tsnap:
    :param blcoal: Baseline dependent coalescing (before gridding)
    :param on_the_fly:
    :param verbosity:
    """

    assert Nfacet > 0
    assert Tsnap > 0

    # Make configuration
    cfg = PipelineConfig(telescope=telescope, pipeline=pipeline, band=band,
                         Bmax=max_baseline, Nf_max=Nf_max, blcoal=blcoal,
                         on_the_fly=on_the_fly,
                         scale_predict_by_facet=scale_predict_by_facet)
    if not check_pipeline_config(cfg, pure_pipelines=True): return

    display(HTML('<font color="blue">Computing the result -- this may take several seconds.'
                 '</font>'))

    # Determine which rows to calculate & show
    (result_map, result_titles, result_units) = mk_result_map_rows(verbosity)

    # Loop through pipelines
    detailed = (verbosity=='Debug')
    result_values = _compute_results(cfg, result_map, detailed, detailed,
                                     adjusts=dict(Tsnap=Tsnap, Nfacet=Nfacet))

    # Show table of results
    display(HTML('<font color="blue">Done computing. Results follow:</font>'))
    show_table('Computed Values', result_titles, result_values, result_units)

    # Show pie graph of FLOP counts
    values = result_values[-1]  # the last value
    colours = default_rflop_plotting_colours(set(values))
    plot_pie('FLOP breakdown for %s' % telescope, values.keys(), list(values.values()), colours)


def seconds_to_hms(seconds):
    """
    Converts a given number of seconds into hours, minutes and
    seconds, returned as a tuple. Useful for display output

    :param seconds:
    :returns: (hours, minutes, seconds)
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return (h, m, s)


def evaluate_hpso_optimized(hpso_key, blcoal=True,
                            on_the_fly=False,
                            scale_predict_by_facet=True,
                            verbosity='Overview'):
    """
    Evaluates a High Priority Science Objective by optimizing NFacet and Tsnap to minimize the total FLOP rate

    :param hpso:
    :param blcoal: Baseline dependent coalescing (before gridding)
    :param on_the_fly:
    :param verbosity:
    """

    tp_default = ParameterContainer()
    ParameterDefinitions.apply_global_parameters(tp_default)
    ParameterDefinitions.apply_hpso_parameters(tp_default, hpso_key)
    telescope = tp_default.telescope
    hpso_pipeline = tp_default.pipeline

    # First we plot a table with all the provided parameters
    param_titles = ('HPSO Number', 'Telescope', 'Pipeline', 'Max Baseline', 'Max # of channels', 'Observation time',
                    'Texp (not used in calc)', 'Tpoint (not used in calc)')
    (hours, minutes, seconds) = seconds_to_hms(tp_default.Tobs)
    Tobs_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
    (hours, minutes, seconds) = seconds_to_hms(tp_default.Texp)
    Texp_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
    (hours, minutes, seconds) = seconds_to_hms(tp_default.Tpoint)
    Tpoint_string = '%d hr %d min %d sec' % (hours, minutes, seconds)
    param_values = (hpso_key, telescope, hpso_pipeline, tp_default.Bmax, tp_default.Nf_max, Tobs_string, Texp_string,
                    Tpoint_string)
    param_units = ('', '', '', 'm', '', '', '', '')
    show_table('Parameters', param_titles, param_values, param_units)

    # Make and check pipeline configuration
    cfg = PipelineConfig(hpso=hpso_key, blcoal=blcoal, on_the_fly=on_the_fly,
                         scale_predict_by_facet=scale_predict_by_facet)
    if not check_pipeline_config(cfg, pure_pipelines=True): return

    # Determine which rows to calculate & show
    (result_map, result_titles, result_units) = mk_result_map_rows(verbosity)

    # Compute
    detailed = (verbosity=='Debug')
    result_values = _compute_results(cfg, result_map, detailed, detailed)
    display(HTML('<font color="blue">Done computing. Results follow:</font>'))

    # Show table of results
    show_table('Computed Values', result_titles, result_values, result_units)

    # Show pie graph of FLOP counts
    values = result_values[-1]  # the last value
    colours = default_rflop_plotting_colours(set(values))
    plot_pie('FLOP breakdown for %s' % telescope, values.keys(), list(values.values()), colours)


def evaluate_telescope_optimized(telescope, band, pipeline, max_baseline="default", Nf_max="default",
                                 blcoal=True, on_the_fly=False, scale_predict_by_facet=True, verbosity='Overview'):
    """
    Evaluates a telescope with manually supplied parameters, but then automatically optimizes NFacet and Tsnap
    to minimize the total FLOP rate for the supplied parameters

    :param telescope:
    :param band:
    :param pipeline:
    :param max_baseline:
    :param Nf_max:
    :param blcoal: Baseline dependent coalescing (before gridding)
    :param on_the_fly:
    :param verbosity:
    """

    # Make configuration
    cfg = PipelineConfig(telescope=telescope, pipeline=pipeline,
                         band=band, Bmax=max_baseline,
                         Nf_max=Nf_max, blcoal=blcoal,
                         on_the_fly=on_the_fly,
                         scale_predict_by_facet=scale_predict_by_facet)
    if not cfg.is_valid(pure_pipelines=True)[0]: return

    # Determine rows to show
    (result_map, result_titles, result_units) = mk_result_map_rows(verbosity)

    # Compute
    display(HTML('<font color="blue">Computing the result -- this may take several seconds.'
                 '</font>'))
    detailed = (verbosity=='Debug')
    result_values = _compute_results(cfg, result_map, detailed, detailed)
    display(HTML('<font color="blue">Done computing. Results follow:</font>'))

    # Make table
    show_table('Computed Values', result_titles, result_values, result_units)

    # Make pie plot
    values = result_values[-1]  # the last value
    colours = default_rflop_plotting_colours(set(values))
    plot_pie('FLOP breakdown for %s' % telescope, values.keys(), list(values.values()), colours)


def _pipeline_configurations(telescopes, bands, pipelines, adjusts={}):
    """Make a list of all valid configuration combinations in the list."""

    configs = []
    for telescope in telescopes:
        for band in bands:
            for pipeline in pipelines:
                cfg = PipelineConfig(telescope=telescope, band=band,
                                     pipeline=pipeline,
                                     adjusts=adjusts)

                # Check whether the configuration is valid
                (okay, msgs) = cfg.is_valid()
                if okay:
                    configs.append(cfg)

    return configs


def write_csv_pipelines(filename, telescopes, bands, pipelines, adjusts="",
                        verbose=False, parallel=0):
    """
    Evaluates all valid configurations of this telescope and dumps the
    result as a CSV file.
    """

    # Make configuration list
    configs = _pipeline_configurations(telescopes, bands, pipelines, adjusts)

    # Calculate
    rows = RESULT_MAP # Everything - hardcoded for now
    results = _batch_compute_results(configs, rows, parallel, verbose, True)

    # Write CSV
    _write_csv(filename, results, rows)


def write_csv_hpsos(filename, hpsos,adjusts="",verbose=False,parallel=0):
    """
    Evaluates all valid configurations of this telescope and dumps the
    result as a CSV file.
    """

    # Make configuration list
    configs = []
    for hpso in hpsos:
        cfg = PipelineConfig(hpso=hpso, adjusts=adjusts)
        configs.append(cfg)

    # Calculate
    rows = RESULT_MAP # Everything - hardcoded for now
    results = _batch_compute_results(configs, rows, parallel, verbose, True)

    # Write CSV
    _write_csv(filename, results, rows)


def _compute_results(pipelineConfig, result_map, verbose=False, detailed=False, adjusts={}):
    """A private method for computing a set of results.

    :param pipelineConfig: Complete pipeline configuration
    :param result_map: results to produce
    :param verbose: Chattiness of parameter generation
    :param detailed: Produce detailed output results?
    :returns: result value array
    """

    # Loop through pipeliness to collect result values
    result_value_array = []
    for pipeline in pipelineConfig.relevant_pipelines:

        # Calculate the telescope parameters
        pipelineConfig.pipeline = pipeline
        tp = pipelineConfig.calc_tel_params(verbose=verbose, adjusts=adjusts)

        # Evaluate expressions from map
        result_expressions = get_result_expressions(result_map, tp)
        results_for_pipeline = imp.evaluate_expressions(result_expressions, tp)
        result_value_array.append(results_for_pipeline)

    # Now transpose, then sum up results from pipelines per row
    result_values = []
    transposed_results = zip(*result_value_array)
    sum_results = get_result_sum(result_map)
    for (row_values, sum_it) in zip(transposed_results, sum_results):
        # Sum up baseline dependency unless in detailed mode
        if not detailed and all([isinstance(vals, list) for vals in row_values]):
            if sum_it:
                row_values = [ sum(vals) for vals in row_values ]
            else:
                row_values = [ [vals[0],"..",vals[-1]] for vals in row_values ]
        # Then also try to sum up pipeline results, if possible
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


def _batch_compute_results(configs, result_map, parallel=0, verbose=False, detailed=False, quiet=False):
    """Calculate a whole bunch of pipeline configurations. """

    if not quiet:
        display(HTML('<font color="blue">Calculating %d configurations -- this may take quite a while.</font>' %
                     len(configs)))

    # Parallelise if requested
    if parallel > 0:
        configQueue = pymp.shared.list(configs)
        results = pymp.shared.dict()
        f = FloatProgress(min=0, max=len(configs))
        display(f)
        with pymp.Parallel(parallel) as p:
            try:
                while True:
                    if p.thread_num == 0:
                        f.value = len(configs) - len(configQueue)
                        f.description = "%d/%d" % (f.value, len(configs))
                    config = configQueue.pop()
                    results[config.describe()] = _batch_compute_results(
                        [config], result_map, verbose=verbose, detailed=detailed,quiet=True)
            except IndexError:
                pass
        return list([res for cfg in configs for res in results[cfg.describe()]])

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
        results.append((cfg, _compute_results(cfg, result_map, verbose=verbose, detailed=detailed)))
    return results


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


def compare_csv(result_file, ref_file,
                ignore_modifiers=True, ignore_units=True,
                row_threshold=0.01,
                export_html=''):
    """
    Read and compare two CSV files with telescope parameters

    :param result_file: CVS file with telescope parameters
    :param ref_file: CVS file with reference parameters
    :param ignore_modifiers: Ignore modifiers when matching columns (say, [blcoal])
    :returns: Sum of differences found in specified rows
    """

    # Read results and reference. Make lookup dictionary for the latter.
    results = _read_csv(result_file)
    ref = dict(_read_csv(ref_file))

    # Strip modifiers from rows
    def strip_modifiers(head, do_it=True):
        if do_it:
            return re.sub('\[[^\]]*\]', '', head).strip(' ')
        return head
    ref = { strip_modifiers(name, ignore_units): row
            for (name, row) in ref.items() }

    # Headings
    stbl = '<table><tr><th></th><th>Mean</th><th>Min</th><th>Max</th>'
    for head, _ in results[0][1]:
        stbl += '<th>%s</th>' % head
    stbl += '</tr>\n'

    # Loop through rows
    all_diffs = []
    all_diff_sums = []
    for row_name, row in results:

        # Heading?
        if row_name.startswith('--'):
            stbl += '<tr><th colspan="%d" style="text-align:left">%s</th></tr>' % (len(row)+4, row_name)
            continue
        else:
            shead = '<tr><td>%s</td>' % row_name

        # Accumulate difference?
        diffs = []

        # Locate reference results
        refRow = ref.get(strip_modifiers(row_name, ignore_units), [])
        refRow = map(lambda h_v: (strip_modifiers(h_v[0], ignore_modifiers), h_v[1]), refRow)
        refRow = dict(refRow)

        # Loop through values
        s = ""
        for head, val in row:
            head = strip_modifiers(head, ignore_modifiers)

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
                    diffs.append(diff)

                # Output
                if not diff is None:
                    s += '<td bgcolor="#%2x%2x00">%s&nbsp;<small>(%+d%%)</small></td>' % (
                        int(min(diff_rel/50*255, 255)),
                        int(255-min(max(0, diff_rel-50)/50*255, 255)),
                        format_result(num),
                        diff)
                else:
                    s += '<td>%s</td>' % format_result(num)

            except ValueError:

                # Get reference as string
                ref_str = refRow.get(head)

                # No number, output as is
                if not ref_str is None:
                    if val == ref_str:
                        if val == '':
                            s += '<td></td>'
                        else:
                            s += '<td bgcolor="#00ff00">%s&nbsp;<small>(same)</small></td>' % val
                            diffs.append(0)
                    else:
                        s += '<td bgcolor="#ffff00">%s&nbsp;<small>(!= %s)</small></td>' % (val, ref_str)
                        diffs.append(100)
                else:
                    s += '<td>%s</td>' % val

        all_diffs.append((row_name, diffs))
        s += '</tr>\n'
        if len(diffs) != 0:
            sdiff = '<td>%+.3g%%</td><td>%+.3g%%</td><td>%+.3g%%</td>' % (
                np.mean(diffs), np.min(diffs), np.max(diffs))
            all_diff_sums.append((row_name, np.mean(diffs), np.min(diffs), np.max(diffs)))
        else:
            sdiff = '<td colspan=3></td>'
        if len(diffs) == 0 or np.max(np.abs(diffs)) >= row_threshold:
            stbl += shead + sdiff + s

    stbl += '</table>'

    # Write HTML report to file - or display
    if export_html != '':
        f = open(export_html, 'w')
        print("<!doctype html>", file=f)
        print("<html>", file=f)
        print("  <title>SDP Parametric Model Result Comparison</title>", file=f)
        print("  <body><p>Comparing %s against %s</p>" % (result_file, ref_file), file=f)
        print(stbl, file=f)
        print("  </body>", file=f)
        print("</html>", file=f)
        f.close()
        display(FileLink(export_html))
    else:
        display(HTML('<h3>Comparison:</h3>'))
        display(HTML(stbl))

    return all_diff_sums


def stack_bars_pipelines(title, telescopes, bands, pipelines,
                         adjusts={}, parallel=0,
                         save=None):
    """
    Evaluates all valid configurations of this telescope and shows
    results as stacked bars.
    """

    # Make configurations
    configs = _pipeline_configurations(telescopes, bands, pipelines, adjusts)

    # Calculate
    rows = [RESULT_MAP[-1]] # Products only
    results = _batch_compute_results(configs, rows, parallel)

    products = list(map(lambda r: r[1][-1], results))
    labels = sorted(set().union(*list(map(lambda p: p.keys(), products))))
    colours = default_rflop_plotting_colours(labels)
    tel_labels = list(map(lambda cfg: cfg.describe().replace(" ", "\n"), configs))
    values = {
        label: list(map(lambda p: p.get(label, 0), products))
        for label in labels
    }

    # Show stacked bar graph
    plot_stacked_bars(title, tel_labels, labels, values, colours, width=0.7, save=save)

def stack_bars_hpsos(title, hpsos, adjusts={}, parallel=0, save=None):
    """
    Evaluates all valid configurations of this telescope and dumps the
    result as a CSV file.
    """

    # Make configuration list
    configs = []
    for hpso in hpsos:
        cfg = PipelineConfig(hpso=hpso, adjusts=adjusts)
        configs.append(cfg)

    rows = [RESULT_MAP[-1]] # Products only
    results = _batch_compute_results(configs, rows, parallel)

    products = list(map(lambda r: r[1][-1], results))
    labels = sorted(set().union(*list(map(lambda p: p.keys(), products))))
    colours = default_rflop_plotting_colours(labels)
    tel_labels = list(map(lambda cfg: cfg.describe(), configs))
    values = {
        label: list(map(lambda p: p.get(label, 0), products))
        for label in labels
    }

    # Show stacked bar graph
    plot_stacked_bars(title, tel_labels, labels, values, colours, width=0.7,
                      xticks_rot='vertical',
                      save=save)
