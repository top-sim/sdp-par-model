
from . import graph, scheduler

import warnings
import random
import numpy
import matplotlib.lines

try:
    import multiprocessing
    HAVE_MP = True
except:
    warning.warn("No multiprocessing, falling back to single-threaded map. "
                 "This should work, but might be slow!")
    HAVE_MP = False

def _determine_durations(cost_amounts):
    csv, hpso_seq, caps, kwargs = cost_amounts
    # Schedule, collect efficiencies
    effs = []
    for cap in caps:
        nodes = graph.hpso_sequence_to_nodes(csv, hpso_seq, cap, **kwargs)
        try:
            usage, task_time, task_edge_end_time = scheduler.schedule(nodes, cap, verbose=False)
            effs.append(usage[graph.Resources.Observatory].end())
        except ValueError:
            effs.append(None)
    return effs

def determine_durations_batch(csv, hpso_list, costs, capacities, update_rates,
                              percent, percent_step, count, **kwargs):
    """Perform Monte-Carlo simulation of the effects of capacity changes.

    :param csv: Cached telescope parameters to use for creating graph
    :param hpso_list: List of HPSOs to permute for randomisation
    :param costs: Costs to vary
    :param capacities: Default capacities
    :param update_rates: Function to update dependent rates after adjustment
    :param percent: Variation range on costs
    :param percent_step: Variation step
    :param count: Numer of simulation runs
    :param **kwargs: Other parameters to pass to `hpso_sequence_to_nodes`
    :returns: Map of costs to a list of (capacity, durations) pairs
    """

    # Make appropriate number of HPSO sequences
    hpso_seqs = []
    for _ in range(count):
        hpso_seq = list(hpso_list)
        random.shuffle(hpso_seq)
        hpso_seqs.append(hpso_seq)
    # Multiply by all the capacity changes we would like to test
    all_work = []
    cap_modifications = numpy.arange(-percent, percent+percent_step, percent_step) / 100
    for graph_ix, cost in enumerate(costs):
        # Adjust capacity, update rates if needed
        caps = []
        for mod in cap_modifications:
            cap = dict(capacities)
            cap[cost] = cap[cost] + int(cap[cost] * mod)
            update_rates(cap)
            caps.append(cap)
        # Create work item
        all_work.extend([(csv, seq, caps, kwargs) for seq in hpso_seqs])

    # Calculate results. Will have to fall back to normal map on Windows
    if HAVE_MP:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as mp_pool:
            all_results = mp_pool.map(_determine_durations, all_work)
    else:
        all_results = map(_determine_durations, all_work)

    # Organise data
    efficiencies = {}
    for graph_ix, cost in enumerate(costs):
        amounts = capacities[cost] + (capacities[cost] * cap_modifications).astype(int)
        effs = numpy.transpose(list(all_results[graph_ix*count:(graph_ix+1)*count]))
        # Add to result, filtering out impossible amounts (where we got only "None" values)
        sel = numpy.all(effs != None, axis=1)
        efficiencies[cost] = list(zip(amounts[sel], effs[sel]))
    return efficiencies

def plot_efficiencies(axis, Tobs_min, cost, capacity, durations,
                      linked_cost = None, link_cost = lambda arg: arg,
                      cost_gradient = None, project_cost = None,
                      yaxis_range=5):

    """ Plot efficiency graphs from monte-carlo simulation results

    :param axis: Matplotlib axis to use for output
    """

    # Draw percentiles
    unit,mult = graph.Resources.units[cost]
    percents = [10,25,50,75,90]; styles = [':', '--', '-', '--', ':']
    amounts = numpy.array([ amount for amount, _ in durations ])
    percentiles = numpy.transpose([
        numpy.percentile(Tobs_min / dur * 100., percents) for _, dur in durations ])
    if cost_gradient is not None:
        adjusted_costs = project_cost + (amounts - capacity) * cost_gradient
        axis.plot(amounts/mult, 100 * (adjusted_costs / project_cost - 1),
                  linestyle = '-.', color='gray', label='optimum')
    for p, eff_ps, style in zip(percents, percentiles, styles):
        if cost_gradient is not None:
            eff_ps = 100 * (100 * adjusted_costs / eff_ps / project_cost - 1)
        axis.plot(amounts/mult, eff_ps, label="{}%".format(p), linestyle=style, color='blue')
    # Draw line for "default" value
    if cost_gradient is not None:
        axis.set_ylim((-yaxis_range, yaxis_range))
    else:
        axis.set_ylim((100-yaxis_range, 101))
    axis.add_line(matplotlib.lines.Line2D(
        [capacity/mult,capacity/mult],axis.get_ylim(), color='black', linestyle=':'))
    # Titles
    axis.set_xlabel("{} [{}]".format(cost, unit))
    if cost_gradient is not None:
        axis.set_ylabel("Approx. Cost Change [%]")
    else:
        axis.set_ylabel("Efficiency [%]")
    axis.legend()
    axis.grid()
    # Add second X axis for linked cost, if any
    if linked_cost is not None:
        unit2,mult2 = graph.Resources.units[linked_cost]
        ax2 = axis.twiny()
        def to_linked_cost(amount):
            cap = { cost : 0 for cost in graph.Resources.All }
            cap[cost] = amount * mult
            link_cost(cap)
            return cap[linked_cost] / mult2
        ax2.set_xlim(to_linked_cost(axis.get_xlim()[0]), to_linked_cost(axis.get_xlim()[1]))
        ax2.set_xlabel("{} [{}]".format(linked_cost, unit2))
