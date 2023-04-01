# Copyright (C) 26/8/20 RW Bunney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import sys
import math
import random
import time
import warnings

import numpy
from ipywidgets import interact_manual, SelectMultiple
from IPython.display import display, Markdown
from matplotlib import pylab
import matplotlib.lines

sys.path.insert(0, "..")
from sdp_par_model.scheduling import efficiency

from sdp_par_model import reports, config
from sdp_par_model.scheduling import graph, level_trace, scheduler
from sdp_par_model.parameters import definitions
from sdp_par_model.parameters.definitions import Telescopes, Pipelines, \
    Constants, HPSOs
from sdp_par_model import config

telescope = Telescopes.SKA1_Mid

# Assumptions about throughput per size for hot and cold buffer
hot_rate_per_size = 3 * Constants.giga / 10 / Constants.tera  # 3 GB/s per 10 TB [NVMe SSD]
cold_rate_per_size = 0.5 * Constants.giga / 16 / Constants.tera  # 0.5 GB/s per 16 TB [SATA SSD]

# Common system sizing
ingest_rate = 0.46 * Constants.tera  # Byte/s
delivery_rate = int(100 / 8 * Constants.giga)  # Byte/s
lts_rate = delivery_rate

# Costing scenarios to assume
scenario = "low-adjusted"
batch_parallelism = 2
if scenario == 'low-cdr':
    telescope = Telescopes.SKA1_Low
    total_flops = int(13.8 * Constants.peta)  # FLOP/s
    input_buffer_size = int((0.5 * 46.0 - 0.6) * Constants.peta)  # Byte
    hot_buffer_size = int(0.5 * 46.0 * Constants.peta)  # Byte
    delivery_buffer_size = int(0.656 * Constants.peta)  # Byte
elif scenario == 'mid-cdr':
    telescope = Telescopes.SKA1_Mid
    total_flops = int(12.1 * Constants.peta)  # FLOP/s
    input_buffer_size = int((0.5 * 39.0 - 1.103) * Constants.peta)  # Byte
    hot_buffer_size = int(0.5 * 39.0 * Constants.peta)  # Byte
    delivery_buffer_size = int(0.03 * 39.0 * Constants.peta)  # Byte
elif scenario == 'low-adjusted':
    telescope = Telescopes.SKA1_Low
    total_flops = int(9.623 * Constants.peta)  # FLOP/s
    # input_buffer_size = int(30.0 * Constants.peta) # Byte # 1
    input_buffer_size = int(43.35 * Constants.peta)  # Byte
    # hot_buffer_size = int(17.5 * Constants.peta) # Byte # 1
    hot_buffer_size = int(25.5 * Constants.peta)  # Byte # 2
    # hot_buffer_size = int(27.472 * Constants.peta) # Byte
    delivery_buffer_size = int(0.656 * Constants.peta)  # Byte
elif scenario == 'mid-adjusted':
    telescope = Telescopes.SKA1_Mid
    total_flops = int(5.9 * Constants.peta)  # FLOP/s
    input_buffer_size = int(48.455 * Constants.peta)  # Byte
    hot_buffer_size = int(40.531 * Constants.peta)  # Byte
    delivery_buffer_size = int(1.103 * Constants.peta)  # Byte
else:
    assert False, "Unknown costing scenario {}!".format(scenario)

# csv = reports.read_csv(reports.newest_csv(reports.find_csvs()))
# csv = reports.strip_csv(csv)

csv = reports.read_csv("2023-03-25_long_HPSOs.csv")

realtime_flops = 0
realtime_flops_hpso = None

for hpso in definitions.HPSOs.all_hpsos:
    if definitions.HPSOs.hpso_telescopes[hpso] != telescope:
        continue
    # Sum FLOP rates over involved real-time pipelines
    rt_flops = 0
    for pipeline in definitions.HPSOs.hpso_pipelines[hpso]:
        cfg_name = config.PipelineConfig(hpso=hpso,
                                         pipeline=pipeline).describe()
        flops = int(math.ceil(float(reports.lookup_csv(csv, cfg_name,
                                                       'Total Compute Requirement')) * definitions.Constants.peta))
        if pipeline in definitions.Pipelines.realtime:
            rt_flops += flops
    # Dominates?
    if rt_flops > realtime_flops:
        realtime_flops = rt_flops
        realtime_flops_hpso = hpso

# Show
# print("Realtime processing requirements:")
batch_flops = total_flops - realtime_flops
# print(
#     " {:.3f} Pflop/s real-time (from {}), {:.3f} Pflop/s left for batch".format(
#         realtime_flops / definitions.Constants.peta,
#         realtime_flops_hpso, batch_flops / definitions.Constants.peta))

capacities = {
    graph.Resources.Observatory: 1,
    graph.Resources.BatchCompute: batch_flops,
    graph.Resources.RealtimeCompute: realtime_flops,
    graph.Resources.InputBuffer: input_buffer_size,
    graph.Resources.HotBuffer: hot_buffer_size,
    graph.Resources.OutputBuffer: delivery_buffer_size,
    graph.Resources.IngestRate: ingest_rate,
    graph.Resources.DeliveryRate: delivery_rate,
    graph.Resources.LTSRate: lts_rate
}


def update_rates(capacities):
    capacities[graph.Resources.HotBufferRate] = hot_rate_per_size * capacities[
        graph.Resources.HotBuffer]
    capacities[graph.Resources.InputBufferRate] = cold_rate_per_size * \
                                                  capacities[
                                                      graph.Resources.InputBuffer]
    capacities[graph.Resources.OutputBufferRate] = cold_rate_per_size * \
                                                   capacities[
                                                       graph.Resources.OutputBuffer]


update_rates(capacities)

Tsequence = 2 * 3600
Tobs_min = 1 * 3600

random.seed(0)
# hpso_sequence, Tobs_sum = graph.create_fixed_sequence(telescope, verbose=True)
hpso_sequence, Tobs_sum = graph.make_hpso_sequence(telescope, Tsequence,
                                                   Tobs_min, verbose=True)
print("{:.3f} d total".format(Tobs_sum / 3600 / 24))
random.shuffle(hpso_sequence)
batch_parallelism = 1
t = time.time()
hpso_sequence = sorted(hpso_sequence)
nodes = graph.hpso_sequence_to_nodes(csv, hpso_sequence, capacities, Tobs_min,
                                     batch_parallelism=batch_parallelism,
                                     force_order=True)
# print("Multi-graph has {} nodes (generation took {:.3f}s)".format(len(nodes),
#                                                                   time.time() - t))

if True:
    for node in nodes:
        print("{} ({}, t={} s)".format(node.name, node.hpso, node.time))
        for cost, amount in node.cost.items():
            if cost in graph.Resources.units:
                unit, mult = graph.Resources.units[cost]
                print(" {}={:.2f} {}".format(cost, amount / mult, unit))
        for cost, amount in node.edge_cost.items():
            if cost in graph.Resources.units:
                unit, mult = graph.Resources.units[cost]
                print(" -> {}={:.2f} {}".format(cost, amount / mult, unit))
        print()

cost_sum = {cost: 0 for cost in capacities.keys()}
for task in nodes:
    for cost, amount in task.all_cost().items():
        assert cost in capacities, "No {} capacity defined, required by {}!".format(
            cost, task.name)
        assert amount <= capacities[
            cost], "Not enough {} capacity to run {} ({:g}<{:g}!)".format(
            cost, task.name, capacities[cost], amount)
        # Try to compute an average. Edges are the main wild-card here: We only know that they stay
        # around at least for the lifetime of the dependency *and* the longest dependent task.
        ttime = task.time
        if cost in task.edge_cost and len(task.rev_deps) > 0:
            ttime += max([d.time for d in task.rev_deps])
        cost_sum[cost] += ttime * amount
    tflops = task.time * batch_flops
    print(f'{task.name}: {tflops}')

# print("Best-case average loads:")
for cost in graph.Resources.All:
    unit, mult = graph.Resources.units[cost]
    avg = cost_sum[cost] / Tobs_sum
    cap = capacities[cost]
    # print(
    #     " {}:\t{:.3f} {} ({:.1f}% of {:.3f} {})".format(cost, avg / mult, unit,
    #                                                     avg / cap * 100,
    #                                                     cap / mult, unit))
    # Warn past 75%
    # if avg > cap:
        # print('Likely insufficient {} capacity!'.format(cost),
              # file=sys.stderr, )
#
t = time.time()
usage, task_time, task_edge_end_time = scheduler.schedule(nodes, capacities,
                                                          verbose=True)
print("Scheduling took {:.3f}s".format(time.time() - t))
print("Observing efficiency: {:.1f}%".format(
    Tobs_sum / usage[graph.Resources.Observatory].end() * 100))

trace_end = max(*task_edge_end_time.values())
print(f'End of trace: {trace_end}')
pylab.figure(figsize=(16, 16));
pylab.subplots_adjust(hspace=0.5)
for n, cost in enumerate(graph.Resources.All):
    levels = usage[cost]
    avg = levels.average(0, trace_end)
    unit, mult = graph.Resources.units[cost]
    pylab.subplot(len(usage), 1, n + 1)
    pylab.step(
        [0] + [t / 24 / 3600 for t in levels._trace.keys()] + [trace_end],
        [0] + [v / mult for v in levels._trace.values()] + [0],
        where='post')
    pylab.title("{}: {:.3f} {} average ({:.2f}%)".format(
        cost, avg / mult, unit, avg / capacities[cost] * 100))
    pylab.xlim((0, trace_end / 24 / 3600));
    pylab.xticks(range(int(trace_end) // 24 // 3600 + 1))
    pylab.ylim((0, capacities[cost] / mult * 1.01))
    pylab.ylabel(unit)
    if n + 1 < len(graph.Resources.All):
        pylab.gca().xaxis.set_ticklabels([])
pylab.xlabel("Days")
pylab.show()

import multiprocessing

interesting_costs = [graph.Resources.BatchCompute, graph.Resources.InputBuffer,
                     graph.Resources.HotBuffer, graph.Resources.OutputBuffer]
linked_cost = {
    graph.Resources.HotBuffer: graph.Resources.HotBufferRate,
    graph.Resources.InputBuffer: graph.Resources.InputBufferRate,
    graph.Resources.OutputBuffer: graph.Resources.OutputBufferRate,
}
# Assumed price to add capacity
cost_gradient = {
    graph.Resources.BatchCompute: 1850000 / Constants.peta,
    graph.Resources.RealtimeCompute: 1850000 / Constants.peta,
    graph.Resources.HotBuffer: 80000 / Constants.peta,
    graph.Resources.InputBuffer: 45000 / Constants.peta,
    graph.Resources.OutputBuffer: 45000 / Constants.peta,
}
# Assumed price of entire telescope to assign cost to inefficiences
total_cost = 250 * Constants.mega


@interact_manual(
    costs=SelectMultiple(options=graph.Resources.All, value=interesting_costs),
    percent=(1, 100, 1), percent_step=(1, 10, 1), count=(1, 100, 1),
    yaxis_range=(1, 20, 1),
    batch_parallelism=(1, 10, 1))
def test_sensitivity(costs=interesting_costs, percent=50, percent_step=5,
                     count=multiprocessing.cpu_count(),
                     batch_parallelism=batch_parallelism, yaxis_range=5,
                     cost_change=False):
    # Calculate
    lengths = efficiency.determine_durations_batch(csv,
                                                   hpso_sequence, costs,
                                                   capacities, update_rates,
                                                   percent, percent_step, count,
                                                   Tobs_min=Tobs_min,
                                                   batch_parallelism=batch_parallelism)
    # Make graph
    graph_count = len(costs)
    pylab.figure(figsize=(8, graph_count * 4))
    pylab.subplots_adjust(hspace=0.4)
    for graph_ix, cost in enumerate(costs):
        pylab.subplot(graph_count, 1, graph_ix + 1)
        efficiency.plot_efficiencies(pylab.gca(), Tobs_sum, cost,
                                     capacities[cost], lengths[cost],
                                     linked_cost.get(cost), update_rates,
                                     cost_gradient.get(
                                         cost) if cost_change else None,
                                     total_cost)


scheduler.schedule(nodes, capacities, task_time, task_edge_end_time,
                   verbose=True);

new_capacities = dict(capacities)
new_capacities[graph.Resources.InputBuffer] = capacities[
                                                  graph.Resources.InputBuffer] // 2
usage2, task_time2, task_edge_end_time2, failed_usage2 = scheduler.reschedule(
    nodes, new_capacities, 5 * 24 * 3600, task_time, task_edge_end_time,
    verbose=False)
usage3, task_time3, task_edge_end_time3, failed_usage3 = scheduler.reschedule(
    nodes, capacities, 8 * 24 * 3600, task_time2, task_edge_end_time2,
    verbose=False)

trace_end = max(*task_edge_end_time3.values())/3600/24
print(f'End of trace: {trace_end}')
pylab.figure(figsize=(16, 16));
pylab.subplots_adjust(hspace=0.5)
for n, cost in enumerate(graph.Resources.All):
    levels = usage3[cost]
    avg = levels.average(0, trace_end)
    unit, mult = graph.Resources.units[cost]
    pylab.subplot(len(usage), 1, n + 1)
    for levels in [failed_usage2[cost] + failed_usage3[cost] + usage3[cost],
                   failed_usage2[cost] + failed_usage3[cost]]:
        pylab.step(
            [0] + [t / 24 / 3600 for t in levels._trace.keys()] + [trace_end],
            [0] + [v / mult for v in levels._trace.values()] + [0],
            where='post')
    pylab.title("{}: {:.3f} {} average ({:.2f}%)".format(
        cost, avg / mult, unit, avg / capacities[cost] * 100))
    pylab.xlim((0, trace_end / 24 / 3600));
    pylab.xticks(range(int(trace_end) // 24 // 3600 + 1))
    pylab.ylim((0, capacities[cost] / mult * 1.01))
    pylab.ylabel(unit)
    if n + 1 < len(graph.Resources.All): pylab.gca().xaxis.set_ticklabels([])
pylab.xlabel("Days")
pylab.show()
