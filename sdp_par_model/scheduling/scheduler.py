
from . import level_trace

def toposort(tasks):
    """Simple topological sort routine for task lists. Not fast, but easy
    - especially leaves nodes in original order if already topologically
    sorted. Dependencies that are not in the tasks list are ignored for
    the purpose of the sort.
    """

    tasks_set = set(tasks)
    tasks_out = []
    tasks_out_set = set()
    to_do = tasks
    while len(to_do) > 0:
        new_to_do = []
        for t in to_do:
            if set(t.deps).intersection(tasks_set).issubset(tasks_out_set):
                tasks_out.append(t)
                tasks_out_set.add(t)
            else:
                new_to_do.append(t)
        to_do = new_to_do
    return tasks_out

def active_dependencies(tasks, active):
    """Walks dependencies of the given task for tasks considered
    'active'. Note that dependencies of inactive tasks are not searched.

    :param tasks: Iterable of tasks
    :param active: Set/dictionary of active tasks
    :return: Topologically sorted list of tasks
    """

    # If we re-schedule the dependency, other tasks depending
    # on it might need to get re-scheduled as well
    repeat = False
    tasks = set(tasks)
    work_set = set(tasks)
    while len(work_set) > 0:
        next_work_set = set()
        for d in work_set:
            for d2 in d.rev_deps:
                if d2 in active and d2 not in tasks:
                    tasks.add(d2)
                    next_work_set.add(d2)
        work_set = next_work_set

    return toposort(tasks)

def _apply_cost(task, time, usage, remove=False):
    for cost,amount in task.cost.items():
        usage[cost].add(time, time + task.time, -amount if remove else amount)

def _apply_edge_cost(task, time, edge_end_time, usage, remove=False):
    for cost,amount in task.edge_cost.items():
        usage[cost].add(time, edge_end_time, -amount if remove else amount)

def assert_schedule_consistency(usage, task_time, task_edge_end_time):

    usage_check = { res: level_trace.LevelTrace() for res in usage.keys() }
    for task in task_time.keys():
        _apply_cost(task, task_time[task], usage_check)
        _apply_edge_cost(task, task_time[task], task_edge_end_time[task], usage_check)
    for cost in usage:
        assert usage[cost] == usage_check[cost], usage[cost] - usage_check[cost]

def schedule(tasks, capacities, verbose=False):
    """Schedules a (multi-)graph of tasks in a way that resource usage
    stays below capacities.

    :param tasks: List of tasks, in topological order
    :param capacities: Dictionary of resource names to capacity
    :return: Tuple (usage, task_time, task_edge_end_time)
    """

    usage = { res: level_trace.LevelTrace() for res in capacities }

    task_constraint = {}
    task_time = {}
    task_edge_end_time = {}

    end_of_time = 1e15
    tasks_to_do = list(tasks)
    while len(tasks_to_do) > 0:
        task = tasks_to_do.pop(0)

        # Remove all edge cost for dependencies - we will add them back
        # once we know where to schedule this task (and therefore how
        # long the edge needs to be)
        time = task_constraint.get(task, 0)
        for d in task.deps:
            time = max(time, task_time[d] + d.time)
            _apply_edge_cost(d, task_time[d], task_edge_end_time[d], usage, remove=True)

        # Find suitable start time. Given the above check this must succeed eventually,
        # because all levels return to zero.
        finished = False
        while not finished:
            finished = True
            for cost, amount in task.all_cost().items():
                if amount > capacities[cost]:
                    raise ValueError('Task {} ({}) impossible, {} cost {} is over capacity {}!'.format(
                        task.name, task.hpso, cost, amount, capacities[cost]))
                new_time = usage[cost].find_period_below(
                    time, end_of_time, capacities[cost] - amount, task.time)
                if new_time > time:
                    if verbose:
                        print("{} for {} ({} s): Amount {} @ {}".format(
                            cost,task.name, task.time, amount, new_time))
                    time = new_time
                    finished = False

        # Alright, now check that we can extend the edges
        task_end_time = time + task.time
        need_to_reschedule = set()
        for d in task.deps:
            # Elongate edge, if needed
            task_edge_end_time[d] = max(task_edge_end_time[d], task_end_time)
            # Check that we have enough capacity to do this
            for cost,amount in d.edge_cost.items():
                if amount > capacities[cost]:
                    raise ValueError('Task {} ({}) impossible, {} edge cost {} is over capacity {}!'.format(
                        task.name, task.hpso, cost, amount, capacities[cost]))
                max_use = usage[cost].maximum(task_time[d], task_edge_end_time[d])
                # Over maximum usage? Note, but add anyway so we don't need to
                # remember what costs we added
                if max_use + amount > capacities[cost]:
                    task_constraint[d] = usage[cost].find_above_backward(time,
                                                                         capacities[cost] - amount)
                    assert usage[cost].get(task_constraint[d]) <= capacities[cost] - amount
                    if verbose:
                        print("{} for {} dependency {} ({} s): overflow ({}+{}), "
                              "need to reschedule past {}".format(
                                  cost,task.name,d.name, task_time[d],
                                  max_use, amount, task_constraint[d]))
                    need_to_reschedule.add(d)
            _apply_edge_cost(d, task_time[d], task_edge_end_time[d], usage)

        # Needs re-scheduling of dependencies?
        if len(need_to_reschedule) > 0:

            # Reschedule tasks, plus previously-scheduled dependencies
            reschedule_list = active_dependencies(need_to_reschedule, task_time)
            if verbose:
                print("Rescheduling", ", ".join([d.name for d in reschedule_list]))
            for t in reschedule_list:
                _apply_cost(t, task_time[t], usage, remove=True)
                _apply_edge_cost(t, task_time[t], task_edge_end_time[t], usage, remove=True)
                del task_time[t]
                del task_edge_end_time[t]

            # Add to front of list
            reschedule_list.append(task)
            tasks_to_do = reschedule_list + tasks_to_do
            continue

        # Finally, add the task itself
        if verbose:
            print("=> {}: {}-{}".format(task.name, time, task_end_time))
        task_time[task] = time
        task_edge_end_time[task] = task_end_time
        _apply_cost(task, task_time[task], usage)
        _apply_edge_cost(task, task_time[task], task_edge_end_time[task], usage)

    # Consistency check
    if verbose:
        assert_schedule_consistency(usage, task_time, task_edge_end_time)

    return usage, task_time, task_edge_end_time
