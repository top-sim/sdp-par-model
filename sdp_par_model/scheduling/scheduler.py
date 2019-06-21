
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

def schedule(tasks, capacities,
             task_time = {}, task_edge_end_time = {}, task_constraint = {},
             resource_check_start = 0, verbose=False):
    """Schedules a (multi-)graph of tasks in a way that resource usage
    stays below capacities.

    :param tasks: List of tasks, in topological order
    :param capacities: Dictionary of resource names to capacity
    :param task_constraint: Earliest allowed starting point for tasks
    :param task_time: Start schedule of tasks
    :param task_edge_end_time: Start schedule of task edges
    :return: Tuple (usage, task_time, task_edge_end_time)
    """

    # Make work copies of parameters
    task_constraint = dict(task_constraint)
    task_time = dict(task_time)
    task_edge_end_time = dict(task_edge_end_time)

    # Initialise usage, taking initial schedule into account
    usage = { res: level_trace.LevelTrace() for res in capacities }
    for task in task_time:
        _apply_cost(task, task_time[task], usage)
        _apply_edge_cost(task, task_time[task], task_edge_end_time[task], usage)

    # Make sure usage is not above capacity already
    end_of_time = 1e15
    for res in usage:
        max_usage = usage[res].maximum(resource_check_start, end_of_time)
        assert max_usage <= capacities[res], \
            "Resource {} over capacity: {} > {}!".format(res, max_usage, capacities[res])

    tasks_to_do = list([ task for task in tasks if task not in task_time])
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

def reschedule(tasks, capacities, start_time,
               task_time, task_edge_end_time,
               task_constraint = {}, verbose=False):
    """Re-schedules a (multi-)graph of tasks assuming that capacity
    changes at a certain point in time.

    The returned schedule will be identical to the one given in the
    parameter up to the starting point. Tasks that overlap the
    capacity change point might fail. If there is a choice involved in
    which task to fail, we use the order in the "tasks" list for
    priority (this is clearly somewhat arbitrary, and possibly
    optimistic). These tasks will be re-scheduled.

    :param tasks: List of tasks, in topological order
    :param capacities: Dictionary of resource names to capacity
    :param start_time: Start point of capacity change.
    :param task_time: Start schedule of tasks
    :param task_edge_end_time: Start schedule of task edges
    :param task_constraint: Earliest allowed starting point for tasks
    :return: Tuple (usage, task_time, task_edge_end_time, failed_tasks)
    """

    # Take over all tasks and edges that ended before the capacity change.
    new_task_time = {}; new_task_edge_end_time = {}
    for task, end_time in task_edge_end_time.items():
        if end_time <= start_time:
            new_task_time[task] = task_time[task]
            new_task_edge_end_time[task] = end_time
    task_constraint = dict(task_constraint)
    failed_tasks = {}
    if verbose:
        print("{} tasks unaffected".format(len(new_task_time)))

    # Now check all tasks and edges that overlap the time in
    # question. We only need to check usage at the starting point
    usage = { res: 0 for res in capacities }
    failed_usage = { res: level_trace.LevelTrace() for res in capacities }
    for task in tasks:

        # Skip if task was not scheduled in the first place, or
        # happens before (=> remains the same) or after (=> gets
        # re-scheduled unconditionally) the point in question
        if task not in task_time or \
           task_time[task] >= start_time or \
           task_edge_end_time[task] <= start_time:
            continue

        # Check if we can schedule it - resources must be there, and
        # no dependency must have failed
        impossible = []
        for dep in task.deps:
            if dep in failed_tasks:
                impossible.append("dependency")
        start = task_time[task]; end = task_time[task] + task.time
        if task_time[task]+task.time > start_time:
            for cost, amount in task.cost.items():
                if usage[cost] + amount > capacities[cost]:
                    impossible.append("{} {} > {}".format(cost, usage[cost]+amount, capacities[cost]))
                    break
        for cost, amount in task.edge_cost.items():
            if usage[cost] + amount > capacities[cost]:
                impossible.append("{} {} > {}".format(cost, usage[cost]+amount, capacities[cost]))
                break

        if verbose:
            if len(impossible) == 0:
                print("Task {} @ {} survived".format(task.name, start))
            else:
                print("Task {} @ {} failed ({})".format(
                    task.name, start, ", ".join(impossible)))

        if len(impossible) > 0:

            # This task failed due to the capacity change, and will
            # need to be restarted. Note that this explicitly includes
            # the case where the task finished, but we lost the
            # capacity to retain its results. In either case, we need
            # to re-schedule the task.
            _apply_cost(task, task_time[task], failed_usage)
            _apply_edge_cost(task, task_time[task], min(start_time, task_edge_end_time[task]), failed_usage)

        else:

            # Otherwise: Add to schedule, count usage
            new_task_time[task] = task_time[task]
            new_task_edge_end_time[task] = task_edge_end_time[task]
            if task_time[task]+task.time > start_time:
                for cost, amount in task.cost.items():
                    usage[cost] += amount
            for cost, amount in task.edge_cost.items():
                usage[cost] += amount

    # At this point we have taken over as much of the existing
    # schedule as we could, and must now re-schedule the remaining
    # tasks. All of the newly scheduled task must happen after the
    # start time, introduce a suitable constraint.
    for task in tasks:
        task_constraint[task] = max(start_time, task_constraint.get(task, 0))

    # Now re-schedule
    ret_usage, ret_task_time, ret_task_edge_end_time = schedule(
        tasks, capacities, new_task_time, new_task_edge_end_time, task_constraint,
        resource_check_start = start_time, verbose=verbose)

    # Check whether any of the tasks prior to the starting point had
    # to be re-scheduled, as that also counts as failure
    for task in new_task_time:
        if task_time[task] != ret_task_time[task]:
            _apply_cost(task, task_time[task], failed_usage)
            _apply_edge_cost(task, task_time[task], min(start_time, task_edge_end_time[task]), failed_usage)

    # Remove failed usage past starting point
    for cost in capacities:
        del failed_usage[cost][start_time:]

    return ret_usage, ret_task_time, ret_task_edge_end_time, failed_usage
