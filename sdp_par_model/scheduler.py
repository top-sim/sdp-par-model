"""
A collection of objects and methods for doing simple SDP task scheduling. This is used for simulating the effect
of processing sequences of tasks on the SDP.
"""
from .parameters.definitions import Constants as c
from .parameters.definitions import HPSOs
from .config import PipelineConfig
import sdp_par_model.reports as iapi  # PyCharm doesn't like this import statement but it is correct
import warnings
import bisect

class Definitions:
    """
    Definitions that are used in the SDP task scheduler
    """
    epsilon = 1e-12  # Used for numerical stability (rounding errors)

    # Needs some refactoring methinks; idea would be to specify HPSOs instead of "letters".
    hpso_lookup = {'A': HPSOs.hpso01,
                   'B': HPSOs.hpso04c,  # TODO: This task non-imaging. Interesting use case.
                   'C': HPSOs.hpso13,
                   'D': HPSOs.hpso14,
                   'E': HPSOs.hpso15,
                   'F': HPSOs.hpso27,
                   'G': HPSOs.hpso37c}

    # The following results map was copied from examples used by Peter Wortmann. It defines values we wish to calculate.
    #               Title                      Unit       Default? Sum?             Expression
    results_map =[('Total buffer ingest rate','TeraBytes/s',True, False, lambda tp: tp.Rvis_ingest*tp.Nbeam*tp.Npp*tp.Mvis/c.tera),
                  ('Working (cache) memory',  'TeraBytes',  True, True,  lambda tp: tp.Mw_cache/c.tera,   ),
                  ('Visibility I/O Rate',     'TeraBytes/s',True, True,  lambda tp: tp.Rio/c.tera,        ),
                  ('Total Compute Rate',       'PetaFLOP/s', True, True,  lambda tp: tp.Rflop/c.peta,      )]


class DataLocation:
    ingest_buffer = "ingestbuffer"
    cold_buffer   = "coldbuffer"
    hot_buffer    = "hotbuffer"
    preserve      = "preserve"

class SDPSchedule:
    """
    An object for storing an SDP schedule
    """
    flops_deltas = None  # maps wall clock times to SDP FLOPS allocation / deallocation (+/-) sizes
    memory_deltas = None  # maps wall clock times to SDP working memory (RAM) allocation / deallocation (+/-) sizes
    hot_buffer_deltas = None  # maps wall clock times to hot buffer allocation / deallocation (+/-) sizes
    cold_buffer_deltas = None  # maps wall clock times to cold buffer allocation / deallocation (+/-) sizes
    preserve_deltas = None  # maps wall clock times to long term preserve allocation sizes (presumably only +)
    tasks_to_timestamps = {}  # Maps tasks to completion times
    timestamps_to_tasks = {}  # Maps completion times to tasks lists

    def __init__(self):
        flops_deltas = {0: 0}  # maps wall clock times to SDP FLOPS allocation / deallocation (+/-) sizes
        memory_deltas = {0: 0}  # maps wall clock times to SDP working memory (RAM) allocation / deallocation (+/-) sizes
        hot_buffer_deltas = {0: 0}  # maps wall clock times to hot buffer allocation / deallocation (+/-) sizes
        cold_buffer_deltas = {0: 0}  # maps wall clock times to cold buffer allocation / deallocation (+/-) sizes
        preserve_deltas = {0: 0}  # maps wall clock times to long term preserve allocation sizes (presumably only +)


class SDPTask:
    uid = None  # Unique ID - must be defined at task creation. Used for hashing and equality.
    description = None  # A human-readable description of the task
    t_min_start = None  # Earliest wall clock time (in seconds) that this task can / may start
    preq_tasks = None  # Prerequisite tasks that needs to complete before this one can start (default = empty set)
    t_fixed = None  # fixed minimum duration (in seconds) of this task (e.g. for an observation)
    data_in = None  # Amount of data (in TB) this task requires to read from (presumably hot) buffer *before* starting
    memsize = None  # The amount of SDP working memory (in TB) needed to perform this task
    flopcount = None  # Number of operations (in PetaFLOP) required to complete this task
    data_out = None  # Amount of data (in TB) this task stores *after* finishing (written to buffer or long-term preserve)
    data_source = None  # Data Location
    data_target = None  # Data Location

    def __init__(self, unique_id, description=None):
        self.uid = unique_id
        self.description = description
        self.preq_tasks = set()

    def __hash__(self):
        return hash(self.uid)  # Required for using Task objects in sets.

    def __eq__(self, other):
        if not isinstance(other, SDPTask):
            return False
        return (self.uid == other.uid)

    def __str__(self):
        s = "SDPTask #%d (type undefined): " % self.uid
        if self.description is not None:
            s = "SDPTask #%d (%s):" % (self.uid, self.description)
        fields = self.__dict__
        for key_string in fields.keys():
            if key_string == 'uid':
                continue  # We already printed the uid
            elif key_string == 'preq_tasks':
                # Prevent recursion by not printing the prerequisite tasks, but only their UIDs
                preq_tasks = fields[key_string]
                assert isinstance(preq_tasks, set)
                value_string = 'None'
                if len(preq_tasks) > 0:
                    value_string = ''
                    for task in fields[key_string]:
                        value_string += "#%d," % task.uid
                    if len(fields[key_string]) > 0:
                        value_string = value_string[:-1]
            else:
                value_string = str(fields[key_string])

            if len(value_string) > 40:
                value_string = value_string[:40] + "... (truncated)"
            s += "\n %s\t\t= %s" % (key_string, value_string)
        return s

    def set_param(self, param_name, value, prevent_overwrite=True, require_overwrite=False):
        """
        Method for setting a parameter in a safe way. By default first checks that the value has not already been defined.
        Useful for preventing situations where values may inadvertently be overwritten.

        :param param_name: The name of the parameter/field that needs to be assigned - provided as text
        :param value: the value to be written (as actual data type, i.e. not necessarily text)
        :param prevent_overwrite: Disallows this value to be overwritten once defined. Default = True.
        :param require_overwrite: Only allows value to be changed if it already exists. Default = False.
        """
        assert isinstance(param_name, str)
        if prevent_overwrite:
            if require_overwrite:
                raise AssertionError(
                    "Cannot simultaneously require and prevent overwrite of parameter '%s'" % param_name)

            if hasattr(self, param_name):
                if eval('self.%s == value' % param_name):
                    print('reassigning value for %s = %s' % (param_name, str(value)))  # TODO remove
                    warnings.warn('Inefficiency : reassigning parameter "%s" with same value as before.' % param_name)
                else:
                    try:
                        assert eval('self.%s == None' % param_name)
                    except AssertionError:
                        raise AssertionError(
                            "The parameter %s has already been defined and may not be overwritten." % param_name)

        elif require_overwrite and (not hasattr(self, param_name)):
            raise AssertionError("Parameter '%s' is undefined and therefore cannot be assigned" % param_name)

        exec('self.%s = value' % param_name)  # Write the value


class Scheduler:
    @staticmethod
    def compute_performance_lookup_table():
        """
        Builds a lookup table that maps each HPSO to its Subtasks, and Subtasks to to Performance Requirements.
        Useful when doing scheduling so that values do not need to be computed more than once
        :return: A dictionary of dictionaries. Maps each HPSO to a dictionary of subtasks that each maps to a performance value accordint to
        """
        performance_dict = {}

        # As a test we loop over all HPSOs we wish to handle, computing results for each
        for task_letter in sorted(Definitions.hpso_lookup.keys()):
            hpso = Definitions.hpso_lookup[task_letter]
            print('*** Processing task type %s => %s ***\n' % (task_letter, hpso))
            if not hpso in performance_dict:
                performance_dict[hpso] = {}

            for subtask in HPSOs.hpso_subtasks[hpso]:
                print('subtask -> %s' % subtask)
                if not subtask in performance_dict[hpso]:
                    performance_dict[hpso][subtask] = {}

                cfg = PipelineConfig(hpso=hpso, hpso_subtask=subtask)
                (valid, msgs) = cfg.is_valid()
                if not valid:
                    print("Invalid configuration!")
                    for msg in msgs:
                        print(msg)
                    raise AssertionError("Invalid config")
                tp = cfg.calc_tel_params()
                results = iapi._compute_results(cfg, False, Definitions.results_map)  # TODO - refactor this method's parameter sequence

                performance_dict[hpso]['Tobs'] = tp.Tobs  # Observation time
                performance_dict[hpso][subtask]['ingestRate'] = results[0]
                performance_dict[hpso][subtask]['cache'] = results[1]
                performance_dict[hpso][subtask]['visRate'] = results[2]
                performance_dict[hpso][subtask]['compRate'] = results[3]

                print('Buffer ingest rate\t= %g TB/s' % results[0])
                print('Cache memory\t= %g TB' % results[1])
                print('Visibility IO rate\t= %g TB/s' % results[2])
                print('Compute Rate\t= %g PetaFLOP/s' % results[3])
                print()

        print('done')
        return performance_dict

    @staticmethod
    def task_letters_to_SDPTask_list(letter_sequence, performance_dict):
        """
        Converts a list of task letters into a list of SDPTask objects
        @param letter_sequence : a sequence of HPSOs, defined by Rosie's A..G lettering scheme. TODO: replace by actual HPSO names
        @param performance_dict : a dictionary of computational requirements for each HPSO; these need to be computed only once
        """
        tasks = []
        uid = -1
        prev_ingest_task = None
        for task_letter in letter_sequence:
            hpso = Definitions.hpso_lookup[task_letter]
            hpso_subtasks = HPSOs.hpso_subtasks[hpso]
            nr_subtasks = len(hpso_subtasks)

            assert nr_subtasks >= 2  # We assume that the tast as *at least* an Ingest and an RCal component
            if not (hpso_subtasks[0] in HPSOs.ingest_subtasks) and (hpso_subtasks[1] in HPSOs.rcal_subtasks):
                # this is assumed true for all HPSOs - hence raising an assertion error if not
                raise AssertionError("Assumption was wrong - some HPSO apparently doesn't involve Ingest + RCal")

            # Ingest and Rcal are combined into a a single task object, as they cannot be separated
            uid += 1  # the unique id of the combined Ingest+Rcal task
            t = SDPTask(uid, 'Ingest+RCal')
            if prev_ingest_task is not None:
                t.preq_tasks.add(prev_ingest_task)  # previous ingest task needs to be complete before this one can start
            t.set_param("t_fixed", performance_dict[hpso]['Tobs'])
            t.set_param("data_source", DataLocation.ingest_buffer)
            t.set_param("data_target", DataLocation.cold_buffer)
            prev_ingest_task = t  # current (ingest+rcal) task remembered for later; subtasks may only start once this completes

            # Ingest
            subtask = hpso_subtasks[0]
            data_in = 0  # data is acquired in real time. TODO: How much?
            memsize = performance_dict[hpso][subtask]['cache']
            flopcount = t.t_fixed * performance_dict[hpso][subtask]['compRate']
            data_out = t.t_fixed * performance_dict[hpso][subtask]['ingestRate']  # ingested data to [cold] buffer.

            # RCal
            subtask = hpso_subtasks[1]
            data_in += 0  # data is acquired in real time. TODO: How much?
            memsize += performance_dict[hpso][subtask]['cache']
            flopcount += t.t_fixed * performance_dict[hpso][subtask]['compRate']
            data_out += 0  # What output does RCal generate? Just set it to zero for now. Are ingestRate and visRate relevant?

            # Set the aggregated task parameters that were computed
            t.set_param("data_in", data_in)
            t.set_param("memsize", memsize)
            t.set_param("flopcount", flopcount)
            t.set_param("data_out", data_out)

            tasks.append(t)
            ingest_rcal_data = data_out  # This is the data generated by the combined Ingest+RCal task; needed by subsequent tasks

            # Now handle the rest of the subtasks (if there are any)
            ical_task = None
            for i in range(2, nr_subtasks):
                subtask = hpso_subtasks[i]
                uid += 1
                t = SDPTask(uid, str(subtask))
                t.set_param("data_in", ingest_rcal_data)  # Shared by all these tasks; needs not to be copied every time
                # TODO - replace "buffersize" as task field. Should be a 'data object' whose movement is modelled separately
                t.set_param("memsize", performance_dict[hpso][subtask]['cache'])
                t.set_param("flopcount", performance_dict[hpso]['Tobs'] * performance_dict[hpso][subtask]['compRate'])

                if i == 2:
                    assert subtask in HPSOs.ical_subtasks  # Assumed that this is an ical subtask
                    t.preq_tasks.add(prev_ingest_task)  # Associated ingest task must complete before this one can start
                    ical_task = t  # Remember this task, as DPrep tasks depend on it
                elif subtask in HPSOs.dprep_subtasks:
                    assert ical_task is not None
                    t.preq_tasks.add(ical_task)

                t.data_out = 0  # TODO: No idea what gets output here. visRate? Temporarily set to zero

                tasks.append(t)
        return tasks

    @staticmethod
    def sum_deltas(deltas_history, timepoint, sorted_delta_keys=None, value_min=None, value_max=None, eps=1e-15):
        """
        Sums all the deltas chronologically up to the timestamp t
        @param deltas_history  : a dictionary that maps wall clock timestamps to a resource's value-changes
        @param timepoint       : The time until which the values should be summed
        @param sorted_delta_keys : Optional sorted timestamps; prevents re-sorting the timestamps for efficiency
        @param value_min       : Lowest allowable value for the resource's balance. Default zero.
        @param value_max       : Higest allowable value for the resource's balance. Default None.
        @param eps             : Numerical rounding tolerance
        @return                : The sum of the deltas from the beginning up to (and including) the timepoint.
                                 Returns False if value_min of value_max are violated
        """
        timestamps_sorted = sorted_delta_keys
        if timestamps_sorted is None:
            timestamps_sorted = sorted(deltas_history.keys())

        stop_before_index = bisect.bisect_left(timestamps_sorted, timepoint)
        if timepoint in deltas_history:
            stop_before_index += 1  # The position found by bisect needs to be included in the summation
        assert stop_before_index > 0  # The chosen time point cannot precede the first entry

        value_at_t = 0
        for i in range(stop_before_index):
            value_at_t += deltas_history[timestamps_sorted[i]]
            if (((value_min is not None) and (value_at_t + eps < value_min)) or
                    ((value_max is not None) and (value_at_t - eps > value_max))):
                raise Exception("Sum of deltas leads to value of %g at time %g sec. Outside imposed bounds of "
                                "[%s,%s] " % (value_at_t, timestamps_sorted[i], value_min, value_max))

        return value_at_t

    @staticmethod
    def find_suitable_time(timestamp_deltas, timepoint, sorted_delta_keys=None, value_min=None, value_max=None,
                           eps=1e-15):
        """
        Finds the earliest time >= timepoint when the sum of deltas is between value_min and value_max (if defined)
        @param timestamp_deltas : dictionary that maps wall clock timestamps to a resource's value-changes
        @param timepoint        : The earliest point in time for the insertion
        @param sorted_delta_keys : Optional sorted timestamps; prevents re-sorting the timestamps for efficiency
        @param value_min        : Lowest allowable value of the resource's balance for insertion.
        @param value_max        : Higest allowable value of the resource's balance for insertion.
        @param eps              : Numerical rounding tolerance
        @return                 : The timestamp. None, if none found.
        """
        # First cover the trivial case where there is no requirement for a suitable value at t=timepoint.
        if (value_min is None) and (value_max is None):
            return timepoint

        timestamps_sorted = sorted_delta_keys
        if timestamps_sorted is None:
            timestamps_sorted = sorted(timestamp_deltas.keys())

        value_at_t = Scheduler.sum_deltas(timestamp_deltas, timepoint, timestamps_sorted)

        # Check whether the value at the supplied time-point is suitable.
        if not (((value_min is not None) and (value_at_t + eps < value_min)) or
                    ((value_max is not None) and (value_at_t - eps > value_max))):
            return timepoint

        # Otherwise, we continue searching until we find a suitable timepoint
        start_at_index = bisect.bisect_left(timestamps_sorted, timepoint)
        if timepoint in timestamp_deltas:
            start_at_index += 1  # value at pos found by bisect already included in summation; increment start pos.
        assert start_at_index > 0

        for i in range(start_at_index, len(timestamps_sorted)):
            t = timestamps_sorted[i]
            value_at_t += timestamp_deltas[t]
            if not (((value_min is not None) and (value_at_t + eps < value_min)) or
                        ((value_max is not None) and (value_at_t - eps > value_max))):
                return t

        # Otherwise, no valid timepoint has been found!
        raise Exception("No valid time point found!")
        return None

    @staticmethod
    def add_delta(deltas_history, delta, t_start, t_end=None, value_min=0, value_max=None, sorted_delta_keys=None,
                  eps=Definitions.epsilon):
        """
        Applies the delta with proposed start and end times (wall clock) to a resource's simulated value change history.
        @param deltas_history  : a dictionary that maps wall clock timestamps to a resource's value-changes
        @param delta           : the change that will be added at t_start and reversed at t_end (iff t_end not None)
        @param t_start         : the wall clock time when the delta is applied
        @param t_end           : the wall clock time at which the delta expires (i.e. is reversed). Default None.
        @param value_min       : Lowest allowable value for the resource's balance. Default zero.
        @param value_max       : Higest allowable value for the resource's balance. Default None.
        @param sorted_delta_keys : Optional sorted timestamps; prevents re-sorting the timestamps for efficiency
        """
        if t_end is not None:
            assert t_end >= t_start
        if (value_min is not None) and (value_max is not None):
            assert value_max >= value_min

        # TODO -- originally we checked whether the delta can be added; now we just do it regardless. Reintroduce check
        '''
        if sorted_delta_keys is None:
            sorted_delta_keys = sorted(deltas_history.keys())

        # We now insert the deltas into the variable's history, and then sum it until the end of the delta's duration
        # to ensure that we do not violate any condition by adding this delta to the history
        deltas_new = deltas_history.copy()

        if t_start in deltas_new:
            deltas_new[t_start] += delta
            if deltas_new[t_start] == 0:
                del deltas_new[t_start]
        else:
            deltas_new[t_start] = delta

        if t_end is not None:
            if t_end in deltas_new:
                deltas_new[t_end] -= delta
                if deltas_new[t_end] == 0:
                    del deltas_new[t_end]
            else:
                deltas_new[t_end] = -delta

        # The step below sums across the whole new delta sequence to make sure that it is valid. Will raise exception if not.
        timestamps_sorted = sorted(deltas_new.keys())
        value_after = Scheduler.sum_deltas(deltas_new, timestamps_sorted[-1], timestamps_sorted, value_min, value_max, eps)
        '''

        if t_start in deltas_history:
            deltas_history[t_start] += delta
            if deltas_history[t_start] == 0:
                del deltas_history[t_start]
        else:
            deltas_history[t_start] = delta

        if t_end is not None:
            if t_end in deltas_history:
                deltas_history[t_end] -= delta
                if deltas_history[t_end] == 0:
                    del deltas_history[t_end]
            else:
                deltas_history[t_end] = -delta

    @staticmethod
    def schedule(task_list, sdp_flops, assign_sdp_fraction=0.5, minimum_flops=1.0, max_nr_iterations=9999,
                 epsilon=Definitions.epsilon):
        """
        :param task_list
        :param sdp_flops
        :param assign_sdp_fraction fraction of the available FLOPS assigned to a task that has no fixed completion time
        :param minimum_flops minimum amount of petaflops to be assigned to a task that has no fixed completion time
        :type task_list: iterable of SDPTask objects
        :return: a SDPSchedule object
        """

        # First, assert that the SDP has enough FLOP/s capacity to handle the real-time tasks. If not, we can't continue.
        tasks_to_be_scheduled = set()
        for task in task_list:
            tasks_to_be_scheduled.add(task)
            if task.t_fixed is not None:
                required_FLOPs = task.flopcount / task.t_fixed
                if (task.flopcount / task.t_fixed) > sdp_flops:
                    raise AssertionError("Task %d (%s) requires %g PetaFLOP/s. SDP capacity of %g PetaFLOPS is insufficient!"
                                         % (task.uid, task.description, required_FLOPs, sdp_flops))
        print('SDP has sufficient FLOPS capacity to handle real-time tasks.')

        # Next, iteratively run through all tasks, scheduling them as we go along (where possible).
        # Scheduling means that each task gets put in a plausible sequence.
        # Repeat until all tasks are scheduled
        nr_iterations = 0
        task_completion_times = {}  # Maps tasks to their completion timestamps

        wall_clock = 0  # Simulated wall clock time (seconds)
        wall_clock_advance = False  # Iff not true, advance the wall clock to this value

        flops_deltas = {0: 0}  # maps wall clock times to SDP FLOPS allocation / deallocation (+/-) sizes
        memory_deltas = {0: 0}  # maps wall clock times to SDP working memory (RAM) allocation / deallocation (+/-) sizes
        hot_buffer_deltas = {0: 0}  # maps wall clock times to hot buffer allocation / deallocation (+/-) sizes
        cold_buffer_deltas = {0: 0}  # maps wall clock times to cold buffer allocation / deallocation (+/-) sizes
        preserve_deltas = {0: 0}  # maps wall clock times to long term preserve allocation sizes (presumably only +)

        while len(task_completion_times) < len(tasks_to_be_scheduled):
            nr_iterations += 1
            print("-= Starting iteration %d =-" % nr_iterations)
            nr_tasks_scheduled_this_iteration = 0

            if nr_iterations > max_nr_iterations:
                print("Warning: Maximum number of iterations exceeded; aborting!")
                warnings.warn('Maximum number of iterations exceeded; aborting!')
                break

            for task in tasks_to_be_scheduled:
                # Check if task has already been scheduled.
                if task in task_completion_times:
                    continue  # Skipping this task (as it is already scheduled)

                # Check if an unfinished prerequisite task prevents this task from being scheduled at this time.
                # If so, continue to next task (we will revisit this task at a later iteration).
                elif len(task.preq_tasks) > 0:
                    unscheduled_preq_task = False
                    task_wall_clock_advance = False
                    for preq_task in task.preq_tasks:
                        if preq_task not in task_completion_times:
                            unscheduled_preq_task = True
                            break
                        elif preq_task.t_end > wall_clock:
                            if not task_wall_clock_advance:
                                task_wall_clock_advance = preq_task.t_end
                            else:
                                # advance the wall clock enough for all of this task's prerequisite tasks to complete
                                task_wall_clock_advance = max(task_wall_clock_advance, preq_task.t_end)

                    if task_wall_clock_advance:
                        if not wall_clock_advance:
                            wall_clock_advance = task_wall_clock_advance
                        else:
                            # advance global wall clock by minimum so that one additional task can complete
                            wall_clock_advance = min(wall_clock_advance, task_wall_clock_advance)

                    if unscheduled_preq_task or task_wall_clock_advance:
                        # Some prerequisite task has a) not yet been scheduled, or b) will only complete in the future
                        continue  # Skipping this task for now

                # If we reach this point in the code (without hitting a 'continue' statement), the task
                # a) has not yet been scheduled and
                # b) does not have an unfinished prerequisite task preventing it from begin scheduled at this
                #    (wall clock) point in time.
                # Hence, we can schedule it.

                flops_assigned_to_task = 0
                if (task.t_fixed is not None):
                    flops_assigned_to_task = task.flopcount / task.t_fixed
                else:
                    flops_available = sdp_flops - Scheduler.sum_deltas(flops_deltas, wall_clock, value_max=sdp_flops)
                    flops_assigned_to_task = max(minimum_flops, flops_available * assign_sdp_fraction)

                start_time = Scheduler.find_suitable_time(flops_deltas, wall_clock,
                                                          value_max=(sdp_flops - flops_assigned_to_task), eps=epsilon)
                t_start = start_time
                t_end = start_time + (task.flopcount / flops_assigned_to_task)

                task.set_param("t_start", t_start)
                task.set_param("t_end", t_end)

                Scheduler.add_delta(flops_deltas, flops_assigned_to_task, t_start, t_end, value_min=0,
                                    value_max=sdp_flops)

                Scheduler.add_delta(memory_deltas, task.memsize, t_start, t_end, value_min=0)
                # add_delta(flops_deltas, flops_required, t_start, t_end, value_min=0, value_max=sdp_flops)
                # add_delta(flops_deltas, flops_required, t_start, t_end, value_min=0, value_max=sdp_flops)
                # add_delta(flops_deltas, flops_required, t_start, t_end, value_min=0, value_max=sdp_flops)

                # Add this task to the 'task_completion_times' and 'timestamps_to_tasks' mappings
                task_completion_times[task] = t_end
                print('* Scheduled Task %d at wall clock time %g sec. Ends at t=%g sec. ' % (task.uid, t_start, t_end))
                nr_tasks_scheduled_this_iteration += 1

            print('Number of scheduled tasks after %d iterations is %d out of %d' % (nr_iterations, len(task_completion_times),
                                                                                     len(tasks_to_be_scheduled)))

            if (nr_tasks_scheduled_this_iteration == 0):
                if wall_clock_advance:
                    print("-> Advancing wall clock to %g sec." % wall_clock_advance)
                    wall_clock = wall_clock_advance
                    wall_clock_advance = False
                else:
                    print("Warning! No tasks scheduled, and wall clock not advanced!")

        '''
            # Here are the task fields, for referece: 
            uid         = None  # Unique ID - must be defined at task creation. Used for hashing and equality.
            description = None  # A human-readable description of the task
            t_min_start = None  # Earliest wall clock time (in seconds) that this task can / may start
            preq_task   = None  # Prerequisite task that needs to complete before this one can start
            t_fixed     = None  # fixed minimum duration (in seconds) of this task (e.g. for an observation)
            data_in     = None  # Amount of data (in TB) this task requires to read from (presumably hot) buffer *before* starting
            memsize     = None  # The amount of SDP working memory (in TB) needed to perform this task
            flopcount   = None  # Number of operations (in PetaFLOP) required to complete this task
            data_out    = None  # Amount of data (in TB) this task stores *after* finishing (written to buffer or long-term preserve)
            data_source = None  # Data Location
            data_target = None  # Data Location
        '''
        print('Done!')

        schedule = SDPSchedule()
        schedule.cold_buffer_deltas = cold_buffer_deltas
        schedule.flops_deltas = flops_deltas
        schedule.memory_deltas = memory_deltas
        schedule.preserve_deltas = preserve_deltas
        schedule.hot_buffer_deltas = hot_buffer_deltas
        schedule.tasks_to_timestamps = task_completion_times

        return schedule