"""
A collection of objects and methods for doing simple SDP task scheduling. This is used for simulating the effect
of processing sequences of tasks on the SDP.

Scheduling is a potentially very hard problem. See the following page for an overview of common scheduling approaches:
https://www.cs.rutgers.edu/~pxk/416/notes/07-scheduling.html
In some cases, optimal scheduling is an intractable problem. See, for example:
https://en.wikipedia.org/wiki/Job_shop_scheduling and
https://dl.acm.org/citation.cfm?id=1740138

This scheduling problem has not been analyzed from a theoretical perspective. We attempt a pragmatic heuristic that will
be approximately correct. At the moment we cannot yet show that the scheduler will always produce a schedule that is
implementable. However, we can check the schedule for implementability after it has been created by
using the Scheduler.sum_deltas() method with maximum and minimum allowable values enforced.
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


class SDPAttributes:
    ingest_buffer = "ingestbuffer"
    working_mem   = "working_mem"
    cold_buffer   = "coldbuffer"
    hot_buffer    = "hotbuffer"
    preserve      = "preserve"

    sdp_flops = 22.8  # PetaFLOP/s

    data_locations = {ingest_buffer, working_mem, cold_buffer, hot_buffer, preserve}

    datapath_speeds = {(ingest_buffer, working_mem) : 0.5,  # all values in TB/s
                       (working_mem, cold_buffer)   : 0.5,
                       (cold_buffer, hot_buffer)    : 0.5,
                       (hot_buffer, working_mem)    : 5.0,
                       (working_mem, hot_buffer)    : 5.0,
                       (hot_buffer, preserve)       : 0.012,
                       None                         : 0.0}

class SDPTask:
    """
    An object that represents an SDP task. This typically has a subset (some of them optional, some not)
    of the following attributes:

    uid          : Unique ID defined at task creation. Used for hashing and equality.
    description  : A human-readable description of the task
    t_min_start  : Earliest wall clock time (in seconds) that this task can / may start
    t_fixed      : fixed minimum duration (in seconds) of this task (e.g. for an observation)
    datapath_in  : The path by which data is transferred towards this task's execution point
    datapath_out : The path by which data is transferred away from this task's execution point
    datasize_in  : The amount of data (in TB) that is transferred via the input path
    datasize_out : The amount of data (in TB) that is transferred via the output path
    memsize      : The amount of SDP working memory (in TB) needed to perform this task
    flopcount    : Number of operations (in PetaFLOP) required to complete this task

    Required attributes are created during in the __init__ method - note that these are all
    """
    class_uid_generator = 0  # Class variable for creating unique IDs

    def __init__(self, datapath_in, datapath_out, datasize_in, datasize_out, memsize, flopcount, dt_fixed=False,
                 preq_tasks=set(), streaming_in=False, streaming_out=False, purge_data=0, description=""):
        """
        :param datapath_in: (source, destination) tuple defining how the task receives its input data
        :param datapath_out: (source, destination) tuple defining how the task sends its output data
        :param datasize_in: Data volume, in TB
        :param datasize_out: Data volume, in TB
        :param memsize: The amount of working memory required by this task
        :param flopcount: The number of floating point operations required by this task
        :param dt_fixed: If a numeric value, specifies a fixed amount of seconds during which this task must complete.
        :param preq_tasks: A collection of tasks that must complete before this task can start
        :param streaming_in: True iff the task can stream its input data instead of reading everything first
        :param streaming_out: True iff the task can stream its output data instead of writing everything first
        :param purge_data: The amount of memory which should be deallocated from the datapath_out source
        :param description: Optional description
        """
        assert datapath_in in SDPAttributes.datapath_speeds
        assert datapath_out in SDPAttributes.datapath_speeds
        if datapath_in is None:
            assert datasize_in == 0
        if datapath_out is None:
            assert datasize_out == 0

        if dt_fixed:
            # Not compulsory but something we assume; we need to update Scheduler.schedule() code if not always true
            assert (streaming_in and streaming_out)

        SDPTask.class_uid_generator += 1
        self.uid = SDPTask.class_uid_generator
        self.datapath_in = datapath_in
        self.datapath_out = datapath_out
        self.datasize_in  = datasize_in
        self.datasize_out = datasize_out
        self.memsize = memsize
        self.flopcount = flopcount
        self.dt_fixed = dt_fixed
        self.preq_tasks = set().union(preq_tasks)  # Prerequisite tasks that needs to complete before this one can start
        self.streaming_in = streaming_in
        self.streaming_out = streaming_out
        self.purge_data = purge_data
        self.description = description

    def __hash__(self):
        return hash(self.uid)  # Required for using Task objects in sets.

    def __eq__(self, other):
        if not isinstance(other, SDPTask):
            return False
        return (self.uid == other.uid)

    def __str__(self):
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
    # TODO : it would be better to logically split the Schedule, Scheduler and simulated SDP into separate classes.

    def __init__(self):
        self.task_list = []

        self.flops_deltas = {0: 0}  # maps wall clock times to SDP FLOPS allocation / deallocation (+/-) sizes
        self.memory_deltas = {0: 0}  # maps wall clock times to SDP working memory (RAM) allocation / deallocation (+/-) sizes
        self.hot_buffer_deltas = {0: 0}  # maps wall clock times to hot buffer allocation / deallocation (+/-) sizes
        self.cold_buffer_deltas = {0: 0}  # maps wall clock times to cold buffer allocation / deallocation (+/-) sizes
        self.preserve_deltas = {0: 0}  # maps wall clock times to long term preserve allocation sizes (presumably only +)
        self.tasks_to_timestamps = {}  # Maps tasks to completion times
        self.timestamps_to_tasks = {}  # Maps completion times to tasks lists

        self.performance_dict = None

    def set_performance_dictionary(self, performance_dictionary):
        assert isinstance(performance_dictionary, dict)
        if self.performance_dict is not None:
            warnings.warn("Performance Dictionary has already been assigned! Overwriting...")
        self.performance_dict = performance_dictionary

    def compute_performance_dictionary(self):
        """
        Builds a lookup table that maps each HPSO to its Subtasks, and Subtasks to to Performance Requirements.
        Useful when doing scheduling so that values do not need to be computed more than once
        :return: A dictionary of dictionaries. Maps each HPSO to a dictionary of subtasks that each maps to a performance value accordint to
        """
        assert self.performance_dict is None
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

        self.performance_dict = performance_dict
        print('done')
        return performance_dict

    def task_letters_to_sdp_task_list(self, letter_sequence):
        """
        Converts a list of task letters into a list of SDPTask objects
        :param letter_sequence : a sequence of HPSOs, defined by Rosie's lettering scheme ('A'..'G' )
        """
        if self.performance_dict is None:
            print("Performance Dictionary has not yet been initialized. Initializing now.")
            self.compute_performance_dictionary()
        performance_dict = self.performance_dict

        tasks = []  # the list of task objects
        prev_ingest_task = None

        for task_letter in letter_sequence:
            hpso = Definitions.hpso_lookup[task_letter]
            hpso_subtasks = HPSOs.hpso_subtasks[hpso]
            nr_subtasks = len(hpso_subtasks)

            assert nr_subtasks >= 2  # We assume that the tast as *at least* an Ingest and an RCal component
            if not (hpso_subtasks[0] in HPSOs.ingest_subtasks) and (hpso_subtasks[1] in HPSOs.rcal_subtasks):
                # this is assumed true for all HPSOs - hence raising an assertion error if not
                raise AssertionError("Assumption was wrong - some HPSO apparently doesn't involve Ingest + RCal")

            # -----------------------------
            # Ingest and Rcal cannot be separated; combined into a a single task object
            # -----------------------------
            datapath_in  = (SDPAttributes.ingest_buffer, SDPAttributes.working_mem)
            datapath_out = (SDPAttributes.working_mem, SDPAttributes.cold_buffer)
            dt_observe = performance_dict[hpso]['Tobs']
            datasize_in  = performance_dict[hpso]['Tobs'] * performance_dict[hpso][hpso_subtasks[0]]['ingestRate']
            datasize_out = datasize_in  # TODO: Add RCal's data output size to this when we know what it is

            memsize = (performance_dict[hpso][hpso_subtasks[0]]['cache'] +
                       performance_dict[hpso][hpso_subtasks[1]]['cache'])
            flopcount = performance_dict[hpso]['Tobs'] * (performance_dict[hpso][hpso_subtasks[0]]['compRate'] +
                                                          performance_dict[hpso][hpso_subtasks[1]]['compRate'])
            preq_task = set()
            if prev_ingest_task is not None:
                preq_task = {prev_ingest_task}  # previous ingest task needs to be complete before this one can start

            t = SDPTask(datapath_in, datapath_out, datasize_in, datasize_out, memsize, flopcount, dt_fixed=dt_observe,
                        streaming_in=True, streaming_out=True, preq_tasks=preq_task, description='Ingest+RCal')
            prev_ingest_task = t  # current (ingest+rcal) task remembered; subtasks may only start once this completes
            tasks.append(t)

            # -----------------------------
            # Creates transfer task to get the data from cold to the hot buffer, purging it in the cold buffer
            # -----------------------------
            memsize = 0    # We assume it takes no working memory to perform a transfer
            flopcount = 0  # We assume it takes no flops to perform a transfer
            datapath_out = (SDPAttributes.cold_buffer, SDPAttributes.hot_buffer)
            preq_task = {prev_ingest_task}
            t = SDPTask(None, datapath_out, 0, datasize_out, memsize, flopcount, preq_tasks=preq_task,
                        purge_data=datasize_out, description='Transfer cold-hot')
            hotbuffer_data_size = datasize_out
            prev_transfer_task = t
            tasks.append(t)

            # -----------------------------
            # We assume transfer from hot buffer to RAM is essentially without any delay (hence no transfer task).
            # In fact it is limited to 5 TB/s, but not clear how much data needs to be copied. Maybe not much.
            # Or maybe all of the hot buffer content, but not all at once before the processing can start, so it is
            # hard to know at what rate it will need to be copied. Is it used linearly as the process runs, and hence
            # a funtion of the execution time, which in turn depends on the amount of allocated FLOPS? Much work to be
            # done here to figure it out.
            # TODO: Investigate how this works.
            # -----------------------------

            # -----------------------------
            # Now handle the ICal subtask (if there is any) and DPrep subtasks (if there are any)
            # -----------------------------
            ical_task = None
            ical_dprep_tasks = set()

            for i in range(2, nr_subtasks):
                subtask = hpso_subtasks[i]

                datapath_in  = (SDPAttributes.hot_buffer, SDPAttributes.working_mem)
                datapath_out = (SDPAttributes.working_mem, SDPAttributes.hot_buffer)
                memsize   = performance_dict[hpso][subtask]['cache']
                flopcount = performance_dict[hpso]['Tobs'] * performance_dict[hpso][subtask]['compRate']
                datasize_in   = memsize  # TODO: is this correct? No idea.
                datasize_out  = memsize  # TODO: Replace by ICal & Dprep output data sizes (when we know what it is)
                t = SDPTask(datapath_in, datapath_out, datasize_in, datasize_out, memsize, flopcount,
                            description=str(subtask))  # TODO: are these streaming tasks? We assume not.
                hotbuffer_data_size += datasize_out

                if i == 2:
                    assert subtask in HPSOs.ical_subtasks  # Assumed that this is an ical subtask
                    t.preq_tasks.add(prev_transfer_task)  # Associated ingest task must complete before this one can start
                    ical_task = t  # Remember this task, as DPrep tasks depend on it
                elif subtask in HPSOs.dprep_subtasks:
                    assert ical_task is not None
                    t.preq_tasks.add(ical_task)
                else:
                    raise Exception("Unknown subtask type!")

                ical_dprep_tasks.add(t)
                tasks.append(t)

            # -----------------------------
            # A last task to flush out the results to Preservation, and then we're done
            # -----------------------------
            memsize = 0    # We assume it takes no working memory to perform a transfer
            flopcount = 0  # We assume it takes no flops to perform a transfer
            datapath_out = (SDPAttributes.hot_buffer, SDPAttributes.preserve)
            datasize_out = 0  # TODO: Replace by Image preservation data size when known
            t = SDPTask(None, datapath_out, 0, datasize_out, memsize, flopcount, preq_tasks=ical_dprep_tasks,
                        purge_data=hotbuffer_data_size, description='Transfer hot-preserve')
            tasks.append(t)

        self.task_list = tasks
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

    def schedule(self, sdp_flops=SDPAttributes.sdp_flops, assign_flops_fraction=0.5, assign_bandwidth_fraction=0.8,
                 minimum_flops=1.0, max_nr_iterations=9999, epsilon=Definitions.epsilon):
        """
        :param sdp_flops
        :param assign_flops_fraction fraction of the available FLOPS assigned to a task that has no fixed completion time
        :param assign_bandwidth_fraction
        :param minimum_flops minimum amount of petaflops to be assigned to a task that has no fixed completion time
        :param max_nr_iterations self-explanatory
        :param epsilon numerical sensitivity parameter
        """

        # First, assert that the SDP has enough capability to handle all of the tasks. Raise exception if not.
        tasks_to_be_scheduled = set()
        for task in self.task_list:
            assert isinstance(task, SDPTask)
            tasks_to_be_scheduled.add(task)
            if task.dt_fixed:
                if task.datapath_in is None:
                    minimum_read_time = 0
                else:
                    datapath_in_capacity = SDPAttributes.datapath_speeds[task.datapath_in]
                    minimum_read_time = task.datasize_in / datapath_in_capacity

                minimum_compute_time = task.flopcount / sdp_flops

                if task.datapath_out is None:
                    minimum_write_time = 0
                else:
                    datapath_out_capacity = SDPAttributes.datapath_speeds[task.datapath_out]
                    minimum_write_time = task.datasize_out / datapath_out_capacity

                assert task.datapath_out is not None  # TODO: This assertion may be false (if so just remove this line)

                minimum_task_time = minimum_compute_time
                if task.streaming_in:
                    minimum_task_time = max(minimum_task_time, minimum_read_time)
                else:
                    minimum_task_time += minimum_read_time
                if task.streaming_out:
                    minimum_task_time = max(minimum_task_time, minimum_write_time)
                else:
                    minimum_task_time += minimum_write_time

                if (minimum_task_time > task.dt_fixed):
                    raise AssertionError("SDP has too little capacity to handle the following real-time task:\n%s"
                                         % str(task))

        print('SDP seems to have sufficient FLOPS and data streaming capacity to handle real-time tasks.')

        # Delta sequences for the resources being used at each *node* in the data path.
        # A delta sequence maps timestamps to numerical changes
        # allocation / deallocation are designated by (+/-) values
        flops_deltas       = {0: 0}
        memory_deltas      = {0: 0}
        hot_buffer_deltas  = {0: 0}
        cold_buffer_deltas = {0: 0}
        preserve_deltas    = {0: 0}

        # Delta sequences for the resources used along each *edge* of the data path
        # A delta sequence maps timestamps to numerical changes
        # allocation / deallocation are designated by (+/-) values
        ingest_pipe_deltas      = {0: 0}
        ingest_cold_pipe_deltas = {0: 0}
        cold_hot_pipe_deltas    = {0: 0}
        hot_mem_pipe_delta      = {0: 0}
        mem_hot_pipe_delta      = {0: 0}
        hot_preserve_pipe_delta = {0: 0}

        datalocation_to_deltas_map = {SDPAttributes.cold_buffer : cold_buffer_deltas,
                                      SDPAttributes.hot_buffer : hot_buffer_deltas,
                                      SDPAttributes.preserve : preserve_deltas,
                                      SDPAttributes.working_mem : memory_deltas}


        datapath_to_deltas_map = {(SDPAttributes.ingest_buffer, SDPAttributes.working_mem) : ingest_pipe_deltas,
                                  (SDPAttributes.working_mem, SDPAttributes.cold_buffer): ingest_cold_pipe_deltas,
                                  (SDPAttributes.cold_buffer, SDPAttributes.hot_buffer): cold_hot_pipe_deltas,
                                  (SDPAttributes.hot_buffer, SDPAttributes.working_mem): hot_mem_pipe_delta,
                                  (SDPAttributes.working_mem, SDPAttributes.hot_buffer): mem_hot_pipe_delta,
                                  (SDPAttributes.hot_buffer, SDPAttributes.preserve): hot_preserve_pipe_delta,
                                  None : None}

        # -----------------------------------------------------------------------------------------------
        # Iteratively run through all tasks, scheduling them as we go along (whenever it becomes possible).
        # Repeat until all tasks are scheduled
        # -----------------------------------------------------------------------------------------------

        nr_iterations = 0
        task_completion_times = {}  # Maps tasks to their *completion* timestamps

        wall_clock = 0  # Simulated wall clock time (seconds)
        wall_clock_advance = False  # Advance the wall clock to this value if no tasks were schedulable

        # Outer loop (keeps repeating until all tasks scheduled)
        while len(task_completion_times) < len(tasks_to_be_scheduled):
            nr_iterations += 1
            nr_tasks_scheduled_this_iteration = 0
            if nr_iterations > max_nr_iterations:
                print("Warning: Maximum number of iterations exceeded; aborting!")
                warnings.warn('Maximum number of iterations exceeded; aborting!')
                break
            print("-= Starting iteration %d =-" % nr_iterations)

            # Loop across all tasks
            for task in tasks_to_be_scheduled:
                assert isinstance(task, SDPTask)

                # ------------------------
                # Check if task has already been scheduled.
                # ------------------------
                if task in task_completion_times:
                    continue  # Skipping this task (as it is already scheduled)

                # ------------------------
                # Check if an unfinished prerequisite task prevents this task from being scheduled at this time.
                # If so, continue to next task (we will revisit this task at a later iteration).
                # ------------------------
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

                # ------------------------
                # If we reach this point in the code (without hitting a 'continue' statement), the task
                # a) has not yet been scheduled and
                # b) does not have an unfinished prerequisite task preventing it from begin scheduled at this
                #    (wall clock) point in time.
                # Hence, we can schedule it, keeping track of the resources that are available and changed by the
                # task's execution.
                # All "_dt" values refer to time durations (in seconds), where "_t" refer to wall clock time (in sec)
                # ------------------------

                # Obtain the relevant the input and output pipeline bandwidths, and their delta histories
                bandwidth_in_full = SDPAttributes.datapath_speeds[task.datapath_in]
                bandwidth_out_full = SDPAttributes.datapath_speeds[task.datapath_out]
                datapath_in_deltas = datapath_to_deltas_map[task.datapath_in]
                datapath_out_deltas = datapath_to_deltas_map[task.datapath_out]

                if datapath_in_deltas is None:
                    assert task.datapath_in == None
                    assert task.datasize_in == 0
                    bandwidth_in_available = 0
                    read_dt = 0
                else:
                    bandwidth_in_available = bandwidth_in_full - Scheduler.sum_deltas(datapath_in_deltas, wall_clock,
                                                                                      value_max=bandwidth_in_full)
                    read_dt = task.datasize_in / (bandwidth_in_available * assign_bandwidth_fraction)

                compute_dt = None
                write_dt = None

                start_read_t = wall_clock
                start_compute_t = start_read_t + read_dt
                start_write_t = None

                if task.streaming_in:
                    start_compute_t = start_read_t

                # The main constraint in starting this task may be that there aren't enough FLOPS available at current
                # wallclock time. If so, we delay the start if the task's computation until enough FLOPS are free.
                flops_assigned = minimum_flops
                if task.dt_fixed:
                    flops_assigned = task.flopcount / task.dt_fixed
                delay = start_compute_t - Scheduler.find_suitable_time(flops_deltas, start_compute_t,
                                                                       value_max=(sdp_flops - flops_assigned),
                                                                       eps=epsilon)
                if delay > 0:
                    #TODO remove print statement
                    print("Needed to delay task %d by %g sec due to resource availability" % (task.uid, delay))
                    # We can now increase the amount of time taken reading the input, thereby lowering the demand
                    # on the input pipeline
                    read_dt += delay
                    start_compute_t += delay

                # Now that FLOPS avalaibility has been confirmed we assume that there is enough of other
                # (e.g. bandwidth) resources available for all tasks to happen at this chosen time point.
                # If not, we have a scheduling problem which we don't yet have a solution for.
                # TODO: check that the pipelines have enough capacity to handle the task with the current timeline

                flops_available = sdp_flops - Scheduler.sum_deltas(flops_deltas, start_compute_t, value_max=sdp_flops)
                assert flops_available >= flops_assigned

                if task.dt_fixed:
                    assert (task.streaming_in and task.streaming_out)  # Reading, computing and writing = simultaneous
                    start_read_t = start_compute_t
                    start_write_t = start_compute_t
                    read_dt = task.dt_fixed
                    compute_dt = task.dt_fixed
                    write_dt = task.dt_fixed
                else:
                    # Assign the timings that will be used, based on whether streaming is used.
                    compute_dt = task.flopcount / (flops_available * assign_flops_fraction)

                    if task.streaming_in:  # streaming input and computation happen simultaneously in this case
                        compute_dt = max(compute_dt, read_dt)  # to accommodate the slowest of the two tasks
                        read_dt = compute_dt

                    start_write_t = start_compute_t + compute_dt
                    if task.streaming_out:
                        start_write_t = start_compute_t

                    bandwidth_out_available = bandwidth_out_full - Scheduler.sum_deltas(datapath_out_deltas,
                                                                                        start_write_t,
                                                                                        value_max=bandwidth_out_full)
                    write_dt = task.datasize_out / (bandwidth_out_available * assign_bandwidth_fraction)
                    if task.streaming_out:
                        write_dt   = max(compute_dt, write_dt)
                        compute_dt = write_dt
                        if task.streaming_in:
                            read_dt = write_dt

                task_completion_time = start_write_t + write_dt
                task.set_param("t_end", task_completion_time)

                # ---------------------------------------------------------------------------
                # Now we have the timings for reading, computing and writing. All that remains is to add their effects
                # to the relevant delta sequences
                # ---------------------------------------------------------------------------


                # Assign the flops and bandwidths actually used (assuming they're constant during execution)
                if read_dt == 0:
                    assert task.datasize_in == 0
                    bandwidth_in_used = 0
                else:
                    bandwidth_in_used = task.datasize_in / read_dt

                if compute_dt == 0:
                    assert task.flopcount == 0
                    flops_used = 0
                else:
                    flops_used = task.flopcount / compute_dt

                if write_dt == 0:
                    assert task.datasize_out == 0
                    bandwidth_out_used = 0
                else:
                    bandwidth_out_used = task.datasize_out / write_dt

                Scheduler.add_delta(memory_deltas, task.memsize, start_read_t, task_completion_time, value_min=0)
                if datapath_in_deltas is not None:
                    Scheduler.add_delta(datapath_in_deltas, bandwidth_in_used, start_read_t, (start_read_t + read_dt),
                                        value_min=0, value_max=bandwidth_in_full)
                Scheduler.add_delta(flops_deltas, flops_used, start_compute_t, (start_compute_t + compute_dt),
                                    value_min=0, value_max=sdp_flops)
                Scheduler.add_delta(datapath_out_deltas, bandwidth_out_used, start_write_t, task_completion_time,
                                    value_min=0, value_max=bandwidth_out_full)

                # Assign delta to the data target (preallocate full data size at start of write)
                deltas = datalocation_to_deltas_map[task.datapath_out[1]]
                Scheduler.add_delta(deltas, task.datasize_out, start_write_t, value_min=0)

                # Purge Data at the end of the whole thing, if so specified
                if task.purge_data > 0:
                    print("Purging %g from %s" % (task.purge_data, task.datapath_out[0]))  # TODO remove
                    deltas = datalocation_to_deltas_map[task.datapath_out[0]]
                    Scheduler.add_delta(deltas, -task.purge_data, task_completion_time, value_min=0)

                # Add this task to the 'task_completion_times' and 'timestamps_to_tasks' mappings
                task_completion_times[task] = task_completion_time
                print('* Scheduled Task %d at wall clock = %g sec. Ends at t=%g sec. ' %
                      (task.uid, start_read_t, task_completion_time))
                nr_tasks_scheduled_this_iteration += 1

            print('Number of scheduled tasks after %d iterations is %d out of %d' % (nr_iterations,
                                                                                     len(task_completion_times),
                                                                                     len(tasks_to_be_scheduled)))

            if nr_tasks_scheduled_this_iteration == 0:
                if wall_clock_advance:
                    print("-> Advancing wall clock to %g sec." % wall_clock_advance)
                    wall_clock = wall_clock_advance
                    wall_clock_advance = False
                else:
                    print("Warning! No tasks scheduled, and wall clock not advanced!")

        print('Done!')

        self.flops_deltas = flops_deltas
        self.memory_deltas = memory_deltas
        self.cold_buffer_deltas = cold_buffer_deltas
        self.hot_buffer_deltas = hot_buffer_deltas
        self.preserve_deltas = preserve_deltas
        self.tasks_to_timestamps = task_completion_times

        self.ingest_pipe_deltas = ingest_pipe_deltas
        self.ingest_cold_pipe_deltas = ingest_cold_pipe_deltas
        self.cold_hot_pipe_deltas = cold_hot_pipe_deltas
        self.hot_mem_pipe_delta = cold_hot_pipe_deltas
        self.mem_hot_pipe_delta = mem_hot_pipe_delta
        self.hot_preserve_pipe_delta = hot_preserve_pipe_delta