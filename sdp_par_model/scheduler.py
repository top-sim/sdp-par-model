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
from .parameters.definitions import Pipelines, HPSOs
from .config import PipelineConfig
from . import reports  # PyCharm doesn't like this import statement but it is correct
import warnings
import bisect
import numpy as np

class Definitions:
    """
    Definitions that are used in the SDP task scheduler
    """
    epsilon = 1e-12  # Used for numerical stability (rounding errors)

    # Needs some refactoring methinks; idea would be to specify HPSOs instead of "letters".
    hpso_lookup = {'A': HPSOs.hpso01,
                   'B': HPSOs.hpso04c,
                   'C': HPSOs.hpso13,
                   'D': HPSOs.hpso14,
                   'E': HPSOs.hpso15,
                   'F': HPSOs.hpso27and33,
                   'G': HPSOs.hpso37c}

    # The is a subset of RESULT_MAP in reports.py -- It defines values we wish to calculate for scheduling purposes.
    # Units are standardised as TB and PFlop for consistency between different calculations
    #               Title                      Unit           Default? Sum?             Expression
    perf_reslt_map =[('Observation time',        'sec',        False, False, lambda tp: tp.Tobs),
                     ('Total buffer ingest rate','TeraBytes/s', True, False, lambda tp: tp.Rvis_ingest*tp.Nbeam*tp.Npp*tp.Mvis / c.tera),
                     ('Visibility I/O Rate',     'TeraBytes/s', True, True,  lambda tp: tp.Rio / c.tera),
                     ('Total Compute Rate',      'PetaFLOP/s',  True, True,  lambda tp: tp.Rflop / c.peta),
                     ('Visibility Buffer',       'TeraBytes',   True, True,  lambda tp: tp.Mbuf_vis / c.tera),
                     ('Working (cache) memory',  'TeraBytes',   True, True,  lambda tp: tp.Mw_cache / c.tera,),
                     ('Output Size',             'TeraBytes',   True, True,  lambda tp: tp.Mout /c.tera),
                     ('Pointing Time',            'sec',       False, False, lambda tp: tp.Tpoint),
                     ('Total Time',               'sec',       False, False, lambda tp: tp.Texp),
                     ('Image cube size',          'TB',         True, False, lambda tp: tp.Mimage_cube / c.tera),
                     ('Calibration output',       'TB',         True, False, lambda tp: tp.Mcal_out / c.tera),
                     ]

class SDPAttr:
    """
    Lists attributes of the SDP; used for lookup
    """
    ingest_buffer = "ingestbuffer"
    working_mem   = "working_mem"
    cold_buffer   = "coldbuffer"
    hot_buffer    = "hotbuffer"
    preserve      = "preserve"

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
        :param purge_data: The amount of data which is deallocated from the *datapath_out* *source*
        :param description: Optional description
        :type preq_tasks set
        """
        assert datapath_in in SDPAttr.datapath_speeds
        assert datapath_out in SDPAttr.datapath_speeds
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
                    #print('reassigning value for %s = %s' % (param_name, str(value)))  # TODO remove
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

class SDPSchedule:
    """
    Represents an SDP Schedule
    """
    def __init__(self):
        self.task_list = []

        self.task_completion_times = {}  # Maps tasks to completion times
        # self.timestamps_to_tasks = {}  # Maps completion times to tasks lists  #TODO Not needed; remove.

        # Delta sequences for the resources being used at each *node* in the data path.
        # A delta sequence maps timestamps to numerical changes
        # allocation / deallocation are designated by (+/-) values
        self.flops_deltas       = {0: 0}
        self.memory_deltas      = {0: 0}
        self.hot_buffer_deltas  = {0: 0}
        self.cold_buffer_deltas = {0: 0}
        self.preserve_deltas    = {0: 0}

        # Delta sequences for the resources used along each *edge* of the data path
        # A delta sequence maps timestamps to numerical changes
        # allocation / deallocation are designated by (+/-) values
        self.ingest_pipe_deltas      = {0: 0}
        self.mem_cold_pipe_deltas = {0: 0}
        self.cold_hot_pipe_deltas    = {0: 0}
        self.cold_mem_pipe_deltas    = {0: 0}
        self.mem_hot_pipe_delta      = {0: 0}
        self.hot_cold_pipe_deltas    = {0: 0}
        self.hot_mem_pipe_delta      = {0: 0}
        self.hot_preserve_pipe_delta = {0: 0}


class Scheduler:
    # TODO : it may be better to logically split the Scheduler and 'SDP Simulation' into separate classes, where the
    # latter can be an instantiated object. For now, we treat the Scheduler as a class with static methods only

    @staticmethod
    def compute_performance_dictionary():
        """
        Builds a lookup table that maps each HPSO to its tasks, and each Task to its Performance Requirements.
        Useful when doing scheduling so that values do not need to be computed more than once
        :return: A dictionary of dictionaries. Each HPSO -> dictionary of Tasks -> performance values
        """
        performance_dict = {}

        # As a test we loop over all HPSOs we wish to handle, computing results for each
        for hpso in HPSOs.available_hpsos:
            print('*** Computing performance reqs for HPSO %s ***\n' % hpso)
            if not hpso in performance_dict:
                performance_dict[hpso] = {}

            assert hpso in HPSOs.hpso_pipelines
            for pipeline in HPSOs.hpso_pipelines[hpso]:
                if not pipeline in performance_dict[hpso]:
                    performance_dict[hpso][pipeline] = {}

                cfg = PipelineConfig(hpso=hpso, hpso_pipe=pipeline)
                (valid, msgs) = cfg.is_valid()
                if not valid:
                    for msg in msgs:
                        print(msg)
                    raise AssertionError("Invalid config")

                results = reports._compute_results(cfg, Definitions.perf_reslt_map)  # TODO - refactor this method's parameter sequence

                # The contents of the results array are determined by Definitions.perf_reslt_map. Refer for details.
                performance_dict[hpso][pipeline]['ingestRate'] = results[1]
                performance_dict[hpso][pipeline]['visRate']    = results[2]
                performance_dict[hpso][pipeline]['compRate']   = results[3]
                performance_dict[hpso][pipeline]['visBuf']     = results[4]
                try:
                    memsize = float(results[5])
                except:
                    memsize = 0
                performance_dict[hpso][pipeline]['memSize']    = memsize
                performance_dict[hpso][pipeline]['outputSize'] = results[6]

                # Observation, Pointing & Total times (tObs, tPoint, tTotal) are HPSO attributes instead of
                # task attributes. Assign them to the HPSO instead of to the (sub)task
                # Although they are computed anew for each task, it should return the same value every time
                if not 'tObs' in performance_dict[hpso]:
                    performance_dict[hpso]['tObs'] = results[0]
                    print('Observation time\t= %g sec' % performance_dict[hpso]['tObs'])
                else:
                    assert performance_dict[hpso]['tObs'] == results[0]

                if not 'tPoint' in performance_dict[hpso]:
                    performance_dict[hpso]['tPoint'] = results[7]
                    print('Pointing time\t= %g sec' % performance_dict[hpso]['tPoint'])
                else:
                    assert performance_dict[hpso]['tPoint'] == results[7]

                try:
                    t_total = float(results[8])
                except:
                    t_total = 0
                if not 'tTotal' in performance_dict[hpso]:
                    performance_dict[hpso]['tTotal'] = t_total
                    print('Total time\t= %g sec' % performance_dict[hpso]['tTotal'])
                else:
                    assert performance_dict[hpso]['tTotal'] == t_total

                performance_dict[hpso][pipeline]['imCubeSize']  = results[9]
                performance_dict[hpso][pipeline]['calDataSize'] = results[10]

                print('\ntask -> %s' % pipeline)

                print('Compute Rate\t\t= %g PetaFLOP/s'   % performance_dict[hpso][pipeline]['compRate'])
                print('Buffer ingest rate\t= %g TB/s'     % performance_dict[hpso][pipeline]['ingestRate'])
                print('Visibility IO rate\t= %g TB/s'     % performance_dict[hpso][pipeline]['visRate'])
                print('Visibility Buffer\t= %g TB'        % performance_dict[hpso][pipeline]['visBuf'])
                print('Working Memory \t\t= %g TB'        % performance_dict[hpso][pipeline]['memSize'])
                print('Data output size \t= %g TB'        % performance_dict[hpso][pipeline]['outputSize'])
                print('Image Cube output size \t= %g TB'  % performance_dict[hpso][pipeline]['imCubeSize'])
                print('Calibration data output = %g TB'   % performance_dict[hpso][pipeline]['calDataSize'])
                print()

        print('Done with building performance dictionary.')
        return performance_dict

    @staticmethod
    def hpso_letters_to_hpsos(letter_sequence):
        """
        Translates a 'HPSO letter sequence' to a sequence of HPSO objects that can be used more logically
        This is an outdated way of doing things; can probably be removed soon.
        :param letter_sequence:
        :return:
        """
        hpso_list = []
        for task_letter in letter_sequence:
            hpso = Definitions.hpso_lookup[task_letter]
            hpso_list.append(hpso)

        return hpso_list

    @staticmethod
    def generate_sequence(hpso_set, performance_dict, dt_block, dt_seq, allow_short_tobs=False):
        """
        Generate sequence of observations from a performance dictionary.
        Based on and refactored from Mark Ashdown's code in scheduling.py. Uses existing data Scheduler data structures
        instead of the text files and custom data structures he used.
        :param hpso_set: the set of HPSOs to include in the sequence
        :param performance_dict: Performance dictionary that maps all HPSOs to tasks to requirements and values
        :param dt_block: the duration of a single scheduling block in seconds (or max length if allow_short_tobs == True)
        :param dt_seq: duration floor of the complete schedule sequence, in seconds
        :param allow_short_tobs: allows short observations for suitable projects (default False)
        """

        hpso_list = sorted(list(hpso_set))  # Indexed array so that we can assign probailities by index
        nr_hpsos = len(hpso_list)
        dt_exp_list       = np.zeros(nr_hpsos)
        dt_obs_list_model = np.zeros(nr_hpsos)
        dt_point_list     = np.zeros(nr_hpsos)
        hpso_prob_list    = np.zeros(nr_hpsos)

        hpso_to_index = {}
        for i in range(nr_hpsos):
            hpso = hpso_list[i]
            hpso_to_index[hpso]  = i
            dt_exp_list[i]       = performance_dict[hpso]['tTotal']
            dt_obs_list_model[i] = performance_dict[hpso]['tObs']
            dt_point_list[i]     = performance_dict[hpso]['tPoint']

        # We build up a random sequence of hpso observations, based on their statistical likelihood of occurring,
        # which we assume is based on how long they take to execute (?)

        seq_hpsos = []  # Sequence of HPSOs
        seq_tobs  = []  # Sequence of observation times corresponding to HPSOs (same array length)

        if allow_short_tobs:
            # This case allows short observations for suitable HPSOs:
            # if an HPSO has tpoint less than tsched, then its observations will be of length tpoint,
            # otherwise they will be of length tsched.
            dt_obs_list = np.where(dt_point_list < dt_block, dt_point_list, dt_block)
            if max(dt_obs_list < dt_obs_list_model):  # True if any element is true
                warnings.warn("At least one HPSO's 'short observation time' is shorter than performance model's 'Observation Time'!")

            # Probability for each HPSO, normalized to sum to 1.0
            hpso_prob = dt_exp_list / dt_obs_list
            hpso_prob /= np.sum(hpso_prob)

            # Iteratively choose an HPSO until the required elapsed-time is reached. Slower than Mark's vectorized code,
            # but still fast to compute. Avoids possible bias against 'unlucky' draws containing many short observations
            ttot = 0.0
            while ttot <= dt_seq:
                hpso = np.random.choice(hpso_list, p=hpso_prob)
                i = hpso_to_index[hpso]
                tobs = dt_obs_list[i]
                ttot += tobs

                seq_hpsos.append(hpso)
                seq_tobs.append(tobs)
        else:
            # This case generates scheduling blocks which are all of
            # length tsched.

            # Probability for each project.
            hpso_prob_list = dt_exp_list / np.sum(dt_exp_list)

            # Calculate number of scheduling blocks.

            nsched = np.ceil(dt_seq / dt_block).astype(int)

            # Create a random sequence of HPSOs. Each HPSO has a constant t_obs = dt_block.

            seq_hpsos = np.random.choice(hpso_list, size=nsched, p=hpso_prob_list)
            seq_tobs = np.ones(nsched) * dt_block

        return (seq_hpsos, seq_tobs)

    @staticmethod
    def hpsos_to_sdp_task_list(hpso_list, performance_dict, tobs_list=None, keep_data_in_coldbuf=False):
        """
        Converts a list of HPSO scheduling block letters into a list of SDPTask objects
        :param hpso_list : a list of HPSOs that need to be executed in sequence
        :param performance_dict : The performance dictionary supplying the costs and requirements of performing a task
        :param tobs_list: (Optional) List of obervation times to use for each hpso, instead of defaults
        :param keep_data_in_coldbuf : Default false. If true, a copy of visibility data is kept in cold buffer until processing completes
        """
        tasks = []  # the list of SDPTask objects
        prev_ingest_task = None

        nr_hpsos = len(hpso_list)
        if tobs_list is not None:
            assert len(tobs_list) == nr_hpsos

        for i in range(nr_hpsos):
            hpso = hpso_list[i]
            if tobs_list is not None:
                dt_obs = tobs_list[i]
            else:
                dt_obs = performance_dict[hpso]['tObs']

            hpso_pipelines = HPSOs.hpso_pipelines[hpso]

            assert len(hpso_pipelines) >= 2  # We assume that the hpso has *at least* an Ingest and an RCal task
            if not (hpso_pipelines[0] == Pipelines.Ingest) and (hpso_pipelines[1] == Pipelines.RCAL):
                # this is assumed true for all HPSOs - hence raising an assertion error if not
                raise AssertionError("Assumption was wrong - some HPSO apparently doesn't involve Ingest + RCal")

            # -----------------------------
            # Ingest and Rcal cannot be separated; combined into a a single task object
            # -----------------------------
            # TODO: does ingest need allocation of SDP working memory? i.e. is the data path below correct?
            datapath_in  = (SDPAttr.ingest_buffer, SDPAttr.working_mem)
            datapath_out = (SDPAttr.working_mem, SDPAttr.cold_buffer)
            memsize = performance_dict[hpso][hpso_pipelines[0]]['memSize'] + performance_dict[hpso][hpso_pipelines[1]]['memSize']
            dt_observe = dt_obs
            datasize_in  = dt_obs * performance_dict[hpso][hpso_pipelines[0]]['ingestRate']  # TeraBytes
            datasize_out = datasize_in + performance_dict[hpso][hpso_pipelines[0]]['outputSize'] + \
                                         performance_dict[hpso][hpso_pipelines[1]]['outputSize'] + \
                                         performance_dict[hpso][hpso_pipelines[0]]['calDataSize'] + \
                                         performance_dict[hpso][hpso_pipelines[1]]['calDataSize']
            datasize_in_coldbuf = datasize_out  # we use this later
            flopcount = dt_obs * (performance_dict[hpso][hpso_pipelines[0]]['compRate'] +
                                  performance_dict[hpso][hpso_pipelines[1]]['compRate'])
            if prev_ingest_task is not None:
                preq_task = {prev_ingest_task}  # previous ingest task needs to be complete before this one can start
            else:
                preq_task = set()  # no prerequisite tasks

            task = SDPTask(datapath_in, datapath_out, datasize_in, datasize_out, memsize, flopcount, dt_fixed=dt_observe,
                        streaming_in=True, streaming_out=True, preq_tasks=preq_task, description='Ingest+RCal')
            prev_ingest_task = task  # current (ingest+rcal) task remembered; subsequent tasks may only start once this completes
            tasks.append(task)

            # -----------------------------
            # Creates transfer task to get the data from cold to the hot buffer.
            # If "keep_data_in_coldbuf" is true, purging of the data from coldbuf is delayed until after processing
            # -----------------------------
            memsize = 0    # We assume it takes no working memory to perform a transfer
            flopcount = 0  # We assume it takes no flops to perform a transfer
            datapath_out = (SDPAttr.cold_buffer, SDPAttr.hot_buffer)
            if keep_data_in_coldbuf:
                data_to_purge = 0
            else:
                data_to_purge = datasize_in_coldbuf
            task = SDPTask(None, datapath_out, 0, datasize_out, memsize, flopcount, preq_tasks={prev_ingest_task},
                           purge_data=data_to_purge, description='Transfer visibility data cold-hot (keep copy = %s)' %
                                                                 keep_data_in_coldbuf)
            datasize_in_hotbuf = datasize_out
            prev_transfer_task = task
            tasks.append(task)

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
            # Now handle the ICal Task (if there is any) and DPrep Tasks (if there are any)
            # -----------------------------
            ical_task = None
            ical_dprep_tasks = set()

            imageDataSize = 0  # In TB

            for i in range(2, len(hpso_pipelines)):
                task_label = hpso_pipelines[i]

                datapath_in  = (SDPAttr.hot_buffer, SDPAttr.working_mem)
                datapath_out = (SDPAttr.working_mem, SDPAttr.hot_buffer)
                memsize   = performance_dict[hpso][task_label]['memSize']
                flopcount = dt_obs * performance_dict[hpso][task_label]['compRate']
                datasize_in   = performance_dict[hpso][task_label]['visBuf']
                datasize_out  = performance_dict[hpso][task_label]['outputSize']
                task = SDPTask(datapath_in, datapath_out, datasize_in, datasize_out, memsize, flopcount,
                               description=str(task_label))  # TODO: are these streaming tasks? We assume not.
                datasize_in_hotbuf += datasize_out

                if i == 2:
                    assert task_label == Pipelines.ICAL  # Assumed that this is an ical task
                    task.preq_tasks.add(prev_transfer_task)  # Associated ingest task must complete before this one can start
                    ical_task = task  # Remember this task, as DPrep tasks depend on it
                elif task_label[0:5] == 'DPrep':
                    assert ical_task is not None
                    imageDataSize += performance_dict[hpso][task_label]['imCubeSize']
                    task.preq_tasks.add(ical_task)
                else:
                    raise Exception("Unknown task type!")

                ical_dprep_tasks.add(task)
                tasks.append(task)

            # -----------------------------
            # A last task to flush out the results to Preservation, and then we're done
            # If a copy of data was kept in the cold buffer, this also creates a task to delete that data.
            # -----------------------------
            memsize = 0    # We assume it takes no working memory to perform a transfer
            flopcount = 0  # We assume it takes no flops to perform a transfer
            datapath_out = (SDPAttr.hot_buffer, SDPAttr.preserve)
            datasize_out = imageDataSize
            if len(ical_dprep_tasks) > 0:
                preq_tasks = ical_dprep_tasks
            else:
                preq_tasks = {prev_transfer_task}
            task = SDPTask(None, datapath_out, 0, datasize_out, memsize, flopcount, preq_tasks=preq_tasks,
                           purge_data=datasize_in_hotbuf, description='Transfer hot-preserve')
            preservation_task = task
            tasks.append(task)

            if keep_data_in_coldbuf:
                memsize = 0  # We assume it takes no working memory to delete data from the cold buffer
                flopcount = 0  # We assume it takes no flops to delete data from the cold buffer
                datapath_out = (SDPAttr.cold_buffer, SDPAttr.hot_buffer)  # The target doesn't matter because data size is zero, but path must be valid
                if len(ical_dprep_tasks) > 0:
                    preq_tasks = ical_dprep_tasks
                else:
                    preq_tasks = {prev_transfer_task}
                task = SDPTask(None, datapath_out, 0, 0, memsize, flopcount, preq_tasks={preservation_task},
                              purge_data=datasize_in_coldbuf, description='Delete visibility data in cold buffer')
                tasks.append(task)

        return tasks

    @staticmethod
    def sum_deltas(deltas_history, timepoint, sorted_delta_keys=None, val_min=None, val_max=None, eps=1e-15):
        """
        Sums all the deltas chronologically up to the timestamp t
        @param deltas_history  : a dictionary that maps wall clock timestamps to a resource's value-changes
        @param timepoint       : The time until which the values should be summed
        @param sorted_delta_keys : Optional sorted timestamps; prevents re-sorting the timestamps for efficiency
        @param val_min         : Lowest allowable value for the resource's balance. Default zero.
        @param val_max         : Higest allowable value for the resource's balance. Default None.
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
            if (((val_min is not None) and (value_at_t + eps < val_min)) or
                    ((val_max is not None) and (value_at_t - eps > val_max))):
                warnings.warn("Sum of deltas leads to value of %g at time %g sec. Outside imposed bounds of "
                               "[%s,%s] " % (value_at_t, timestamps_sorted[i], val_min, val_max))

        return value_at_t

    @staticmethod
    def find_suitable_time(timestamp_deltas, proposed_timepoint, sorted_delta_keys=None, val_min=None, val_max=None,
                           eps=1e-15):
        """
        Finds the earliest time >= timepoint when the sum of deltas is between value_min and value_max (if defined)
        @param timestamp_deltas : dictionary that maps wall clock timestamps to a resource's value-changes
        @param proposed_timepoint : The earliest point in time for the insertion
        @param sorted_delta_keys  : Optional sorted timestamps; prevents re-sorting the timestamps for efficiency
        @param val_min            : Lowest allowable value of the resource's balance for insertion.
        @param val_max            : Higest allowable value of the resource's balance for insertion.
        @param eps                : Numerical rounding tolerance
        @return                   : The timestamp. None, if none found.
        """
        # First cover the trivial case where there is no requirement for a suitable value at t=timepoint.
        if (val_min is None) and (val_max is None):
            return proposed_timepoint

        timestamps_sorted = sorted_delta_keys
        if timestamps_sorted is None:
            timestamps_sorted = sorted(timestamp_deltas.keys())

        value_at_t = Scheduler.sum_deltas(timestamp_deltas, proposed_timepoint, timestamps_sorted)

        # Check whether the value at the supplied time-point is suitable.
        if not (((val_min is not None) and (value_at_t + eps < val_min)) or
                ((val_max is not None) and (value_at_t - eps > val_max))):
            return proposed_timepoint

        # Otherwise, we continue searching until we find a suitable timepoint
        start_at_index = bisect.bisect_left(timestamps_sorted, proposed_timepoint)
        if proposed_timepoint in timestamp_deltas:
            start_at_index += 1  # value at pos found by bisect already included in summation; increment start pos.
        assert start_at_index > 0

        for i in range(start_at_index, len(timestamps_sorted)):
            new_timepoint = timestamps_sorted[i]
            value_at_t += timestamp_deltas[new_timepoint]
            if not (((val_min is not None) and (value_at_t + eps < val_min)) or
                    ((val_max is not None) and (value_at_t - eps > val_max))):
                return new_timepoint

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
    def schedule(tasks, flops_cap, hotbuf_cap, coldbuf_cap, assign_flops_fraction=0.5, verbose=False,
                 assign_bw_fraction=0.8, minimum_flops_frac=0.05, minimum_bw_frac=0.1,
                 max_nr_iterations=9999, epsilon=Definitions.epsilon):
        """
        :param tasks = an iterable collection of SDPTask objects that need to be scheduled
        :param flops_cap the cap on the usable SDP flops (PetaFLOPS)
        :param hotbuf_cap the capacity of the Hot Buffer (PetaBytes)
        :param coldbuf_cap the capacity of the Cold Buffer (PetaBytes)
        :param assign_flops_fraction of avail. FLOPS assigned to a task that has no fixed completion time
        :param assign_bw_fraction of avail. bandwidth assigned to a task that has no fixed completion time
        :param minimum_flops_frac fraction of flops_cap that needs to be available to schedule a task
        :param minimum_bw_frac fraction of bandwidth that needs to be available to schedule a task
        :param max_nr_iterations self-explanatory
        :param epsilon numerical sensitivity parameter
        """

        # -----------------------------------------------------------------------------------------------
        # First, assert that the SDP has enough capability to handle all of the tasks. Raise exception if not.
        # -----------------------------------------------------------------------------------------------
        tasks_to_be_scheduled = set()
        for task in tasks:
            assert isinstance(task, SDPTask)
            tasks_to_be_scheduled.add(task)
            datapipe_in_cap = SDPAttr.datapath_speeds[task.datapath_in]    # TB/s
            datapipe_out_cap = SDPAttr.datapath_speeds[task.datapath_out]  # TB/s

            if task.dt_fixed:
                if task.datapath_in is None:
                    assert task.datasize_in == 0
                    minimum_read_time = 0
                else:
                    assert datapipe_in_cap > 0
                    minimum_read_time = task.datasize_in / datapipe_in_cap

                minimum_compute_time = task.flopcount / flops_cap

                if task.datapath_out is None:
                    assert task.datasize_out == 0
                    minimum_write_time = 0
                else:
                    assert datapipe_out_cap > 0
                    minimum_write_time = task.datasize_out / datapipe_out_cap

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

        if verbose:
            print('SDP seems to have sufficient FLOPS and Data Streaming capacity to handle any real-time tasks.')

        # -----------------------------------------------------------------------------------------------
        # Start building a Schedule, with associated "deltas" lists.
        # Eacb "deltas" list describes all changes that are made to a given resource's allocation.
        # A "deltas" list is equivalent to (or from) the time evolution of the remaining resource, but is more
        # convenient for achronological insertion or deletion of new deltas.
        # -----------------------------------------------------------------------------------------------

        schedule = SDPSchedule()

        datalocation_to_limits_map = {SDPAttr.cold_buffer : coldbuf_cap,
                                      SDPAttr.hot_buffer  : hotbuf_cap,
                                      SDPAttr.working_mem : float('inf'),  # infinite working memory
                                      SDPAttr.preserve    : float('inf'),  # infinite preservation memory
                                      None: None}

        datalocation_to_deltas_map = {SDPAttr.cold_buffer : schedule.cold_buffer_deltas,
                                      SDPAttr.hot_buffer  : schedule.hot_buffer_deltas,
                                      SDPAttr.working_mem : schedule.memory_deltas,
                                      SDPAttr.preserve    : schedule.preserve_deltas,
                                      None : None}

        # Note - not all conceivable pipes exist, e.g. no pipe from cold buffer to preserve.
        datapath_to_deltas_map = {(SDPAttr.ingest_buffer, SDPAttr.working_mem) : schedule.ingest_pipe_deltas,
                                  (SDPAttr.working_mem, SDPAttr.cold_buffer)   : schedule.mem_cold_pipe_deltas,
                                  (SDPAttr.working_mem, SDPAttr.hot_buffer)    : schedule.mem_hot_pipe_delta,
                                  (SDPAttr.cold_buffer, SDPAttr.hot_buffer)    : schedule.cold_hot_pipe_deltas,
                                  (SDPAttr.cold_buffer, SDPAttr.working_mem)   : schedule.cold_mem_pipe_deltas,
                                  (SDPAttr.hot_buffer, SDPAttr.working_mem)    : schedule.hot_mem_pipe_delta,
                                  (SDPAttr.hot_buffer, SDPAttr.cold_buffer)    : schedule.hot_cold_pipe_deltas,
                                  (SDPAttr.hot_buffer, SDPAttr.preserve)       : schedule.hot_preserve_pipe_delta,
                                  None : None}

        # -----------------------------------------------------------------------------------------------
        # Iteratively run through all tasks, scheduling them as we go along (whenever it becomes possible).
        # Repeat until all tasks are scheduled
        # -----------------------------------------------------------------------------------------------

        nr_iterations = 0
        schedule.task_completion_times = {}  # Maps tasks to their *completion* timestamps

        wall_clock = 0  # Simulated wall clock time (seconds)
        wall_clock_advance = False  # Advance the wall clock to this value if no tasks were schedulable

        # Outer loop (repeats until all tasks scheduled, or until max number of iterations reached)
        while len(schedule.task_completion_times) < len(tasks_to_be_scheduled):
            nr_iterations += 1
            nr_tasks_scheduled_this_iteration = 0
            if nr_iterations > max_nr_iterations:
                print("Warning: Maximum number of iterations exceeded; aborting!")
                warnings.warn('Maximum number of iterations exceeded; aborting!')
                break
            if verbose:
                print("-= Starting iteration %d =-" % nr_iterations)
                if nr_iterations == 7:
                    print('reached')  # TODO remove

            # Loop across all tasks
            for task in tasks_to_be_scheduled:
                assert isinstance(task, SDPTask)
                datapipe_in_cap = SDPAttr.datapath_speeds[task.datapath_in]
                datapipe_out_cap = SDPAttr.datapath_speeds[task.datapath_out]

                # ------------------------
                # Check if task has already been scheduled.
                # ------------------------
                if task in schedule.task_completion_times:
                    continue  # Skipping this task (as it is already scheduled)

                # ------------------------
                # Check if an unfinished prerequisite task prevents this task from being scheduled at this time.
                # If so, continue to next task (we will revisit this task at a later iteration).
                # ------------------------
                elif len(task.preq_tasks) > 0:
                    unscheduled_preq_task = False
                    task_wall_clock_advance = False
                    for preq_task in task.preq_tasks:
                        if preq_task not in schedule.task_completion_times:
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
                # If we reach this point in the code (without hitting a 'continue' statement), the current task
                # a) has not yet been scheduled and
                # b) does not have an unfinished prerequisite task preventing it from begin scheduled at this
                #    (wall clock) point in time.
                # Hence, we can schedule it, keeping track of the resources that are available and changed by the
                # task's execution.
                # All "_dt" values refer to time durations (in seconds), where "_t" refer to wall clock time (in sec)
                # ------------------------

                # Obtain the relevant the input and output pipeline bandwidths, and their delta histories
                min_req_bw_in = datapipe_in_cap * minimum_bw_frac
                datapipe_in_deltas = datapath_to_deltas_map[task.datapath_in]

                min_req_flops = flops_cap * minimum_flops_frac

                min_req_bw_out = datapipe_out_cap * minimum_bw_frac
                datapipe_out_deltas = datapath_to_deltas_map[task.datapath_out]

                # Obtain the output data target's size limit and delta history
                target_out_cap    = datalocation_to_limits_map[task.datapath_out[1]]
                target_out_deltas = datalocation_to_deltas_map[task.datapath_out[1]]
                assigned_bw_in  = 0  # to be overwritten
                assigned_bw_out = 0  # to be overwritten
                assigned_flops  = 0  # to be overwritten

                # -------
                # Find a suitable times to start reading data, computing data, and writing data
                # The earliest completion time is attempted. When this is known, associated timestamps are
                # adjusted retroactively to occur as late as possible to match this completion time. E.g.: if a lack
                # of available FLOPS force later completion, the start_read_t may be moved later, as long as it doesn't
                # cause additional delay to to completion
                # -------
                start_t = wall_clock
                found_suitable_schedule_time = False  # True iff a suitable schedule time has been found
                nr_suitable_start_time_iterations = 0

                if task.dt_fixed:
                    # Technically not a hard requirement, but we simplify our lives by assuming the following:
                    assert (task.streaming_in and task.streaming_out)  #TODO in future allow sequential fixed-time tasks
                    # Assign bandwidths and flops to the task. We do not take minimum rates into consideration here
                    # because the timespan of the task has primary importance
                    assigned_bw_in  = task.datasize_in / task.dt_fixed   # TeraBytes / sec
                    assigned_flops = task.flopcount / task.dt_fixed      # PetaFLOP  / sec
                    assigned_bw_out = task.datasize_out / task.dt_fixed  # TeraBytes / sec

                    while not found_suitable_schedule_time:
                        nr_suitable_start_time_iterations += 1
                        assert nr_suitable_start_time_iterations < 999  # prevent infinite loop; should never happen

                        start_read_t = Scheduler.find_suitable_time(datapipe_in_deltas, start_t,
                                                                    val_max=(datapipe_in_cap - assigned_bw_in)
                                                                    , eps=epsilon)

                        start_comp_t = Scheduler.find_suitable_time(schedule.flops_deltas, start_t,
                                                                       val_max=(flops_cap - assigned_flops),
                                                                       eps=epsilon)

                        start_writepipe_t = Scheduler.find_suitable_time(datapipe_out_deltas, start_t,
                                                                         val_max=(datapipe_out_cap - assigned_bw_out),
                                                                         eps=epsilon)

                        start_writedest_t = Scheduler.find_suitable_time(target_out_deltas, start_t,
                                                                         val_max=(target_out_cap - task.datasize_out),
                                                                         eps=epsilon)

                        if (start_read_t is None) or (start_comp_t is None) or (start_writepipe_t is None) or (start_writedest_t is None):
                            break
                        elif (start_read_t != start_comp_t) or (start_read_t != start_writepipe_t) \
                                or (start_read_t != start_writedest_t):
                            start_t = max(start_read_t, start_comp_t, start_writepipe_t, start_writedest_t)
                            continue  # and repeat another loop
                        else:
                            start_read_t = start_t
                            start_comp_t = start_t
                            start_write_t = start_t
                            read_dt = task.dt_fixed
                            comp_dt = task.dt_fixed
                            write_dt = task.dt_fixed
                            end_task_t = start_write_t + write_dt
                            found_suitable_schedule_time = True

                # All other cases: task does not have a fixed time slot and could, in principle, take any amount of
                # time as long as minimum bandwidth and flops rates are attained
                elif task.streaming_in and task.streaming_out:
                    start_read_t = 0
                    start_comp_t = 0
                    start_write_t = 0
                    read_dt = 0
                    comp_dt = 0
                    write_dt = 0
                    end_task_t = 0
                    raise Exception("Not Implemented")

                elif task.streaming_in:  # Streaming in, not out
                    start_read_t = 0
                    start_comp_t = 0
                    start_write_t = 0
                    read_dt = 0
                    comp_dt = 0
                    write_dt = 0
                    end_task_t = 0
                    raise Exception("Not Implemented")

                elif task.streaming_out:  # Streaming out, not in
                    start_read_t = 0
                    start_comp_t = 0
                    start_write_t = 0
                    read_dt = 0
                    comp_dt = 0
                    write_dt = 0
                    end_task_t = 0
                    raise Exception("Not Implemented")

                else:  # Neither streaming out, not streaming in
                    while not found_suitable_schedule_time:
                        nr_suitable_start_time_iterations += 1
                        assert nr_suitable_start_time_iterations < 999  # prevent infinite loop; should never happen

                        assigned_bw_in = 0
                        read_dt = 0
                        start_read_t = start_t
                        if (task.datapath_in is not None) and (task.datasize_in > 0):
                            start_read_t = Scheduler.find_suitable_time(datapipe_in_deltas, start_t,
                                                                        val_max=(datapipe_in_cap - min_req_bw_in),
                                                                        eps=epsilon)
                            if start_read_t is None:
                                break
                            bw_in_available = datapipe_in_cap - Scheduler.sum_deltas(datapipe_in_deltas, start_read_t,
                                                                                     val_max=datapipe_in_cap, eps=epsilon)

                            assigned_bw_in = max(bw_in_available * assign_bw_fraction, min_req_bw_in)
                            read_dt = task.datasize_in / assigned_bw_in

                        assigned_flops = 0
                        comp_dt = 0
                        start_comp_t = start_t + read_dt
                        if task.flopcount > 0:
                            start_comp_t = Scheduler.find_suitable_time(schedule.flops_deltas, start_comp_t,
                                                                        val_max=(flops_cap - min_req_flops),
                                                                        eps=epsilon)
                            if start_comp_t is None:
                                break
                            flops_available = flops_cap - Scheduler.sum_deltas(schedule.flops_deltas, start_comp_t,
                                                                               val_max=flops_cap, eps=epsilon)
                            assigned_flops = max(flops_available * assign_flops_fraction, min_req_flops)
                            comp_dt = task.flopcount / assigned_flops

                        assigned_bw_out = 0
                        write_dt = 0
                        start_write_t = start_comp_t + comp_dt
                        end_task_t = start_write_t

                        if (task.datapath_out is not None) and (task.datasize_out > 0):
                            start_writepipe_t = Scheduler.find_suitable_time(datapipe_out_deltas, start_write_t,
                                                                         val_max=(datapipe_out_cap - min_req_bw_out),
                                                                         eps=epsilon)
                            start_writedest_t = Scheduler.find_suitable_time(target_out_deltas, start_write_t,
                                                                         val_max=(target_out_cap - task.datasize_out),
                                                                         eps=epsilon)
                            if (start_writepipe_t is None) or (start_writedest_t is None):
                                break
                            start_write_t = max(start_writepipe_t, start_writedest_t)

                            bw_out_available = datapipe_out_cap - Scheduler.sum_deltas(datapipe_out_deltas, start_write_t,
                                                                                       val_max=datapipe_out_cap, eps=epsilon)
                            target_out_avail = target_out_cap - Scheduler.sum_deltas(target_out_deltas, start_write_t,
                                                                                     val_max=target_out_cap, eps=epsilon)
                            if (bw_out_available < min_req_bw_out) or (target_out_avail < task.datasize_out):
                                start_t += (start_write_t - min(start_writepipe_t, start_writedest_t))
                                continue
                            else:
                                assigned_bw_out = max(bw_out_available * assign_bw_fraction, min_req_bw_out)
                                write_dt = task.datasize_out / assigned_bw_out
                                end_task_t = start_write_t + write_dt
                                found_suitable_schedule_time = True
                        else:
                            found_suitable_schedule_time = True

                if not found_suitable_schedule_time:
                    if verbose:
                        print('-- > did not found suitable schedule time for task %d; postponing' % task.uid)
                    continue
                else:
                    if verbose:
                        print('-- > found suitable schedule after %d iterations' % nr_suitable_start_time_iterations)
                    task.set_param("t_end", end_task_t)

                # ---------------------------------------------------------------------------
                # Now we have the timings for reading, computing and writing. All that remains is to add their effects
                # to the relevant delta sequences
                # ---------------------------------------------------------------------------

                Scheduler.add_delta(schedule.memory_deltas, task.memsize, start_read_t, end_task_t, value_min=0)
                Scheduler.add_delta(schedule.flops_deltas, assigned_flops, start_comp_t, (start_comp_t + comp_dt),
                                    value_min=0, value_max=flops_cap)
                if task.datapath_in is not None:
                    Scheduler.add_delta(datapipe_in_deltas, assigned_bw_in, start_read_t, (start_read_t + read_dt),
                                        value_min=0, value_max=datapipe_in_cap)
                if task.datapath_out is not None:
                    Scheduler.add_delta(datapipe_out_deltas, assigned_bw_out, start_write_t, end_task_t,
                                        value_min=0, value_max=datapipe_out_cap)
                    # preallocate full data size at start of write
                    Scheduler.add_delta(target_out_deltas, task.datasize_out, start_write_t, value_min=0,
                                        value_max=target_out_cap)

                # Purge specified amount of data from the output pipe's source at the end of the task's completion
                if task.purge_data > 0:
                    Scheduler.add_delta(datalocation_to_deltas_map[task.datapath_out[0]], -task.purge_data,
                                        end_task_t, value_min=0)

                # Add this task to the 'task_completion_times' and 'timestamps_to_tasks' mappings
                schedule.task_completion_times[task] = end_task_t
                if verbose:
                    print('* Scheduled Task %d to start at wall clock = %g sec. Ends at t=%g sec. ' %
                          (task.uid, start_read_t, end_task_t))
                nr_tasks_scheduled_this_iteration += 1

            if verbose:
                print('After %d iterations -> %d out of %d tasks scheduled ' % (nr_iterations,
                                                                                len(schedule.task_completion_times),
                                                                                len(tasks_to_be_scheduled)))

            if nr_tasks_scheduled_this_iteration == 0:
                if wall_clock_advance:
                    if verbose:
                        print("-> Advancing wall clock to %g sec." % wall_clock_advance)
                    wall_clock = wall_clock_advance
                    wall_clock_advance = False
                elif verbose:
                    print("Warning! No tasks scheduled, and wall clock not advanced!")

        return schedule
