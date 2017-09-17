"""
A script for debugging using Pycharm -- better than doing live coding in the Jupyter Notebook
"""

# Imports
import sys
sys.path += ['..']
from sdp_par_model import reports as iapi
from sdp_par_model import evaluate as imp
from sdp_par_model.config import PipelineConfig
from sdp_par_model.parameters.definitions import *
from sdp_par_model.parameters.definitions import Constants as c
import numpy as np
import collections
import warnings

# Define useful structures and methods

class SDPTask:
    uid          = None  # Optional: unique ID; can be used for sequencing
    t_min_start  = None  # Earliest wall clock time that this task can / may start (in seconds)
    prec_task    = None  # Preceding task that needs to complete before this one can be executed
    t_fixed      = None  # fixed minimum duration of this task (e.g. for an observation)
    flopcount    = None  # Number of floating point operations required to complete this task
    data_in      = None  # Amount of data (in TB) that this task needs for input (usually read from the hot buffer)    
    data_out     = None  # Amount of data (in TB) that this task outputs (usually written to hot buffer)    

def add_delta(deltas, t, delta):
    """
    Adds a {t : delta} pair to a timestamped dictionary that maps timestamps to delta values.
    If the supplied t already maps to a value, the supplied delta is added
    """
    if t in deltas:
        warnings.warn('Timestamp entry already exists in the timeline')
        deltas[t] += delta
    else:
        deltas[t] = delta

if __name__ == '__main__':
    # Needs some refactoring methinks; idea would be to specify HPSOs instead of "letters".
    hpso_lookup = {'A' : HPSOs.hpso01,
                   #'B' : (HPSOs.hpso04c,),  # This one probably not properly defined yet
                   'C' : HPSOs.hpso13,
                   'D' : HPSOs.hpso14,
                   'E' : HPSOs.hpso15,
                   'F' : HPSOs.hpso27,
                   'G' : HPSOs.hpso37c}

    sorted(hpso_lookup.keys())
    task_letter = 'A'
    hpso = hpso_lookup[task_letter]

    # The following results map was copied from examples used by Peter Wortmann. It defines values we wish to calculate.
    #               Title                      Unit       Default? Sum?             Expression
    results_map = [('Total buffer ingest rate', 'TeraBytes/s', True, False,
                    lambda tp: tp.Rvis_ingest * tp.Nbeam * tp.Npp * tp.Mvis / c.tera),
                   ('Working (cache) memory', 'TeraBytes', True, True, lambda tp: tp.Mw_cache / c.tera,),
                   ('Visibility I/O Rate', 'TeraBytes/s', True, True, lambda tp: tp.Rio / c.tera,),
                   ('Total Compute Rate', 'PetaFLOP/s', True, True, lambda tp: tp.Rflop / c.peta,),
                   ('Comp Req Breakdown ->', 'PetaFLOP/s', True, True,
                    lambda tp: tp.get_products('Rflop', scale=c.peta),)]
    del results_map[4]  # We actually don't care about the breakdown for now; but it is useful to know how to get it

    print('*** Processing task type %s => %s ***\n' % (task_letter, hpso))

    for subtask in HPSOs.hpso_subtasks[hpso]:
        print('subtask -> %s' % subtask)
        cfg = PipelineConfig(hpso=hpso, hpso_subtask=subtask)

        (valid, msgs) = cfg.is_valid()
        if not valid:
            print("Invalid configuration!")
            for msg in msgs:
                print(msg)
            raise AssertionError("Invalid config")
        tp = cfg.calc_tel_params()
        results = iapi._compute_results(cfg, False, results_map)  #TODO - refactor this method's parameter sequence
        print('Buffer ingest rate\t= %g TB/s' % results[0])
        print('Cache memory\t= %g TB' % results[1])
        print('Visibility IO rate\t= %g TB/s' % results[2])
        print('Compute Rate\t= %g PetaFLOP/s' % results[3])
        print()

    print('done')