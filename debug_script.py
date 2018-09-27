"""
A script for debugging using Pycharm -- better than doing live coding in the Jupyter Notebook
"""

# Imports
import sys
import warnings
import os
import pickle

from sdp_par_model import reports as iapi
from sdp_par_model.config import PipelineConfig
from sdp_par_model.parameters.definitions import *
from sdp_par_model.parameters.definitions import Constants as c
from sdp_par_model.scheduler import Scheduler

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


def entry(title, val, unit='', doc=''):
    return (title, val, unit, doc)


def visibility_analysis(tp, Qspeed, Nnode, Npredict, Mvisnode):
    Mbuf = tp.Npp * tp.Rvis.eval_sum(tp.baseline_bins) * tp.Nbeam * tp.Mvis * tp.Tobs
    Rvis = tp.Mvis * tp.Nbeam * tp.Npp * tp.Rvis_ingest
    Nmajor = 1 + tp.Nmajortotal
    Rvisnode = Qspeed * (Rvis * Nmajor) / float(Nnode)
    Rpredictnode = Npredict * Rvisnode
    twork = Mvisnode * c.giga / (Rvisnode + Rpredictnode)
    Msnapnode = tp.Mvis * tp.Npp * tp.Rvis_ingest * tp.Tsnap / Nnode
    Cmemprice = 0.5  # €/GB

    return [
        entry('-- Visibilities --', None, None),
        entry('Q_speed = ', Qspeed, '', 'Compute time scale compared with ingest'),
        entry('t_major = ', tp.Tobs / Qspeed / Nmajor, 's', 'Average time we have to run one major loop'),
        entry('M_buf/node =', Mbuf / Nnode / c.giga, 'GB',
              'Visibility buffer size we need per node to hold the observation'),
        entry('R_vis/node = ', Rvisnode / c.giga, 'GB/s',
              'Minimum visibility buffer read rate to sustain to read visibilities once per major loop'),
        entry('M_snap/node = ', Msnapnode / c.giga, "GB", 'Size of snapshot data per node (without predict)'),
        entry('R_predict/node = ', Rpredictnode / c.giga, 'GB/s',
              'Rate we need to produce predicted visibilities at to match visibility buffer rate'),
        entry('M_vis/node =', Mvisnode, 'GB', 'Assumed memory we have per node for holding visibilities'),
        entry('t_work =', twork, 's', 'Time we can work on every visibility before we run out of memory'),
        entry('C_work =', Nnode * Cmemprice * (Rvisnode + Rpredictnode) / c.giga, '€/s',
              'Cost for extending the above number by one second')
    ]


def get_image_size(tp):
    Mfacet = tp.Mpx * tp.Npix_linear ** 2
    Mimage = Mfacet * tp.Nfacet ** 2
    return Mfacet, Mimage


def imaging_analysis(tp, Qspeed, Nnode, Nsubbands):
    Mfacet, Mimage = get_image_size(tp)

    # Calculating image size including facetting, and therefore overlap. I think this is more fair.
    Mimagenodemin = max(Nsubbands * tp.Npp * tp.Ntt * Mimage / Nnode, Mfacet)
    Rphrot = Qspeed * (tp.products.get(Products.PhaseRotationPredict, {}).get('Rflop', 0) + \
                       tp.products.get(Products.PhaseRotation, {}).get('Rflop', 0))

    return [
        entry('-- Imaging --', None, None),
        entry('N_subbands = ', Nsubbands, '', 'Minimum number of frequencies we *need* to image separately'),
        entry('N_pp N_tt M_image = ', tp.Npp * tp.Ntt * Mimage / c.tera, 'TB',
              'Continuum image/grid size - data a single visibility updates'),
        entry('N_facet = ', tp.Nfacet, '', 'How much we split the image into facets'),
        entry('R_phaserotate = ', Rphrot / c.peta, 'Pflop/s', 'Compute cost for facetting (imaging+predict)'),
        entry('M_image/node,max = ', tp.Npp * tp.Ntt * Mfacet / c.giga, 'GB',
              'Facet data per node, assuming no distribution in polarisation/Taylor terms'),
        entry('M_image/node,min = ', Mimagenodemin / c.giga, 'GB',
              'Facet data per node, assuming complete distribution in polarisation/Taylor terms')
    ]


def _get_distribution(tp, Nnode, Nsubbands, Ndist_pptt):
    Ndist_ft = max(1, float(Nnode / (Nsubbands * tp.Nfacet ** 2 * Ndist_pptt)))
    Nseq = Nsubbands * tp.Nfacet ** 2 * Ndist_pptt * Ndist_ft / Nnode
    return Ndist_ft, Nseq


def imaging_data_analysis(tp, Qspeed, Nnode, Nisland, Nsubbands, Ndist_pptt):
    Mfacet, Mimage = get_image_size(tp)
    Ndist_ft, Nseq = _get_distribution(tp, Nnode, Nsubbands, Ndist_pptt)

    Mimagenode = max(Ndist_ft * Nsubbands * tp.Npp * tp.Ntt * Mimage / Nnode / Nseq, Mfacet)

    Nmajor = 1 + tp.Nmajortotal
    Tsnapwork = tp.Tsnap / Qspeed / Nmajor / Nseq
    Rimagenode = Mimagenode / Tsnapwork
    Tobswork = tp.Tobs / Qspeed / Nmajor / Nseq
    Routnode = Mimagenode / Tobswork

    entries_data = [
        entry('-- Imaging Data --', None, None),
        entry('N_dist,pp/tt = ', Ndist_pptt, '', 'Distribution degree in polarisation / Taylor Terms'),
        entry('N_dist,f/t = ', Ndist_ft, '',
              'Remaining distribution degree in frequency / time to get enough parallelism'),
        entry('N_seq = ', Nseq, '', ''),
        entry('M_image/node = ', Mimagenode / c.giga, 'GB', 'Resulting facet data per node'),
        entry('t_snap,work = ', Tsnapwork, 's',
              'Time every node has to work on a snapshot (assuming no time distribution)'),
        entry('R_image/node = ', Rimagenode / c.giga, 'GB/s',
              'Data rate for "spilling" facets to storage every snapshot'),
        entry('R_out/node = ', Routnode / c.giga, 'GB/s', 'Data rate for "spilling" facets to storage every snapshot'),
    ]

    # Number of nodes visibility data needs to be distributed to. Frequency/time distribution
    # doess not need communication, as we can assume visibilities to have been appropriately
    # distributed beforehand
    Ncopy = max(1, Nseq * Nnode / Ndist_ft / Nsubbands)  # Nfacet**2 * Ndist_pptt
    Rfacet = Qspeed * float(tp.Rfacet_vis) / Nnode

    # Every node needs to send a facet visibility stream to Ncopy-1 nodes
    Rdistributenode = Qspeed * tp.Rfacet_vis / Nnode * (Ncopy - 1)

    # Every island node needs to send at least (Ncopy-Nisland) off-island nodes a facet
    # visibility stream. Stream emitted by an island is therefore Nisland times higher.
    Rdistributeisland = Qspeed * tp.Rfacet_vis / Nnode * max(0, Ncopy - Nisland) * Nisland

    return Mimagenode, Rdistributenode, Rdistributeisland, entries_data + [
        entry('-- Imaging Communication --', None, None),
        entry('N_copy = ', Ncopy, '', 'Number of times we need to copy (facet!) visibilities from each node'),
        entry("R_facet =", Rfacet / c.mega, "MB/s"),
        entry('R_distribute/node =', float(Rdistributenode / c.giga), "GB/s",
              'Facet visibility output and input rate of every node'),
        entry('R_distribute/island =', float(Rdistributeisland / c.giga), "GB/s",
              'Facet visibility output and input rate of every island'),
        entry('R_distribute/island2island =', float(Qspeed * tp.Rfacet_vis / Nnode * Nisland ** 2 / c.giga), "GB/s",
              'Maximum visibility exchange between pair of islands involved in all-to-all')
    ]


def calculate_working_set(telescope, band, pipeline, adjusts,
                          Nnode=1500, Nisland=56,
                          Rflop=15, Npredict=4, Mvisnode=64,
                          Ntt=3, Nfacet=7, Ndist_pptt=4):
    """
    :param Nnode: Number of nodes in cluster
    :param Nisland: Number of nodes in a compute island
    :param Qspeed: How much faster we need to process compared with ingest
       (to leave space for other pipelines to run)
    :param Npredict: Number of separate predicts we need to do
    :param Mvisnode: Memory per node to cache raw visibilities [GB]
    :param Ntt: Number of Taylor terms
    :param Nfacet: Number of facets (horizontal+vertical)
      Reduces working set and increases distribution, but requires more flops (phase rotation)
    :param Ndist_pptt: Degree of distribution in polarisation and Taylor terms.
      Reduces working set and increases distribution, but also increases communication.
    """

    adjusts_ = ("Nfacet=%d Ntt=%d" % (Nfacet, Ntt)) + adjusts
    config = PipelineConfig(telescope, pipeline, band, adjusts=adjusts_)
    if not config.is_valid()[0]:
        print(*config.is_valid()[1])
        return

    # Do calculations
    tp = config.calc_tel_params()
    Nsubbands = tp.Nsubbands
    if pipeline in [Pipelines.DPrepB, Pipelines.DPrepC]:
        Nsubbands = tp.Nf_out
    Qspeed = Rflop / float(tp.Rflop) * c.peta

    # Perform analyses
    entries = []

    def add_entry(title, val, unit='', doc=''):
        entries.append((title, val, unit, doc))

    entries.extend(visibility_analysis(tp, Qspeed, Nnode, Npredict, Mvisnode))
    entries.extend(imaging_analysis(tp, Qspeed, Nnode, Nsubbands))
    Mws_img, Rnode_img, Risland_img, img_entries = \
        imaging_data_analysis(tp, Qspeed, Nnode, Nisland, Nsubbands, Ndist_pptt)
    entries.extend(img_entries)
    Mws_cal, Rcal, cal_entries = calibration_analysis(tp, Qspeed, Nnode, Nsubbands, Ndist_pptt)
    entries.extend(cal_entries)

    # Compose summary
    add_entry('-- Summary --', None, None)
    Mworkset = Mvisnode * c.giga + 4 * Mws_img + Mws_cal
    add_entry('Mworkset/node = ', Mworkset / c.giga, "GB")
    add_entry('Rnode = ', (2 * Rnode_img + Rcal) / c.giga, "GB/s")
    add_entry('Risland = ', (2 * Risland_img + Rcal) / c.giga, "GB/s")
    add_entry('Rflop = ', Qspeed * float(tp.Rflop) / c.peta, 'Pflop/s')
    add_entry('Rflop/node = ', Qspeed * float(tp.Rflop) / Nnode / c.tera, 'Tflop/s')
    iapi.show_table("Working Set Analysis", *zip(*entries))

def calibration_analysis(tp, Qspeed, Nnode, Nsubbands, Ndist_pptt):
    # 2 sets (model+predicted) of visibilities per polarisation and baseline
    Ndist_ft, Nseq = _get_distribution(tp, Nnode, Nsubbands, Ndist_pptt)
    Tsnapwork = tp.Tsnap / Qspeed / (1 + tp.Nmajortotal) / Nseq

    Mcal = 2 * tp.Mvis * tp.Npp * tp.Nbl
    MGcal = Mcal * tp.Ncal_G_obs / Nnode
    MBcal = Mcal * tp.Ncal_B_obs / Nnode
    MIcal = Mcal * tp.Ncal_I_obs / Nnode
    RGcal_snap = Mcal * tp.Ncal_G_solve / Tsnapwork
    RBcal_snap = Mcal * tp.Ncal_B_solve / Tsnapwork
    RIcal_snap = Mcal * tp.Ncal_I_solve / Tsnapwork

    return MGcal + MBcal + MIcal, RGcal_snap + RBcal_snap + RIcal_snap, [
        entry('-- Calibration --', None, None),
        entry('M_cal = ', Mcal / c.mega, "MB", 'Size of input into an individual calibration "problem"'),
        entry('', ['Gain', 'Band', 'Ion'], "", ''),
        entry('N_cal = ', [tp.Ncal_G_obs, tp.Ncal_B_obs, tp.Ncal_I_obs], "k", 'Calibration problem count'),
        entry('M_cal/node = ', [MGcal / c.giga, MBcal / c.giga, MIcal / c.giga], "GB",
              'Calibration working set per node (assuming perfect distribution)'),
        entry('N_cal,snap = ', [tp.Ncal_G_solve, tp.Ncal_B_solve, tp.Ncal_I_solve], "",
              'Number of calibration problems overlapping a snapshot and subband'),
        entry('M_cal,snap = ',
              [Mcal * tp.Ncal_G_solve / c.giga, Mcal * tp.Ncal_B_solve / c.giga, Mcal * tp.Ncal_I_solve / c.giga], "GB",
              'Number of calibration problems overlapping a snapshot and subband'),
        entry('R_cal,snap = ', [RGcal_snap / c.giga, RBcal_snap / c.giga, RIcal_snap / c.giga], "GB/s",
              'Data rate per node for exchanging calibration problems after every snapshot')
    ]


def run_experiment_1():
    # Needs some refactoring methinks; idea would be to specify HPSOs instead of "letters".
    hpso_lookup = {'A' : HPSOs.hpso01,
                   'B' : HPSOs.hpso04c,  # This one probably not properly defined yet
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

    print('*** Processing task type %s => %s ***\n' % (task_letter, hpso))

    for subtask in HPSOs.hpso_subtasks[hpso]:
        print('subtask -> %s' % subtask)
        cfg = PipelineConfig(hpso=hpso, hpso_task=subtask)

        (valid, msgs) = cfg.is_valid()
        if not valid:
            print("Invalid configuration!")
            for msg in msgs:
                print(msg)
            raise AssertionError("Invalid config")
        tp = cfg.calc_tel_params()
        results = iapi._compute_results(cfg, results_map)  #TODO - refactor this method's parameter sequence
        print('Buffer ingest rate\t= %g TB/s' % results[0])
        print('Cache memory\t= %g TB' % results[1])
        print('Visibility IO rate\t= %g TB/s' % results[2])
        print('Compute Rate\t= %g PetaFLOP/s' % results[3])
        print()

    print('Done with running debug script.')

def build_performance_dict():
    sdp_scheduler = Scheduler()
    performance_lookup_filename = "performance_dict.data"
    if os.path.isfile(performance_lookup_filename):
        performance_dict = None
        with open(performance_lookup_filename, "rb") as f:
            performance_dict = pickle.load(f)
        sdp_scheduler.set_performance_dictionary(performance_dict)
    else:
        # Create a performance dictionary and write it to file
        performance_dict = sdp_scheduler.compute_performance_dictionary()
        with open(performance_lookup_filename, "wb") as f:
            pickle.dump(performance_dict, f, pickle.HIGHEST_PROTOCOL)
    print('done.')

def run_working_sets():
    calculate_working_set(telescope=Telescopes.SKA1_Low,
                            band=Bands.Low,
                            pipeline=Pipelines.ICAL,
                            adjusts="Tobs=6*3600")

def schedule_task_seq():
    sdp_scheduler = Scheduler()
    performance_lookup_filename = "performance_dict.data"
    if os.path.isfile(performance_lookup_filename):
        performance_dict = None
        with open(performance_lookup_filename, "rb") as f:
            performance_dict = pickle.load(f)
        sdp_scheduler.set_performance_dictionary(performance_dict)
    else:
        # Create a performance dictionary and write it to file
        performance_dict = sdp_scheduler.compute_performance_dictionary()
        with open(performance_lookup_filename, "wb") as f:
            pickle.dump(performance_dict, f, pickle.HIGHEST_PROTOCOL)

    seqL = ('B','A','A',)+('B',)*32+ ('A', 'A',) +('B',)*73 + ('A',) +('B',)*43
    seqM = ('B','G',)+('B',)*34 +('G','C','F',)+('B',)*110 +('F',)*91 +('G',)*2 + ('E',)*4 + ('D',)
    seqSmall = ('A', 'A')

    sequence_to_simulate = seqSmall

    task_list = sdp_scheduler.task_letters_to_sdp_task_list(sequence_to_simulate)
    '''To show how the tasks are created, can print the sequence of Task objects.'''

    #for task in task_list:
    #    print(task)

    sdp_low_flops_capacity = 13.8  # PetaFlops
    sdp_mid_flops_capacity = 12.1  # PetaFlops

    sdp_low_hot_buffer_size = 14e3  # TeraBytes
    sdp_mid_hot_buffer_size = 14e3  # TeraBytes

    sdp_low_cold_buffer_size = 15e3  # TeraBytes -- arbitrary for now
    sdp_mid_cold_buffer_size = 15e3  # TeraBytes -- arbitrary for now

    flops_cap = 5
    hotbuf_cap = 12e3
    coldbuf_cap = 15e3

    schedule = sdp_scheduler.schedule(task_list, flops_cap, hotbuf_cap, coldbuf_cap,
                                      assign_flops_fraction=0.5, assign_bw_fraction=0.5, max_nr_iterations=1000)
    last_preservation_timestamp = sorted(schedule.preserve_deltas.keys())[-1]
    max_t = last_preservation_timestamp
    print("SDP task sequence completes at t = %g hrs" % (max_t / 3600))

def test_tp():
    performance_dictionary = Scheduler.compute_performance_dictionary()

if __name__ == '__main__':
    #print(sys.path)
    #build_performance_dict()
    #run_working_sets()
    #schedule_task_seq()
    test_tp()