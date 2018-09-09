import csv
import numpy as np

# Data type for parametric model output (extracted from CSV file).
#
# - name: name
# - tele: telescope
# - pipe: pipeline
# - tobs: observation time [s]
# - tpoint: total observation time for a single pointing [s]
# - texp: total observation time for all pointings [s]
# - rflop: compute rate [PFLOP/s]
# - mvis: input visibility size [TB]
# - mout: output data size [TB]

dtype_pmout = np.dtype([('name',   'U30'),
                        ('tele',   'U10'),
                        ('pipe',   'U10'),
                        ('tobs',   'f8' ),
                        ('tpoint', 'f8' ),
                        ('texp',   'f8' ),
                        ('rflop',  'f8' ),
                        ('mvis',   'f8' ),
                        ('mout',   'f8' )])

# Data type for project list.
# 
# - name: name of project ('HPSO-x')
# - tpoint: total observation time for a single pointing [s]
# - texp: total observation time for all pointings [s]
# - rflop_r: real-time compute rate [PFLOP/s]
# - rflop_b: batch compute rate [PFLOP/s]
# - rinp: input data rate [TB/s] (for visibilities)
# - rout: output data rate [TB/s] (for data produced continuously,
#         like averaged visibilities)
# - mout: output data size per pointing [TB] (for fixed-size data
#         objects, like images)

dtype_proj = np.dtype([('name',    'U10'),
                       ('tpoint',  'f8' ),
                       ('texp',    'f8' ),
                       ('rflop_r', 'f8' ),
                       ('rflop_b', 'f8' ),
                       ('rinp',    'f8' ),
                       ('rout',    'f8' ),
                       ('mout',    'f8' )])

# Data type for observation sequence.
# 
# - uid: unique ID of scheduling block
# - name: name of project ('HPSO-x')
# - tobs: observation time [s]
# - rflop_r: real-time compute rate [PFLOP/s]
# - rflop_b: batch compute rate [PFLOP/s]
# - minp: size of input data [TB]
# - mout: size of output data [TB]

dtype_seq = np.dtype([('uid',     'i8' ),
                      ('name',    'U10'),
                      ('tobs',    'f8' ),
                      ('rflop_r', 'f8' ),
                      ('rflop_b', 'f8' ),
                      ('minp',    'f8' ),
                      ('mout',    'f8' )])

def write_projects(ofile, projects):
    '''Write csv list of projects to file'''
    header = 'name,Tpoint,Texp,Rflop_r,Rflop_b,Rinp,Rout,Mout'
    fmt = '%s' + 7 * ',%e'
    np.savetxt(ofile, projects, header=header, fmt=fmt)

def read_projects(ifile):
    '''Read list of projects from file'''
    projects = np.loadtxt(ifile, dtype=dtype_proj, delimiter=',')
    return projects

def write_sequence(ofile, sequence):
    '''Write csv sequence of observations to file'''
    header = 'uid,name,Tobs,Rflop_r,Rflop_b,Minp,Mout'
    fmt = '%d,%s' + 5 * ',%f'
    np.savetxt(ofile, sequence, header=header, fmt=fmt)

def extract_projects(ifile, tele):
    '''Extract list of projects from parametric model output'''

    # Read data from parametric model output CSV file.

    with open(ifile, 'r') as f:
        r = csv.reader(f)
        l = next(r)
        pmout = np.zeros(len(l)-1, dtype=dtype_pmout)
        pmout['name'] = [x.split()[0] for x in l[1:]]
        for l in r:
            if l[0] == 'Telescope':
                pmout['tele'] = l[1:]
            elif l[0] == 'Pipeline':
                pmout['pipe'] = l[1:]
            elif l[0] == 'Observation Time [s]':
                pmout['tobs'] = l[1:]
            elif l[0] == 'Pointing Time [s]':
                pmout['tpoint'] = l[1:]
            elif l[0] == 'Total Time [s]':
                pmout['texp'] = l[1:]
            elif l[0] == 'Total Compute Requirement [PetaFLOP/s]':
                pmout['rflop'] = l[1:]
            elif l[0] == 'Visibility Buffer [PetaBytes]':
                pmout['mvis'] = l[1:]
            elif l[0] == 'Output size [TB]':
                pmout['mout'] = l[1:]

    # Convert visibility data size to TB.

    pmout['mvis'] *= 1000.0

    # Get list of HPSOs for this telescope by finding the ICAL
    # pipelines. Extract the properties which do not change between
    # pipelines (times and visibility data size).

    i = (pmout['tele'] == tele) & (pmout['pipe'] == 'ICAL')
    ninp = [x[:x.find('ICAL')] for x in pmout[i]['name']]
    name = ['HPSO-' + x for x in ninp]
    tobs = pmout[i]['tobs']
    tpoint = pmout[i]['tpoint']
    texp = pmout[i]['texp']
    mvis = pmout[i]['mvis']
    npoint = np.ceil(tobs/tpoint).astype(int)

    proj = np.zeros(len(name), dtype=dtype_proj)

    proj['name'] = name
    proj['tpoint'] = tpoint
    proj['texp'] = texp
    proj['rinp'] = mvis / tobs

    # For each HPSO, get sum of compute rate values for all pipelines,
    # the output data sizes for DPrep{A,B,C}, and the output
    # visibility rate for DPrepD. (Note only batch pipelines have
    # been included in the parametric model HPSO output so far.)

    for k, n in enumerate(ninp):
        i = (pmout['tele'] == tele) & (np.char.find(pmout['name'], n) == 0)
        j = (pmout['pipe'] == 'DPrepD')
        proj[k]['rflop_b'] = np.sum(pmout[i]['rflop'])
        proj[k]['rout'] = np.sum(pmout[i & j]['mout']) / tobs[k]
        proj[k]['mout'] = np.sum(pmout[i & ~j]['mout']) / npoint[k]
            
    return proj

def generate_sequence(projects, tsched, tseq, allow_short_tobs=False):
    '''Generate sequence of observations from list of projects'''

    if allow_short_tobs:

        # This case allows short observations for suitable projects:
        # if a project has tpoint less than tsched, then its
        # observations will be of length tpoint, otherwise they will
        # be of length tsched.

        # Probability for each project.

        texp = projects['texp']
        tpoint = projects['tpoint']
        tobs = np.where(tpoint < tsched, tpoint, tsched)
        prob = texp / tobs
        prob /= np.sum(prob)
        
        # Calculate number of scheduling blocks to draw. This number
        # is twice the number needed to get a long enough sequence,
        # given the average observation length. This should give a
        # long enough sequence in most draws.

        tobs_mean = np.sum(prob * tobs)
        nsched = np.ceil(2.0 * tseq / tobs_mean).astype(int)

        # Draw sequence until it is long enough, then truncate it to
        # the right length.

        ttot = 0.0
        while ttot < tseq:
            p = np.random.choice(projects, size=nsched, p=prob)
            tobs = np.where(p['tpoint'] < tsched, p['tpoint'], tsched)
            ttot = np.sum(tobs)

        tcum = np.cumsum(tobs)
        nsched = np.where(tcum >= tseq)[0][0] + 1
        p = p[0:nsched]
        tobs = tobs[0:nsched]
        
        # Calculate quantities for each entry.

        seq = np.zeros(nsched, dtype=dtype_seq)
        seq['uid'] = np.arange(nsched)
        seq['name'] = p['name']
        seq['tobs'] = tobs
        seq['rflop_r'] = p['rflop_r']
        seq['rflop_b'] = p['rflop_b']
        seq['minp'] = p['rinp'] * tobs
        seq['mout'] = p['rout'] * tobs + p['mout']

    else:

        # This case generates scheduling blocks which are all of
        # length tsched.

        # Probability for each project.

        texp = projects['texp']
        prob = texp / np.sum(texp)

        # Calculate number of scheduling blocks.

        nsched = np.ceil(tseq/tsched).astype(int)

        # Draw sequence and calculate quantities for each entry.

        p = np.random.choice(projects, size=nsched, p=prob)
        seq = np.zeros(nsched, dtype=dtype_seq)
        seq['uid'] = np.arange(nsched)
        seq['name'] = p['name']
        seq['tobs'] = tsched
        seq['rflop_r'] = p['rflop_r']
        seq['rflop_b'] = p['rflop_b']
        seq['minp'] = p['rinp'] * tsched
        seq['mout'] = p['rout'] * tsched + p['mout'] * tsched/p['tpoint']

    return seq
