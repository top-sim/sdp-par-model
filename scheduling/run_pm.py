#!/usr/bin/env python3
#
# Run parametric model to generate CSV output for the HPSOs.

import sys
sys.path.append('..')

from sdp_par_model import reports
from sdp_par_model.parameters.definitions import HPSOs

ofile = 'hpsos.csv'
parallel = 0

reports.write_csv_hpsos(ofile, HPSOs.all_hpsos, parallel=parallel,
                        verbose=True)
