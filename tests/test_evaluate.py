
from sdp_par_model.config import PipelineConfig
from sdp_par_model.parameters.container import ParameterContainer
from sdp_par_model.parameters.definitions import Telescopes, Bands, Products
from sdp_par_model.evaluate import *

import unittest
import io

import time

from sympy import Symbol, Min, Max, log, sign, floor

class EvaluateTests(unittest.TestCase):

    def test_minimise_expr(self):

        # Test optimising a simple expression
        xi = Symbol("xi", integer=True)
        yi = Symbol("yi", integer=True)
        z = Symbol("z")
        lower_bound = {'xi': 0, 'yi': 0, 'z': 0}
        upper_bound = {'xi': 10, 'yi': 5}
        upper_bound = {'xi': 10, 'yi': 5, 'z': 15 }

        tp = ParameterContainer()
        def test_xyz(expr):
            return minimise_parameters(tp, expression=expr,
                                       lower_bound=lower_bound, upper_bound=upper_bound)
        self.assertDictEqual(test_xyz(xi+yi), {'xi': 0, 'yi': 0})
        self.assertDictEqual(test_xyz(xi-yi), {'xi': 0, 'yi': 5})
        self.assertAlmostEqual(test_xyz(xi+z)['z'], 0, places=3)
        self.assertAlmostEqual(test_xyz(xi-z)['z'], 15, places=3)
        self.assertDictEqual(test_xyz(xi+(z-5)**2), {'xi': 0, 'z': 5})
        self.assertDictEqual(test_xyz((xi-2)**2+(yi-3)**2), {'xi':2, 'yi':3})

    def test_minimise_parameters(self):

        unopt_cfg = PipelineConfig(telescope=Telescopes.SKA1_Mid,
                                   band=Bands.Mid1,
                                   pipeline=Pipelines.ICAL)

        # Optimise manually, set as adjustment
        unopt_tp = unopt_cfg.calc_tel_params(optimize_expression=None)
        opt = minimise_parameters(unopt_tp)
        opt_tp = unopt_cfg.calc_tel_params(adjusts=opt)

        # Compare with auto-optimised version (uses subst internally)
        opt_tp2 = unopt_cfg.calc_tel_params()

        self.assertAlmostEqual(float(opt_tp.Rflop), float(opt_tp2.Rflop), delta=10)
        self.assertAlmostEqual(float(opt_tp.Nfacet), float(opt_tp2.Nfacet), places=4)
        self.assertAlmostEqual(float(opt_tp.products[Products.DFT]['Rflop']),
                               float(opt_tp2.products[Products.DFT]['Rflop']), places=3)
