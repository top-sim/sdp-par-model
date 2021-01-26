#!/usr/bin/env python

from setuptools import setup

setup(name='sdp-par-model',
      version='0.6',
      description='SDP Parameteric Model',
      long_description=open('README.md').read(),
      author='Rosie Bolton, Francois Malan, Bojan Nikolic, Andreas Wicenec, Peter Wortmann',
      url='https://github.com/SKA-ScienceDataProcessor/sdp-par-model',
      license='Apache License Version 2.0',
      packages=['sdp_par_model', 'sdp_par_model.parameters', 'sdp_par_model.scheduling' ],
      test_suite="tests",
      tests_require=['pytest'],
      )
